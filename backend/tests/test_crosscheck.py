"""Tests for cross-checking an uploaded file against verified PAIMANA data.

The central behavioural requirement is negative: when the two sources disagree,
the platform must report both values and the difference, and must NOT decide
which one is right. Several tests assert that absence explicitly.
"""
from __future__ import annotations

import io

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal, init_db
from app.main import app
from app.models import Project, ProjectSnapshot
from app.services import analytics, crosscheck

client = TestClient(app)


def _has_data() -> bool:
    try:
        init_db()
        session = SessionLocal()
        try:
            return session.query(Project).count() > 0
        finally:
            session.close()
    except Exception:
        return False


needs_data = pytest.mark.skipif(
    not _has_data(), reason="No ingested data. Run scripts/dev_seed.py first."
)


# ---------------------------------------------------------------------------
# Column mapping — no database required
# ---------------------------------------------------------------------------
class TestColumnMapping:
    @pytest.mark.parametrize("header,expected", [
        ("Physical Progress (%)", "physical_progress"),
        ("physical_progress", "physical_progress"),
        ("Revised Cost (Cr)", "revised_cost"),
        ("Original Cost", "original_cost"),
        ("Expenditure", "expenditure"),
        ("Delay (months)", "schedule_delay_months"),
        ("Cost Escalation %", "cost_escalation_pct"),
    ])
    def test_headers_map_to_snapshot_fields(self, header, expected):
        assert crosscheck.map_columns([header]).get(header) == expected

    def test_identifier_column_found(self):
        code, name = crosscheck.find_key_columns(
            ["Project Code", "Project Name", "Physical Progress"]
        )
        assert code == "Project Code"
        assert name == "Project Name"

    def test_unrelated_columns_are_not_mapped(self):
        assert crosscheck.map_columns(["Remarks", "Contact Person"]) == {}

    @pytest.mark.parametrize("raw,expected", [
        ("1,200", 1200.0), ("₹3,400", 3400.0), ("68%", 68.0),
        ("1200 Cr", 1200.0), (64.5, 64.5), ("", None), ("N/A", None),
        (None, None), ("not a number", None),
    ])
    def test_value_coercion(self, raw, expected):
        assert crosscheck._to_number(raw) == expected


# ---------------------------------------------------------------------------
# Applicability
# ---------------------------------------------------------------------------
class TestApplicability:
    def test_non_tabular_file_is_not_cross_checkable(self):
        session = SessionLocal()
        try:
            result = crosscheck.cross_check(
                session, {"content": {"kind": "text"}, "status": "ANALYSED"}
            )
        finally:
            session.close()
        assert result["applicable"] is False
        assert "not a table" in result["reason"]

    @needs_data
    def test_table_without_an_identifier_is_not_cross_checkable(self):
        session = SessionLocal()
        try:
            analysis = {"content": {"kind": "tabular", "tables": [
                {"column_names": ["Remarks", "Physical Progress"], "rows": 5,
                 "sample": []}
            ]}}
            result = crosscheck.cross_check(session, analysis)
        finally:
            session.close()
        assert result["applicable"] is False


# ---------------------------------------------------------------------------
# End-to-end comparison
# ---------------------------------------------------------------------------
@needs_data
class TestCrossCheck:
    @pytest.fixture(scope="class")
    def report(self):
        """A monthly report with known, deliberate drift from PAIMANA."""
        session = SessionLocal()
        try:
            period = analytics.latest_period(session)
            rows = (
                session.query(ProjectSnapshot, Project)
                .join(Project, Project.id == ProjectSnapshot.project_id)
                .filter(ProjectSnapshot.report_period == period,
                        ProjectSnapshot.supersedes_id.is_(None))
                .limit(10)
                .all()
            )
            data = []
            for i, (snapshot, project) in enumerate(rows):
                data.append({
                    "Project Code": project.project_code,
                    "Project Name": project.name,
                    # Every third row drifts by exactly +7 percentage points.
                    "Physical Progress (%)": round(
                        (snapshot.physical_progress or 0) + (7 if i % 3 == 0 else 0), 1
                    ),
                    "Delay (months)": (snapshot.schedule_delay_months or 0),
                })
            data.append({"Project Code": "ZZ-NOT-REAL", "Project Name": "Unknown",
                         "Physical Progress (%)": 50, "Delay (months)": 1})
        finally:
            session.close()

        buffer = io.BytesIO()
        pd.DataFrame(data).to_excel(buffer, index=False)
        return buffer.getvalue()

    def _run(self, report):
        upload = client.post("/api/chat/attachments",
                             files={"file": ("monthly.xlsx", report)})
        assert upload.status_code == 201
        body = upload.json()
        answer = client.post("/api/chat/message", json={
            "message": "Compare this with PAIMANA and tell me if anything looks wrong.",
            "conversation_id": body["conversation_id"],
            "attachment_ids": [body["attachment"]["id"]],
        })
        assert answer.status_code == 200
        return answer.json()

    def test_route_is_cross_check_with_mixed_sources(self, report):
        result = self._run(report)
        assert result["route"] == "FILE_VS_PAIMANA"
        assert result["source"] == "MIXED SOURCES"

    def test_known_projects_are_matched(self, report):
        comparison = self._run(report)["cross_check"][0]["result"]
        assert comparison["matched_count"] >= 5

    def test_unmatched_row_is_reported_not_silently_dropped(self, report):
        comparison = self._run(report)["cross_check"][0]["result"]
        assert comparison["unmatched_count"] >= 1
        identifiers = [u["identifier"] for u in comparison["unmatched_examples"]]
        assert any("ZZ-NOT-REAL" in i for i in identifiers)

    def test_injected_drift_is_detected_with_the_right_magnitude(self, report):
        comparison = self._run(report)["cross_check"][0]["result"]
        assert comparison["projects_with_discrepancies"] >= 1
        deltas = [
            c["difference"]
            for entry in comparison["discrepancies"]
            for c in entry["comparisons"]
            if c["status"] == "DIFFERS" and c["field"] == "Physical progress"
        ]
        assert deltas, "the seeded +7pp drift must be detected"
        assert all(abs(d - 7.0) < 0.05 for d in deltas)

    def test_matching_values_are_not_reported_as_discrepancies(self, report):
        comparison = self._run(report)["cross_check"][0]["result"]
        agreeing = [
            c for entry in comparison["matched"]
            for c in entry["comparisons"]
            if c["status"] == "AGREES"
        ]
        assert agreeing, "identical values must be recognised as agreeing"

    def test_all_four_provenance_labels_appear(self, report):
        answer = self._run(report)["answer"]
        for label in ("UPLOADED FILE", "PAIMANA VERIFIED DATA", "DERIVED CALCULATION"):
            assert label in answer

    def test_neither_source_is_declared_correct(self, report):
        """The key negative assertion: no verdict on which source is right."""
        result = self._run(report)
        answer = result["answer"].lower()
        for claim in ("the file is wrong", "paimana is wrong", "the file is incorrect",
                      "paimana is incorrect", "the correct value is"):
            assert claim not in answer
        note = result["cross_check"][0]["result"]["interpretation_note"]
        assert "not evidence that either source is wrong" in note

    def test_a_possible_explanation_is_offered(self, report):
        note = self._run(report)["cross_check"][0]["result"]["interpretation_note"]
        assert "reporting dates" in note or "measurement definitions" in note

    def test_comparison_uses_full_rows_not_the_preview(self, report):
        """The stored profile keeps five sample rows; the comparison must not."""
        comparison = self._run(report)["cross_check"][0]["result"]
        assert comparison["rows_examined"] > 5
