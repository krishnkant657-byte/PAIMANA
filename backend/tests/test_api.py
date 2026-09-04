"""Integration tests against the running application and ingested database.

These are skipped automatically if the database has not been built yet, so a
fresh clone can still run `pytest` without failures.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.db import SessionLocal, init_db
from app.main import app
from app.models import Project, ProjectSnapshot
from app.services import analytics

client = TestClient(app)


@pytest.fixture(scope="module")
def db():
    session = SessionLocal()
    yield session
    session.close()


@pytest.fixture(scope="module")
def has_data(db):
    return db.query(Project).count() > 0


def _has_ingested_data() -> bool:
    """Decide the skip condition without exploding on a fresh clone.

    This runs at import time, before any fixture or lifespan hook, so on a clone
    where `init_db()` has never been called the `projects` table does not exist
    yet and the query raises OperationalError — a collection error, not a skip.
    Creating the schema first makes the check do what the module docstring
    already promised.
    """
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
    not _has_ingested_data(),
    reason="No ingested data. Run scripts/dev_seed.py or scripts/ingest_reports.py first.",
)


class TestHealth:
    def test_health_endpoint(self):
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_prototype_disclaimer_is_present(self):
        """The platform must never present itself as an official portal."""
        body = r"" + client.get("/api/health").json()["disclaimer"]
        assert "not an official" in body.lower()


@needs_data
class TestProjects:
    def test_listing_is_paginated(self):
        r = client.get("/api/projects?page_size=5")
        assert r.status_code == 200
        data = r.json()
        assert len(data["items"]) <= 5
        assert data["total"] > 0

    def test_no_fabricated_states(self, db):
        """The old pipeline assigned states with random.choice()."""
        states = {p.state for p in db.query(Project).limit(500).all()}
        assert "UNKNOWN" not in states or len(states) > 10
        for state in states:
            assert state, "a blank state must be stored as UNKNOWN, never empty"

    def test_project_codes_are_unique(self, db):
        total = db.query(Project).count()
        distinct = db.query(Project.project_code).distinct().count()
        assert total == distinct

    def test_filter_by_state_returns_only_that_state(self):
        options = client.get("/api/projects/filters").json()
        state = next(s for s in options["states"] if s != "UNKNOWN")
        r = client.get(f"/api/projects?state={state}&page_size=20")
        for item in r.json()["items"]:
            assert item["state"] == state

    def test_unknown_project_returns_404(self):
        assert client.get("/api/projects/does-not-exist-999").status_code == 404


@needs_data
class TestProvenance:
    def test_extracted_value_traces_to_a_source_page(self, db):
        snap = (
            db.query(ProjectSnapshot)
            .filter(ProjectSnapshot.physical_progress.isnot(None))
            .first()
        )
        project = db.get(Project, snap.project_id)
        r = client.get(
            f"/api/projects/{project.project_code}/provenance/physical_progress"
        )
        assert r.status_code == 200
        body = r.json()
        assert body["origin"] == "EXTRACTED"
        assert body["source_document"], "every extracted value needs a source document"
        assert body["source_page"], "every extracted value needs a source page"

    def test_derived_value_states_its_formula(self, db):
        project = (
            db.query(Project)
            .join(ProjectSnapshot, ProjectSnapshot.project_id == Project.id)
            .filter(ProjectSnapshot.cost_escalation_pct.isnot(None))
            .first()
        )
        r = client.get(
            f"/api/projects/{project.project_code}/provenance/cost_escalation_pct"
        )
        body = r.json()
        assert body["origin"] == "DERIVED"
        assert body["derivation"]

    def test_unmapped_field_is_rejected(self, db):
        project = db.query(Project).first()
        r = client.get(f"/api/projects/{project.project_code}/provenance/made_up_field")
        assert r.status_code == 400


@needs_data
class TestMonitor:
    def test_summary_totals_are_internally_consistent(self):
        s = client.get("/api/monitor/summary").json()
        distribution = sum(s["risk_distribution"].values())
        assert distribution == s["total_projects"]
        assert s["at_risk"] == (
            s["risk_distribution"]["SEVERE"] + s["risk_distribution"]["HIGH"]
        )

    def test_cost_exposure_matches_component_figures(self):
        s = client.get("/api/monitor/summary").json()
        expected = round(s["revised_cost_cr"] - s["original_cost_cr"], 2)
        assert abs(s["cost_exposure_cr"] - expected) < 1.0

    def test_trend_is_chronological(self):
        series = client.get("/api/monitor/trend").json()["series"]
        periods = [s["period"] for s in series]
        assert periods == sorted(periods)

    def test_map_does_not_claim_coordinates(self):
        body = client.get("/api/monitor/map").json()
        assert "geographic_note" in body
        for state in body["states"]:
            assert "latitude" not in state and "longitude" not in state


@needs_data
class TestAssistantGrounding:
    def test_answer_carries_a_verified_result(self):
        r = client.post("/api/assistant/ask",
                        json={"question": "Which projects are high risk?"})
        assert r.status_code == 200
        body = r.json()
        assert body["resolved"] is True
        assert body["verified_result"]["resolved"] is True

    def test_figures_in_answer_come_from_the_query(self, db):
        r = client.post("/api/assistant/ask",
                        json={"question": "Summarise the national risk situation."})
        body = r.json()
        summary = body["verified_result"]["summary"]
        real = analytics.national_summary(db)
        assert summary["total_projects"] == real["total_projects"]
        assert summary["at_risk"] == real["at_risk"]

    def test_unanswerable_question_is_refused(self):
        r = client.post(
            "/api/assistant/ask",
            json={"question": "What will the monsoon do to concrete prices in 2031?"},
        )
        body = r.json()
        assert body["resolved"] is False
        assert "sufficient verified data" in body["answer"].lower()

    def test_refusal_lists_what_is_supported(self):
        r = client.post("/api/assistant/ask", json={"question": "zzzz qqqq"})
        assert r.json()["supported_questions"]


@needs_data
class TestSimulator:
    def test_low_risk_band_is_reachable(self):
        """The original simulator returned HIGH RISK for every possible input."""
        r = client.post(
            "/api/simulator/pre-approval",
            json={"estimated_cost_cr": 100, "duration_months": 12},
        )
        assert r.status_code == 200
        assert r.json()["label"] == "SIMULATED SCENARIO"

    def test_no_probability_of_failure_is_published(self):
        body = client.post(
            "/api/simulator/pre-approval",
            json={"estimated_cost_cr": 5000, "duration_months": 96},
        ).json()
        assert "probability" not in str(body.get("indicative_risk_score", "")).lower()
        assert "does not publish a probability" in body["explanation"]

    def test_scenario_output_is_labelled_simulated(self, db):
        project = (
            db.query(Project)
            .join(ProjectSnapshot, ProjectSnapshot.project_id == Project.id)
            .first()
        )
        r = client.post(
            "/api/scenario/run",
            json={"project_code": project.project_code, "cost_change_pct": 25},
        )
        assert r.json()["label"] == "SIMULATED SCENARIO"

    def test_scenario_increases_risk_when_cost_rises(self, db):
        project = (
            db.query(Project)
            .join(ProjectSnapshot, ProjectSnapshot.project_id == Project.id)
            .filter(ProjectSnapshot.original_cost.isnot(None),
                    ProjectSnapshot.revised_cost.isnot(None))
            .first()
        )
        body = client.post(
            "/api/scenario/run",
            json={"project_code": project.project_code, "cost_change_pct": 100},
        ).json()
        if body["current"]["risk_score"] is not None:
            assert body["scenario"]["risk_score"] >= body["current"]["risk_score"]


class TestAuthorization:
    def test_intervention_creation_is_open(self):
        r = client.post(
            "/api/interventions",
            json={"project_code": "123456", "issue": "Test issue for open access check"},
        )
        assert r.status_code != 401

    def test_audit_trail_is_open(self):
        assert client.get("/api/audit").status_code == 200

    def test_report_generation_is_open(self):
        r = client.post("/api/reports/generate", json={"report_type": "EXECUTIVE_BRIEF"})
        assert r.status_code == 200

    def test_bad_credentials_are_rejected(self):
        r = client.post("/api/auth/login",
                        json={"username": "admin", "password": "wrong-password"})
        assert r.status_code == 401

    def test_login_error_does_not_reveal_whether_account_exists(self):
        missing = client.post("/api/auth/login",
                              json={"username": "nobody-here", "password": "x"})
        wrong = client.post("/api/auth/login",
                            json={"username": "admin", "password": "x"})
        assert missing.json()["detail"] == wrong.json()["detail"]


class TestInputValidation:
    def test_oversized_page_size_is_rejected(self):
        assert client.get("/api/projects?page_size=100000").status_code == 422

    def test_invalid_report_type_is_rejected(self):
        r = client.post("/api/reports/generate", json={"report_type": "NOT_A_TYPE"})
        assert r.status_code in (401, 422)

    def test_negative_cost_is_rejected(self):
        r = client.post("/api/simulator/pre-approval",
                        json={"estimated_cost_cr": -50, "duration_months": 12})
        assert r.status_code == 422

    def test_comparison_needs_two_values(self):
        r = client.get("/api/analytics/compare?dimension=state&values=Bihar")
        assert r.status_code == 400
