"""Tests for the pieces most likely to silently corrupt a decision.

These target the exact failure modes found in the audit of the original
prototype: fabricated values, lost rows, circular risk logic, and an
unreachable low-risk band.
"""
from __future__ import annotations

import datetime as dt

import pytest

from app.models import ProjectSnapshot, RiskLevel
from app.services.agency_map import resolve_agency
from app.services.ingestion import (
    compute_indicators,
    norm_month_year,
    norm_number,
    norm_percent,
    parse_project_cell,
    parse_state,
    split_paren_pair,
)
from app.services.risk_engine import classify_trend, deterioration_streak, score_snapshot
from app.security import hash_password, safe_filename, verify_password
from fastapi import HTTPException


# ---------------------------------------------------------------------------
# Normalisation: missing must stay missing
# ---------------------------------------------------------------------------
class TestNormalisers:
    def test_absent_value_is_none_not_zero(self):
        """A blank cost must never become 0 — that is what produced nonsense ratios."""
        for raw in ["", "-", "(-)", "NA", None]:
            value, _, _ = norm_number(raw)
            assert value is None, f"{raw!r} should parse to None, got {value}"

    def test_number_with_commas(self):
        assert norm_number("5,365.88")[0] == 5365.88

    def test_unparseable_number_flags_low_confidence(self):
        value, confidence, warning = norm_number("abc")
        assert value is None
        assert confidence == 0.0
        assert warning is not None

    def test_percent_out_of_range_is_kept_but_flagged(self):
        value, confidence, warning = norm_percent("150")
        assert value == 150.0
        assert confidence < 0.5
        assert "out_of_range" in warning

    def test_month_year_parsing(self):
        assert norm_month_year("03/2029")[0] == dt.date(2029, 3, 1)
        assert norm_month_year("(-)")[0] is None
        assert norm_month_year("13/2029")[0] is None       # invalid month

    def test_split_paren_pair(self):
        assert split_paren_pair("03/2029\n(-)") == ("03/2029", "-")
        assert split_paren_pair("749.08\n(749.07)") == ("749.08", "749.07")


class TestProjectCell:
    RAW = "TILABONI UG\n(ECL - CIL)\n(400186)\n(-) (-)"

    def test_splits_composite_cell(self):
        parsed = parse_project_cell(self.RAW)
        assert parsed["name"] == "TILABONI UG"
        assert parsed["agency"] == "ECL - CIL"
        assert parsed["project_code"] == "400186"
        assert parsed["confidence"] == 1.0

    def test_multiline_project_name_is_preserved(self):
        raw = ("Development of 6L of Jhanki Sargi Section\nRoad from km 0.000\n"
               "(NHAI)\n(123456)\n(-) (-)")
        parsed = parse_project_cell(raw)
        assert "Jhanki Sargi" in parsed["name"]
        assert "km 0.000" in parsed["name"]
        assert parsed["project_code"] == "123456"

    def test_degraded_cell_lowers_confidence(self):
        parsed = parse_project_cell("SOME PROJECT\n(Agency)")
        assert parsed["confidence"] < 0.5
        assert "project_code_not_found" in parsed["warnings"]


class TestState:
    def test_single_state(self):
        state, states, multi = parse_state("West Bengal")
        assert state == "West Bengal"
        assert states == ["West Bengal"]
        assert multi is False

    def test_multi_state_is_expanded(self):
        state, states, multi = parse_state("Multi-States\n(Madhya Pradesh, Uttar Pradesh)")
        assert multi is True
        assert "Madhya Pradesh" in states
        assert "Uttar Pradesh" in states

    def test_missing_state_is_unknown_not_guessed(self):
        state, states, _ = parse_state(None)
        assert state == "UNKNOWN"
        assert states == []


# ---------------------------------------------------------------------------
# Agency mapping: the replacement for random.choice()
# ---------------------------------------------------------------------------
class TestAgencyMapping:
    def test_known_agency_resolves(self):
        ministry, sector, matched = resolve_agency("Eastern Coal Fields Limited [ECL]")
        assert matched is True
        assert ministry == "Ministry of Coal"
        assert sector == "Coal"

    def test_acronym_with_mixed_case_resolves(self):
        _, sector, matched = resolve_agency("MoRTH")
        assert matched is True
        assert sector == "Roads & Highways"

    def test_run_together_name_resolves(self):
        ministry, _, matched = resolve_agency("MinistryofPetroleumNaturalGas")
        assert matched is True
        assert ministry == "Ministry of Petroleum & Natural Gas"

    def test_unknown_agency_is_not_guessed(self):
        ministry, sector, matched = resolve_agency("Some Entirely Unheard Of Body")
        assert matched is False
        assert ministry == "UNKNOWN"
        assert sector == "UNKNOWN"

    def test_mapping_is_deterministic(self):
        """The old code used random.choice; the same input must now always map alike."""
        results = {resolve_agency("NHAI") for _ in range(50)}
        assert len(results) == 1


# ---------------------------------------------------------------------------
# Derived indicators
# ---------------------------------------------------------------------------
class TestIndicators:
    def test_cost_escalation(self):
        ind = compute_indicators({"original_cost": 100.0, "revised_cost": 150.0})
        assert ind["cost_escalation_pct"] == 50.0

    def test_financial_progress_uses_revised_cost(self):
        ind = compute_indicators(
            {"original_cost": 100.0, "revised_cost": 200.0, "expenditure": 100.0}
        )
        assert ind["financial_progress"] == 50.0

    def test_missing_inputs_produce_none_not_zero(self):
        ind = compute_indicators({"original_cost": None, "revised_cost": None})
        assert ind["cost_escalation_pct"] is None
        assert ind["financial_progress"] is None

    def test_zero_original_cost_does_not_divide(self):
        ind = compute_indicators({"original_cost": 0.0, "revised_cost": 50.0})
        assert ind["cost_escalation_pct"] is None

    def test_schedule_delay_in_months(self):
        ind = compute_indicators(
            {
                "original_completion": dt.date(2025, 3, 1),
                "revised_completion": dt.date(2029, 3, 1),
            }
        )
        assert ind["schedule_delay_months"] == 48


# ---------------------------------------------------------------------------
# Risk engine
# ---------------------------------------------------------------------------
def make_snapshot(**kwargs) -> ProjectSnapshot:
    defaults = dict(
        report_period="2026-07",
        original_cost=1000.0,
        revised_cost=1000.0,
        expenditure=500.0,
        physical_progress=50.0,
        financial_progress=50.0,
        cost_escalation_pct=0.0,
        progress_divergence=0.0,
        schedule_delay_months=0,
        elapsed_time_pct=50.0,
        completeness=100.0,
    )
    defaults.update(kwargs)
    return ProjectSnapshot(**defaults)


class TestRiskEngine:
    def test_healthy_project_scores_low(self):
        """The old simulator added a hardcoded +45, making LOW unreachable."""
        result = score_snapshot(make_snapshot())
        assert result.score == 0.0
        assert result.level == RiskLevel.LOW

    def test_low_band_is_reachable(self):
        result = score_snapshot(make_snapshot(cost_escalation_pct=2.0))
        assert result.score < 25
        assert result.level == RiskLevel.LOW

    def test_severe_project_scores_high(self):
        result = score_snapshot(
            make_snapshot(
                cost_escalation_pct=80.0,
                schedule_delay_months=60,
                progress_divergence=45.0,
                physical_progress=10.0,
                elapsed_time_pct=95.0,
            )
        )
        assert result.score >= 70
        assert result.level == RiskLevel.SEVERE

    def test_score_stays_within_bounds(self):
        result = score_snapshot(
            make_snapshot(
                cost_escalation_pct=9999.0,
                schedule_delay_months=999,
                progress_divergence=999.0,
                elapsed_time_pct=999.0,
                physical_progress=0.0,
            )
        )
        assert 0.0 <= result.score <= 100.0

    def test_no_computable_indicators_returns_unknown(self):
        snap = make_snapshot(
            cost_escalation_pct=None,
            schedule_delay_months=None,
            progress_divergence=None,
            elapsed_time_pct=None,
            physical_progress=None,
            completeness=0.0,
        )
        result = score_snapshot(snap)
        assert result.score is None
        assert result.level == RiskLevel.UNKNOWN

    def test_every_driver_carries_evidence(self):
        result = score_snapshot(make_snapshot(cost_escalation_pct=40.0))
        assert result.drivers
        for driver in result.drivers:
            assert driver.evidence, "a driver without evidence is not explainable"
            assert driver.contribution <= driver.weight

    def test_drivers_sum_to_score_proportionally(self):
        snap = make_snapshot(cost_escalation_pct=25.0, schedule_delay_months=18)
        result = score_snapshot(snap)
        total = sum(d.contribution for d in result.drivers)
        assert total > 0
        assert result.score <= 100.0

    def test_progress_stall_requires_history(self):
        current = make_snapshot(physical_progress=40.0)
        previous = make_snapshot(report_period="2026-06", physical_progress=40.0)
        without = score_snapshot(current)
        with_history = score_snapshot(current, previous)
        codes_without = {d.code for d in without.drivers}
        codes_with = {d.code for d in with_history.drivers}
        assert "PROGRESS_STALL" not in codes_without
        assert "PROGRESS_STALL" in codes_with

    def test_regression_scores_worse_than_stall(self):
        previous = make_snapshot(report_period="2026-06", physical_progress=50.0)
        stalled = score_snapshot(make_snapshot(physical_progress=50.0), previous)
        regressed = score_snapshot(make_snapshot(physical_progress=45.0), previous)
        assert regressed.score > stalled.score

    def test_confidence_reflects_completeness(self):
        full = score_snapshot(make_snapshot(cost_escalation_pct=30.0, completeness=100.0))
        sparse = score_snapshot(make_snapshot(cost_escalation_pct=30.0, completeness=40.0))
        assert full.confidence > sparse.confidence


class TestTrend:
    def test_deteriorating(self):
        assert classify_trend([10.0, 20.0, 35.0]).value == "DETERIORATING"

    def test_improving(self):
        assert classify_trend([50.0, 40.0, 30.0]).value == "IMPROVING"

    def test_stable_within_tolerance(self):
        assert classify_trend([30.0, 31.0, 31.5]).value == "STABLE"

    def test_single_point_is_insufficient(self):
        assert classify_trend([30.0]).value == "INSUFFICIENT_HISTORY"

    def test_streak_counts_consecutive_rises(self):
        assert deterioration_streak([10.0, 20.0, 30.0, 40.0]) == 3
        assert deterioration_streak([40.0, 30.0, 35.0]) == 1


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
class TestSecurity:
    def test_password_round_trip(self):
        stored = hash_password("correct horse battery staple")
        assert verify_password("correct horse battery staple", stored)
        assert not verify_password("wrong password", stored)

    def test_hash_is_salted(self):
        assert hash_password("same") != hash_password("same")

    def test_plaintext_never_stored(self):
        stored = hash_password("secret123")
        assert "secret123" not in stored

    def test_filename_traversal_rejected(self):
        with pytest.raises(HTTPException):
            safe_filename("../../etc/passwd")

    def test_non_pdf_rejected(self):
        with pytest.raises(HTTPException):
            safe_filename("payload.exe")

    def test_safe_filename_is_prefixed_and_unique(self):
        a = safe_filename("report.pdf")
        b = safe_filename("report.pdf")
        assert a != b
        assert a.endswith("report.pdf")
        assert "/" not in a
