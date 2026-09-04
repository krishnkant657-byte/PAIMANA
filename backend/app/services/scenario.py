"""Scenario Lab and Pre-Approval Simulator.

Both features re-run the *same deterministic risk engine* used everywhere else
against modified inputs. There is no separate "prediction" model and no
hardcoded offset — the previous implementation added a flat +45 to every score,
which made the low-risk band unreachable and every input look dangerous.

Outputs are labelled SIMULATED SCENARIO and carry an explicit statement of what
the result does and does not mean.
"""
from __future__ import annotations

import copy
import datetime as dt

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import Project, ProjectSnapshot, RiskLevel
from .risk_engine import cohort_percentile, cohort_stats, score_snapshot

SIMULATION_LABEL = "SIMULATED SCENARIO"
MIN_COHORT = 8


class _Detached:
    """Lightweight stand-in for a snapshot so we never mutate a stored row."""

    def __init__(self, source: ProjectSnapshot | None = None, **kwargs):
        fields = [
            "original_cost", "revised_cost", "expenditure", "physical_progress",
            "financial_progress", "approval_date", "start_date", "original_completion",
            "revised_completion", "cost_escalation_pct", "progress_divergence",
            "schedule_delay_months", "elapsed_time_pct", "completeness", "report_period",
            "risk_drivers",
        ]
        for f in fields:
            setattr(self, f, getattr(source, f, None) if source is not None else None)
        for k, v in kwargs.items():
            setattr(self, k, v)


def _recompute(s: _Detached, as_of: dt.date | None = None) -> _Detached:
    effective = s.revised_cost if s.revised_cost not in (None, 0) else s.original_cost
    s.financial_progress = (
        round(100.0 * s.expenditure / effective, 2)
        if s.expenditure is not None and effective not in (None, 0)
        else None
    )
    s.cost_escalation_pct = (
        round(100.0 * (s.revised_cost - s.original_cost) / s.original_cost, 2)
        if s.original_cost not in (None, 0) and s.revised_cost is not None
        else None
    )
    s.progress_divergence = (
        round(s.financial_progress - s.physical_progress, 2)
        if s.financial_progress is not None and s.physical_progress is not None
        else None
    )
    if s.original_completion and s.revised_completion:
        s.schedule_delay_months = (
            (s.revised_completion.year - s.original_completion.year) * 12
            + (s.revised_completion.month - s.original_completion.month)
        )
    start = s.start_date or s.approval_date
    target = s.revised_completion or s.original_completion
    ref = as_of or dt.date.today()
    if start and target and target > start:
        total = (target.year - start.year) * 12 + (target.month - start.month)
        elapsed = (ref.year - start.year) * 12 + (ref.month - start.month)
        s.elapsed_time_pct = (
            round(max(0.0, min(100.0 * elapsed / total, 999.0)), 2) if total > 0 else None
        )
    return s


def _add_months(d: dt.date | None, months: int) -> dt.date | None:
    if d is None:
        return None
    total = d.year * 12 + (d.month - 1) + months
    return dt.date(total // 12, total % 12 + 1, 1)


# ---------------------------------------------------------------------------
# Scenario Lab — what-if on an existing project
# ---------------------------------------------------------------------------
def run_scenario(
    db: Session,
    project: Project,
    *,
    cost_change_pct: float = 0.0,
    schedule_change_days: int = 0,
    progress_change_pp: float = 0.0,
    expenditure_change_pct: float = 0.0,
) -> dict:
    snaps = (
        db.query(ProjectSnapshot)
        .filter_by(project_id=project.id, supersedes_id=None)
        .order_by(ProjectSnapshot.report_period)
        .all()
    )
    if not snaps:
        return {"error": "This project has no snapshots to run a scenario against."}

    current = snaps[-1]
    previous = snaps[-2] if len(snaps) > 1 else None
    baseline = score_snapshot(current, previous)

    modified = _Detached(current)
    if modified.revised_cost is not None:
        modified.revised_cost = round(modified.revised_cost * (1 + cost_change_pct / 100.0), 2)
    if modified.expenditure is not None:
        modified.expenditure = round(
            modified.expenditure * (1 + expenditure_change_pct / 100.0), 2
        )
    if modified.physical_progress is not None:
        modified.physical_progress = round(
            max(0.0, min(100.0, modified.physical_progress + progress_change_pp)), 2
        )
    if schedule_change_days:
        months = round(schedule_change_days / 30.44)
        base_date = modified.revised_completion or modified.original_completion
        modified.revised_completion = _add_months(base_date, months)

    _recompute(modified)
    scenario = score_snapshot(modified, previous)

    def delta(a, b):
        if a is None or b is None:
            return None
        return round(b - a, 2)

    return {
        "label": SIMULATION_LABEL,
        "project_code": project.project_code,
        "project_name": project.name,
        "base_period": current.report_period,
        "inputs": {
            "cost_change_pct": cost_change_pct,
            "schedule_change_days": schedule_change_days,
            "progress_change_pp": progress_change_pp,
            "expenditure_change_pct": expenditure_change_pct,
        },
        "current": {
            "risk_score": baseline.score,
            "risk_level": baseline.level.value,
            "confidence": baseline.confidence,
            "revised_cost_cr": current.revised_cost,
            "expenditure_cr": current.expenditure,
            "physical_progress": current.physical_progress,
            "financial_progress": current.financial_progress,
            "cost_escalation_pct": current.cost_escalation_pct,
            "schedule_delay_months": current.schedule_delay_months,
        },
        "scenario": {
            "risk_score": scenario.score,
            "risk_level": scenario.level.value,
            "confidence": scenario.confidence,
            "revised_cost_cr": modified.revised_cost,
            "expenditure_cr": modified.expenditure,
            "physical_progress": modified.physical_progress,
            "financial_progress": modified.financial_progress,
            "cost_escalation_pct": modified.cost_escalation_pct,
            "schedule_delay_months": modified.schedule_delay_months,
        },
        "difference": {
            "risk_score": delta(baseline.score, scenario.score),
            "cost_escalation_pct": delta(current.cost_escalation_pct, modified.cost_escalation_pct),
            "progress_divergence": delta(
                current.progress_divergence, modified.progress_divergence
            ),
            "schedule_delay_months": delta(
                current.schedule_delay_months, modified.schedule_delay_months
            ),
        },
        "scenario_drivers": [d.as_dict() for d in scenario.drivers],
        "limitation": (
            "This is a deterministic recalculation of the published risk indicators under "
            "modified inputs. It shows how the current scoring model would read the project "
            "if these values changed. It is not a forecast of what will happen, and it does "
            "not account for factors absent from the Flash Report."
        ),
    }


# ---------------------------------------------------------------------------
# Pre-Approval Simulator — indicative risk for a hypothetical project
# ---------------------------------------------------------------------------
def cohort_benchmark(db: Session, sector: str | None, state: str | None) -> dict:
    """Real peer statistics from ingested snapshots. No synthetic benchmarks."""
    period = db.query(func.max(ProjectSnapshot.report_period)).scalar()
    q = (
        db.query(ProjectSnapshot)
        .join(Project, Project.id == ProjectSnapshot.project_id)
        .filter(
            ProjectSnapshot.report_period == period,
            ProjectSnapshot.supersedes_id.is_(None),
        )
    )
    scope = []
    if sector and sector != "ANY":
        q = q.filter(Project.sector == sector)
        scope.append(f"sector={sector}")
    if state and state != "ANY":
        q = q.filter(Project.state == state)
        scope.append(f"state={state}")

    snaps = q.all()
    return {
        "period": period,
        "scope": ", ".join(scope) or "all projects",
        "count": len(snaps),
        "sufficient": len(snaps) >= MIN_COHORT,
        "cost_escalation_pct": cohort_stats([s.cost_escalation_pct for s in snaps]),
        "schedule_delay_months": cohort_stats(
            [float(s.schedule_delay_months) if s.schedule_delay_months is not None else None
             for s in snaps]
        ),
        "risk_score": cohort_stats([s.risk_score for s in snaps]),
        "physical_progress": cohort_stats([s.physical_progress for s in snaps]),
        "_risk_values": [s.risk_score for s in snaps if s.risk_score is not None],
    }


def pre_approval_simulation(
    db: Session,
    *,
    estimated_cost_cr: float,
    duration_months: int,
    sector: str | None = None,
    state: str | None = None,
    expected_progress_year1: float | None = None,
) -> dict:
    """Indicative risk profile for a project that does not exist yet.

    We do NOT produce a "probability of failure". There is no outcome-labelled
    training data in the corpus, so such a number would be invented. Instead we
    place the proposal against the observed behaviour of comparable ongoing
    projects and state the sensitivities explicitly.
    """
    bench = cohort_benchmark(db, sector, state)
    risk_values = bench.pop("_risk_values")

    warnings: list[dict] = []
    notes: list[str] = []

    # --- Cost sensitivity, using the cohort's observed escalation -----------
    esc = bench["cost_escalation_pct"]
    projected_cost = None
    if bench["sufficient"] and esc["median"] is not None:
        projected_cost = round(estimated_cost_cr * (1 + esc["median"] / 100.0), 2)
        p90_cost = round(estimated_cost_cr * (1 + (esc["p90"] or esc["median"]) / 100.0), 2)
        notes.append(
            f"Comparable projects in this cohort show a median cost escalation of "
            f"{esc['median']:.2f}% (90th percentile {esc['p90']:.2f}%)."
        )
        if esc["median"] > 5:
            warnings.append(
                {
                    "code": "COHORT_COST_ESCALATION",
                    "severity": "MEDIUM",
                    "message": (
                        f"Peer projects escalate by a median {esc['median']:.1f}%. At that rate "
                        f"this proposal would reach ₹{projected_cost:,.0f} Cr, and "
                        f"₹{p90_cost:,.0f} Cr at the 90th percentile."
                    ),
                }
            )
    else:
        notes.append(
            f"Only {bench['count']} comparable project(s) found — too few to derive a "
            f"reliable cost benchmark. Cost sensitivity is not shown."
        )

    # --- Schedule sensitivity ---------------------------------------------
    delay = bench["schedule_delay_months"]
    projected_duration = None
    if bench["sufficient"] and delay["median"] is not None:
        projected_duration = round(duration_months + delay["median"])
        if delay["median"] >= 1:
            warnings.append(
                {
                    "code": "COHORT_SCHEDULE_SLIPPAGE",
                    "severity": "MEDIUM" if delay["median"] < 12 else "HIGH",
                    "message": (
                        f"Median schedule slippage in this cohort is {delay['median']:.1f} months. "
                        f"A {duration_months}-month plan would typically land at about "
                        f"{projected_duration} months."
                    ),
                }
            )

    # --- Structural checks on the proposal itself -------------------------
    if duration_months > 84:
        warnings.append(
            {
                "code": "LONG_DURATION",
                "severity": "MEDIUM",
                "message": (
                    f"A {duration_months}-month sanctioned duration exceeds seven years, over "
                    f"which cost revision and scope change are materially more likely."
                ),
            }
        )
    if estimated_cost_cr >= 1000 and bench["sufficient"]:
        rs = bench["risk_score"]
        if rs["median"] is not None:
            warnings.append(
                {
                    "code": "MEGA_PROJECT_EXPOSURE",
                    "severity": "LOW",
                    "message": (
                        f"At ₹{estimated_cost_cr:,.0f} Cr this is a mega project. Comparable "
                        f"ongoing projects currently carry a median composite risk of "
                        f"{rs['median']:.0f}."
                    ),
                }
            )
    if expected_progress_year1 is not None:
        implied = 100.0 * 12 / max(duration_months, 1)
        if expected_progress_year1 < implied * 0.7:
            warnings.append(
                {
                    "code": "FRONT_LOADED_SCHEDULE",
                    "severity": "MEDIUM",
                    "message": (
                        f"Expected first-year progress of {expected_progress_year1:.0f}% is well "
                        f"below the {implied:.0f}% implied by a straight-line "
                        f"{duration_months}-month schedule, which pushes delivery risk into "
                        f"later years."
                    ),
                }
            )

    # --- Indicative band ---------------------------------------------------
    indicative_score = None
    percentile = None
    if bench["sufficient"] and bench["risk_score"]["median"] is not None:
        indicative_score = bench["risk_score"]["median"]
        percentile = cohort_percentile(indicative_score, risk_values)
        band = (
            RiskLevel.SEVERE if indicative_score >= 70
            else RiskLevel.HIGH if indicative_score >= 50
            else RiskLevel.MODERATE if indicative_score >= 25
            else RiskLevel.LOW
        ).value
    else:
        band = "UNKNOWN"

    return {
        "label": SIMULATION_LABEL,
        "decision_support_only": True,
        "inputs": {
            "estimated_cost_cr": estimated_cost_cr,
            "duration_months": duration_months,
            "sector": sector or "ANY",
            "state": state or "ANY",
            "expected_progress_year1": expected_progress_year1,
        },
        "benchmark": bench,
        "indicative_risk_score": indicative_score,
        "indicative_risk_band": band,
        "cohort_percentile": percentile,
        "projected_cost_cr": projected_cost,
        "projected_duration_months": projected_duration,
        "warning_indicators": warnings,
        "notes": notes,
        "explanation": (
            "This output is a benchmark comparison, not a prediction. It places the proposal "
            "against the observed cost, schedule and risk behaviour of comparable ongoing "
            "projects in the ingested Flash Reports. PAIMANA AI does not publish a probability "
            "of failure, because the available corpus contains no completed-project outcomes "
            "against which such a model could be trained or validated."
        ),
    }
