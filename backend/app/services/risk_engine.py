"""PAIMANA composite risk engine.

Layer 1  Deterministic indicators   (cost, schedule, divergence, stall)
Layer 2  Trend analytics            (month-over-month direction)
Layer 3  Cohort benchmarking        (percentile within sector/state peers)
Layer 4  Machine learning           -- DELIBERATELY NOT IMPLEMENTED, see below
Layer 5  Composite score            (weighted, with confidence + drivers)
Layer 6  LLM                        (explains Layer 5 output; never computes it)

WHY THERE IS NO TRAINED MODEL HERE
----------------------------------
The corpus is four consecutive monthly reports. That gives three month-to-month
transitions and no observed completion outcomes, so there is no legitimate
target label for a supervised risk model. The previous implementation
manufactured a label with a hand-written rule and then trained a Random Forest
on the output of that rule; its strongest feature was a randomly generated
column. That is not prediction, it is circular reasoning with a confidence
interval attached.

This engine is fully deterministic and every point of the score is attributable
to a named driver backed by a source value. When enough reporting periods
accumulate that projects reach completion, `docs/ARCHITECTURE.md` sets out the
conditions under which Layer 4 becomes defensible.
"""
from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import asdict, dataclass, field

from sqlalchemy.orm import Session

from ..models import (
    Project,
    ProjectSnapshot,
    RiskEvent,
    RiskLevel,
    TrendDirection,
)

ENGINE_VERSION = "2.0-deterministic"

# Weights sum to 100. Each is a maximum contribution to the composite score.
WEIGHTS = {
    "cost_escalation": 25.0,
    "schedule_delay": 25.0,
    "progress_divergence": 20.0,
    "progress_stall": 15.0,
    "time_progress_gap": 15.0,
}


@dataclass
class RiskDriver:
    code: str
    label: str
    detail: str
    contribution: float          # points added to the composite score
    weight: float                # maximum possible for this driver
    evidence: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        d["contribution"] = round(d["contribution"], 2)
        return d


@dataclass
class RiskResult:
    score: float | None
    level: RiskLevel
    confidence: float
    drivers: list
    engine_version: str = ENGINE_VERSION

    def as_dict(self) -> dict:
        return {
            "score": None if self.score is None else round(self.score, 1),
            "level": self.level.value,
            "confidence": round(self.confidence, 3),
            "drivers": [d.as_dict() for d in self.drivers],
            "engine_version": self.engine_version,
        }


def _band(score: float) -> RiskLevel:
    if score >= 70:
        return RiskLevel.SEVERE
    if score >= 50:
        return RiskLevel.HIGH
    if score >= 25:
        return RiskLevel.MODERATE
    return RiskLevel.LOW


def _scale(value: float, low: float, high: float, weight: float) -> float:
    """Linear ramp from 0 points at `low` to full `weight` at `high`."""
    if value <= low:
        return 0.0
    if value >= high:
        return weight
    return weight * (value - low) / (high - low)


# ---------------------------------------------------------------------------
# Layer 1 + 5 : indicators -> composite
# ---------------------------------------------------------------------------
def score_snapshot(snap: ProjectSnapshot, previous: ProjectSnapshot | None = None) -> RiskResult:
    drivers: list[RiskDriver] = []
    available, total = 0, 5

    # --- cost escalation --------------------------------------------------
    if snap.cost_escalation_pct is not None:
        available += 1
        pts = _scale(snap.cost_escalation_pct, 0.0, 50.0, WEIGHTS["cost_escalation"])
        if pts > 0:
            drivers.append(
                RiskDriver(
                    code="COST_ESCALATION",
                    label="Cost escalation against approved cost",
                    detail=(
                        f"Revised cost is {snap.cost_escalation_pct:.1f}% above the originally "
                        f"approved cost."
                    ),
                    contribution=pts,
                    weight=WEIGHTS["cost_escalation"],
                    evidence={
                        "original_cost": snap.original_cost,
                        "revised_cost": snap.revised_cost,
                        "escalation_pct": snap.cost_escalation_pct,
                        "report_period": snap.report_period,
                    },
                )
            )

    # --- schedule delay ---------------------------------------------------
    if snap.schedule_delay_months is not None:
        available += 1
        pts = _scale(float(snap.schedule_delay_months), 0.0, 36.0, WEIGHTS["schedule_delay"])
        if pts > 0:
            drivers.append(
                RiskDriver(
                    code="SCHEDULE_DELAY",
                    label="Completion date moved beyond baseline",
                    detail=(
                        f"Revised completion is {snap.schedule_delay_months} month(s) later than "
                        f"the original target date."
                    ),
                    contribution=pts,
                    weight=WEIGHTS["schedule_delay"],
                    evidence={
                        "original_completion": (
                            snap.original_completion.isoformat()
                            if snap.original_completion else None
                        ),
                        "revised_completion": (
                            snap.revised_completion.isoformat()
                            if snap.revised_completion else None
                        ),
                        "delay_months": snap.schedule_delay_months,
                    },
                )
            )

    # --- progress divergence (money spent ahead of work done) -------------
    if snap.progress_divergence is not None:
        available += 1
        pts = _scale(snap.progress_divergence, 5.0, 40.0, WEIGHTS["progress_divergence"])
        if pts > 0:
            drivers.append(
                RiskDriver(
                    code="PROGRESS_DIVERGENCE",
                    label="Financial progress ahead of physical progress",
                    detail=(
                        f"{snap.financial_progress:.1f}% of cost has been spent against "
                        f"{snap.physical_progress:.1f}% physical completion — a gap of "
                        f"{snap.progress_divergence:.1f} percentage points."
                    ),
                    contribution=pts,
                    weight=WEIGHTS["progress_divergence"],
                    evidence={
                        "financial_progress": snap.financial_progress,
                        "physical_progress": snap.physical_progress,
                        "divergence": snap.progress_divergence,
                        "expenditure": snap.expenditure,
                    },
                )
            )

    # --- progress stall (needs history) -----------------------------------
    if previous is not None and None not in (
        snap.physical_progress, previous.physical_progress
    ):
        available += 1
        movement = snap.physical_progress - previous.physical_progress
        if movement <= 0.05:
            pts = WEIGHTS["progress_stall"] * (1.0 if movement < 0 else 0.7)
            drivers.append(
                RiskDriver(
                    code="PROGRESS_STALL",
                    label="Physical progress stalled or regressed",
                    detail=(
                        f"Physical progress moved {movement:+.2f} percentage points between "
                        f"{previous.report_period} and {snap.report_period}."
                    ),
                    contribution=pts,
                    weight=WEIGHTS["progress_stall"],
                    evidence={
                        "previous_progress": previous.physical_progress,
                        "current_progress": snap.physical_progress,
                        "movement": round(movement, 2),
                        "from_period": previous.report_period,
                        "to_period": snap.report_period,
                    },
                )
            )

    # --- elapsed time vs physical progress --------------------------------
    if snap.elapsed_time_pct is not None and snap.physical_progress is not None:
        available += 1
        gap = snap.elapsed_time_pct - snap.physical_progress
        pts = _scale(gap, 10.0, 60.0, WEIGHTS["time_progress_gap"])
        if pts > 0:
            drivers.append(
                RiskDriver(
                    code="TIME_PROGRESS_GAP",
                    label="Elapsed time outpacing physical delivery",
                    detail=(
                        f"{snap.elapsed_time_pct:.0f}% of the sanctioned timeline has elapsed "
                        f"against {snap.physical_progress:.1f}% physical progress."
                    ),
                    contribution=pts,
                    weight=WEIGHTS["time_progress_gap"],
                    evidence={
                        "elapsed_time_pct": snap.elapsed_time_pct,
                        "physical_progress": snap.physical_progress,
                        "gap": round(gap, 2),
                    },
                )
            )

    if available == 0:
        return RiskResult(None, RiskLevel.UNKNOWN, 0.0, [])

    # Score is renormalised over the indicators we could actually compute, so a
    # project with missing fields is not artificially rewarded for the gap.
    achievable = 0.0
    if snap.cost_escalation_pct is not None:
        achievable += WEIGHTS["cost_escalation"]
    if snap.schedule_delay_months is not None:
        achievable += WEIGHTS["schedule_delay"]
    if snap.progress_divergence is not None:
        achievable += WEIGHTS["progress_divergence"]
    if previous is not None and None not in (snap.physical_progress, previous.physical_progress):
        achievable += WEIGHTS["progress_stall"]
    if snap.elapsed_time_pct is not None and snap.physical_progress is not None:
        achievable += WEIGHTS["time_progress_gap"]

    raw = sum(d.contribution for d in drivers)
    score = 100.0 * raw / achievable if achievable else 0.0
    score = max(0.0, min(100.0, score))

    # Confidence reflects how much of the model we could evaluate and how
    # complete the underlying record was — not how sure we are of an outcome.
    coverage = available / total
    completeness = (snap.completeness or 0.0) / 100.0
    confidence = round(0.65 * coverage + 0.35 * completeness, 3)

    drivers.sort(key=lambda d: d.contribution, reverse=True)
    return RiskResult(score, _band(score), confidence, drivers)


# ---------------------------------------------------------------------------
# Layer 2 : trend
# ---------------------------------------------------------------------------
def classify_trend(scores: list[float | None], tolerance: float = 3.0) -> TrendDirection:
    series = [s for s in scores if s is not None]
    if len(series) < 2:
        return TrendDirection.INSUFFICIENT_HISTORY
    delta = series[-1] - series[0]
    if delta > tolerance:
        return TrendDirection.DETERIORATING
    if delta < -tolerance:
        return TrendDirection.IMPROVING
    return TrendDirection.STABLE


def deterioration_streak(scores: list[float | None]) -> int:
    """Number of consecutive period-on-period increases at the end of the series."""
    series = [s for s in scores if s is not None]
    streak = 0
    for i in range(len(series) - 1, 0, -1):
        if series[i] > series[i - 1]:
            streak += 1
        else:
            break
    return streak


# ---------------------------------------------------------------------------
# Layer 3 : cohort benchmarking
# ---------------------------------------------------------------------------
def cohort_percentile(value: float, cohort: list[float]) -> float | None:
    clean = [c for c in cohort if c is not None]
    if len(clean) < 8:
        return None                     # too small a peer group to be meaningful
    below = sum(1 for c in clean if c < value)
    return round(100.0 * below / len(clean), 1)


def cohort_stats(values: list[float | None]) -> dict:
    clean = [v for v in values if v is not None]
    if not clean:
        return {"count": 0, "median": None, "mean": None, "p90": None}
    clean.sort()
    idx = max(0, min(len(clean) - 1, int(0.9 * len(clean)) - 1))
    return {
        "count": len(clean),
        "median": round(statistics.median(clean), 2),
        "mean": round(statistics.fmean(clean), 2),
        "p90": round(clean[idx], 2),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def recalculate_project(db: Session, project: Project) -> int:
    """Score every snapshot of one project in chronological order."""
    snaps = (
        db.query(ProjectSnapshot)
        .filter_by(project_id=project.id, supersedes_id=None)
        .order_by(ProjectSnapshot.report_period)
        .all()
    )
    previous = None
    for snap in snaps:
        result = score_snapshot(snap, previous)
        snap.risk_score = result.score
        snap.risk_level = result.level
        snap.risk_confidence = result.confidence
        snap.risk_drivers = [d.as_dict() for d in result.drivers]
        snap.risk_engine_version = ENGINE_VERSION

        if previous is not None and None not in (snap.risk_score, previous.risk_score):
            delta = snap.risk_score - previous.risk_score
            if abs(delta) >= 3.0:
                db.add(
                    RiskEvent(
                        project_id=project.id,
                        from_period=previous.report_period,
                        to_period=snap.report_period,
                        from_score=previous.risk_score,
                        to_score=snap.risk_score,
                        delta=round(delta, 2),
                        direction=(
                            TrendDirection.DETERIORATING if delta > 0
                            else TrendDirection.IMPROVING
                        ),
                        explanation=explain_change(previous, snap),
                    )
                )
        previous = snap
    return len(snaps)


def explain_change(before: ProjectSnapshot, after: ProjectSnapshot) -> dict:
    """Why did risk move between two periods? Every line traced to two values."""
    changes = []

    def add(label, prev, curr, unit="", fmt="{:.2f}"):
        if prev is None or curr is None or abs(curr - prev) < 1e-9:
            return
        changes.append(
            {
                "field": label,
                "from": round(prev, 2),
                "to": round(curr, 2),
                "delta": round(curr - prev, 2),
                "unit": unit,
                "text": (
                    f"{label} moved from {fmt.format(prev)}{unit} to "
                    f"{fmt.format(curr)}{unit} ({curr - prev:+.2f}{unit})"
                ),
            }
        )

    add("Physical progress", before.physical_progress, after.physical_progress, "%")
    add("Financial progress", before.financial_progress, after.financial_progress, "%")
    add("Expenditure", before.expenditure, after.expenditure, " Cr")
    add("Revised cost", before.revised_cost, after.revised_cost, " Cr")
    add("Cost escalation", before.cost_escalation_pct, after.cost_escalation_pct, "%")
    add("Progress divergence", before.progress_divergence, after.progress_divergence, " pp")

    if (
        before.schedule_delay_months is not None
        and after.schedule_delay_months is not None
        and before.schedule_delay_months != after.schedule_delay_months
    ):
        changes.append(
            {
                "field": "Schedule delay",
                "from": before.schedule_delay_months,
                "to": after.schedule_delay_months,
                "delta": after.schedule_delay_months - before.schedule_delay_months,
                "unit": " months",
                "text": (
                    f"Schedule delay moved from {before.schedule_delay_months} to "
                    f"{after.schedule_delay_months} months "
                    f"({after.schedule_delay_months - before.schedule_delay_months:+d})"
                ),
            }
        )

    return {
        "from_period": before.report_period,
        "to_period": after.report_period,
        "from_score": before.risk_score,
        "to_score": after.risk_score,
        "changes": changes,
        "new_drivers": [
            d["code"]
            for d in (after.risk_drivers or [])
            if d["code"] not in {x["code"] for x in (before.risk_drivers or [])}
        ],
    }


def recalculate_all(db: Session, batch: int = 500) -> dict:
    projects = db.query(Project).all()
    total_snaps = 0
    for i, project in enumerate(projects, start=1):
        total_snaps += recalculate_project(db, project)
        if i % batch == 0:
            db.commit()
    db.commit()
    return {"projects": len(projects), "snapshots_scored": total_snaps,
            "engine_version": ENGINE_VERSION}
