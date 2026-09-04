"""Early Warning Engine.

Every warning is a deterministic rule over two consecutive snapshots or one
snapshot's indicators. Each carries severity, trigger text, and the evidence
values that fired it, so an official can check the finding against the source.
"""
from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from ..models import Alert, AlertStatus, Project, ProjectSnapshot, Severity

# Thresholds are stated here rather than buried in code so they can be reviewed
# and tuned by domain experts.
THRESHOLDS = {
    "PROGRESS_DIVERGENCE": 15.0,        # percentage points, financial ahead of physical
    "COST_ACCELERATION": 5.0,           # % increase in revised cost vs previous snapshot
    "SCHEDULE_DETERIORATION": 6,        # months added to forecast completion
    "RISK_DETERIORATION": 2,            # consecutive periods of rising risk
    "PROGRESS_STALL_PERIODS": 2,        # consecutive periods with no movement
    "SEVERE_RISK": 70.0,
}


def _sev(value: float, medium: float, high: float, critical: float) -> Severity:
    if value >= critical:
        return Severity.CRITICAL
    if value >= high:
        return Severity.HIGH
    if value >= medium:
        return Severity.MEDIUM
    return Severity.LOW


def evaluate_project(db: Session, project: Project) -> list[Alert]:
    snaps = (
        db.query(ProjectSnapshot)
        .filter_by(project_id=project.id, supersedes_id=None)
        .order_by(ProjectSnapshot.report_period)
        .all()
    )
    if not snaps:
        return []

    latest = snaps[-1]
    prev = snaps[-2] if len(snaps) > 1 else None
    created: list[Alert] = []

    def emit(code, title, description, severity, trigger, evidence):
        exists = (
            db.query(Alert)
            .filter_by(project_id=project.id, code=code, report_period=latest.report_period)
            .first()
        )
        if exists:
            return
        alert = Alert(
            project_id=project.id,
            code=code,
            title=title,
            description=description,
            severity=severity,
            status=AlertStatus.OPEN,
            trigger=trigger,
            evidence=evidence,
            report_period=latest.report_period,
            snapshot_id=latest.id,
        )
        db.add(alert)
        created.append(alert)

    # --- PROGRESS DIVERGENCE ---------------------------------------------
    if (latest.progress_divergence or 0) >= THRESHOLDS["PROGRESS_DIVERGENCE"]:
        emit(
            "PROGRESS_DIVERGENCE",
            "Financial progress significantly ahead of physical progress",
            (
                f"{latest.financial_progress:.1f}% of the revised cost has been drawn against "
                f"{latest.physical_progress:.1f}% physical completion."
            ),
            _sev(latest.progress_divergence, 15, 30, 50),
            f"progress_divergence >= {THRESHOLDS['PROGRESS_DIVERGENCE']} pp",
            {
                "financial_progress": latest.financial_progress,
                "physical_progress": latest.physical_progress,
                "divergence_pp": latest.progress_divergence,
                "expenditure_cr": latest.expenditure,
                "report_period": latest.report_period,
            },
        )

    # --- COST ACCELERATION ------------------------------------------------
    if prev and None not in (latest.revised_cost, prev.revised_cost) and prev.revised_cost:
        change = 100.0 * (latest.revised_cost - prev.revised_cost) / prev.revised_cost
        if change >= THRESHOLDS["COST_ACCELERATION"]:
            emit(
                "COST_ACCELERATION",
                "Project cost increased against the previous reporting period",
                (
                    f"Revised cost rose {change:.1f}% from ₹{prev.revised_cost:,.2f} Cr to "
                    f"₹{latest.revised_cost:,.2f} Cr in a single reporting cycle."
                ),
                _sev(change, 5, 15, 30),
                f"revised_cost increase >= {THRESHOLDS['COST_ACCELERATION']}%",
                {
                    "previous_cost_cr": prev.revised_cost,
                    "current_cost_cr": latest.revised_cost,
                    "increase_pct": round(change, 2),
                    "from_period": prev.report_period,
                    "to_period": latest.report_period,
                },
            )

    # --- SCHEDULE DETERIORATION -------------------------------------------
    if prev and None not in (latest.schedule_delay_months, prev.schedule_delay_months):
        added = latest.schedule_delay_months - prev.schedule_delay_months
        if added >= THRESHOLDS["SCHEDULE_DETERIORATION"]:
            emit(
                "SCHEDULE_DETERIORATION",
                "Forecast completion moved substantially beyond baseline",
                (
                    f"Forecast completion slipped a further {added} month(s) this cycle; "
                    f"cumulative slippage is now {latest.schedule_delay_months} months."
                ),
                _sev(float(added), 6, 12, 24),
                f"schedule slip >= {THRESHOLDS['SCHEDULE_DETERIORATION']} months in one cycle",
                {
                    "previous_delay_months": prev.schedule_delay_months,
                    "current_delay_months": latest.schedule_delay_months,
                    "added_months": added,
                    "original_completion": (
                        latest.original_completion.isoformat()
                        if latest.original_completion else None
                    ),
                    "revised_completion": (
                        latest.revised_completion.isoformat()
                        if latest.revised_completion else None
                    ),
                },
            )

    # --- RISK DETERIORATION ------------------------------------------------
    scores = [s.risk_score for s in snaps if s.risk_score is not None]
    if len(scores) >= THRESHOLDS["RISK_DETERIORATION"] + 1:
        streak = 0
        for i in range(len(scores) - 1, 0, -1):
            if scores[i] > scores[i - 1] + 0.5:
                streak += 1
            else:
                break
        if streak >= THRESHOLDS["RISK_DETERIORATION"]:
            rise = scores[-1] - scores[-1 - streak]
            emit(
                "RISK_DETERIORATION",
                "Risk score rising across consecutive reporting periods",
                (
                    f"Composite risk has increased in {streak} consecutive cycles, from "
                    f"{scores[-1 - streak]:.0f} to {scores[-1]:.0f}."
                ),
                _sev(rise, 5, 15, 25),
                f"risk increased for >= {THRESHOLDS['RISK_DETERIORATION']} consecutive periods",
                {
                    "series": [
                        {"period": s.report_period, "score": s.risk_score}
                        for s in snaps if s.risk_score is not None
                    ],
                    "streak": streak,
                    "increase": round(rise, 2),
                },
            )

    # --- PROGRESS STALL ----------------------------------------------------
    if len(snaps) >= THRESHOLDS["PROGRESS_STALL_PERIODS"] + 1:
        window = snaps[-(THRESHOLDS["PROGRESS_STALL_PERIODS"] + 1):]
        vals = [s.physical_progress for s in window]
        if all(v is not None for v in vals) and (max(vals) - min(vals)) <= 0.05:
            emit(
                "PROGRESS_STALL",
                "Physical progress unchanged across multiple reporting periods",
                (
                    f"Physical progress has remained at {vals[-1]:.2f}% across "
                    f"{len(window)} consecutive reports."
                ),
                Severity.HIGH if (vals[-1] or 0) < 50 else Severity.MEDIUM,
                f"no movement over {THRESHOLDS['PROGRESS_STALL_PERIODS']} periods",
                {
                    "series": [
                        {"period": s.report_period, "physical_progress": s.physical_progress}
                        for s in window
                    ]
                },
            )

    # --- SEVERE RISK EXPOSURE ---------------------------------------------
    if (latest.risk_score or 0) >= THRESHOLDS["SEVERE_RISK"]:
        emit(
            "SEVERE_RISK_EXPOSURE",
            "Project in severe composite risk band",
            (
                f"Composite risk score is {latest.risk_score:.0f} "
                f"({latest.risk_level.value}) as at {latest.report_period}."
            ),
            Severity.CRITICAL if (latest.risk_score or 0) >= 85 else Severity.HIGH,
            f"risk_score >= {THRESHOLDS['SEVERE_RISK']}",
            {
                "risk_score": latest.risk_score,
                "risk_level": latest.risk_level.value,
                "confidence": latest.risk_confidence,
                "drivers": [d["code"] for d in (latest.risk_drivers or [])],
            },
        )

    return created


def generate_all_alerts(db: Session, batch: int = 500) -> dict:
    projects = db.query(Project).all()
    total = 0
    for i, project in enumerate(projects, start=1):
        total += len(evaluate_project(db, project))
        if i % batch == 0:
            db.commit()
    db.commit()
    return {"projects_evaluated": len(projects), "alerts_created": total}
