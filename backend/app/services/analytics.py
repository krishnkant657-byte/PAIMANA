"""Aggregation and comparison queries.

Every figure returned here is computed from stored snapshots. Nothing is
seeded, smoothed, or padded. Where a cohort is too small to be meaningful the
service says so instead of returning a number.
"""
from __future__ import annotations

from sqlalchemy import and_, func
from sqlalchemy.orm import Session

from ..models import (
    Alert,
    AlertStatus,
    DataQualityIssue,
    Intervention,
    InterventionStatus,
    Project,
    ProjectSnapshot,
    RiskLevel,
)

MIN_COHORT = 8      # below this we report the count but suppress derived stats


def latest_period(db: Session) -> str | None:
    return db.query(func.max(ProjectSnapshot.report_period)).scalar()


def available_periods(db: Session) -> list[str]:
    rows = (
        db.query(ProjectSnapshot.report_period)
        .distinct()
        .order_by(ProjectSnapshot.report_period)
        .all()
    )
    return [r[0] for r in rows]


def latest_snapshots_query(db: Session, period: str | None = None):
    """Snapshots for the given (or newest) reporting period, excluding superseded."""
    period = period or latest_period(db)
    return (
        db.query(ProjectSnapshot)
        .filter(
            ProjectSnapshot.report_period == period,
            ProjectSnapshot.supersedes_id.is_(None),
        )
    )


def national_summary(db: Session, period: str | None = None) -> dict:
    period = period or latest_period(db)
    if period is None:
        return {"period": None, "message": "No data ingested yet."}

    q = latest_snapshots_query(db, period)
    snaps = q.all()
    total = len(snaps)

    def count_level(level: RiskLevel) -> int:
        return sum(1 for s in snaps if s.risk_level == level)

    costs = [(s.original_cost, s.revised_cost) for s in snaps]
    original_total = sum(c[0] for c in costs if c[0] is not None)
    revised_total = sum(
        (c[1] if c[1] is not None else c[0]) for c in costs if (c[1] or c[0]) is not None
    )
    expenditure_total = sum(s.expenditure for s in snaps if s.expenditure is not None)

    delayed = [s for s in snaps if (s.schedule_delay_months or 0) > 0]
    escalated = [s for s in snaps if (s.cost_escalation_pct or 0) > 0]

    progress_values = [s.physical_progress for s in snaps if s.physical_progress is not None]

    open_alerts = (
        db.query(func.count(Alert.id))
        .filter(Alert.status == AlertStatus.OPEN)
        .scalar()
    )
    open_interventions = (
        db.query(func.count(Intervention.id))
        .filter(Intervention.status.notin_([InterventionStatus.CLOSED]))
        .scalar()
    )

    return {
        "period": period,
        "total_projects": total,
        "risk_distribution": {
            "SEVERE": count_level(RiskLevel.SEVERE),
            "HIGH": count_level(RiskLevel.HIGH),
            "MODERATE": count_level(RiskLevel.MODERATE),
            "LOW": count_level(RiskLevel.LOW),
            "UNKNOWN": count_level(RiskLevel.UNKNOWN),
        },
        "at_risk": count_level(RiskLevel.SEVERE) + count_level(RiskLevel.HIGH),
        "delayed_projects": len(delayed),
        "delayed_pct": round(100.0 * len(delayed) / total, 1) if total else None,
        "cost_escalated_projects": len(escalated),
        "original_cost_cr": round(original_total, 2),
        "revised_cost_cr": round(revised_total, 2),
        "cost_exposure_cr": round(revised_total - original_total, 2),
        "expenditure_cr": round(expenditure_total, 2),
        "mean_physical_progress": (
            round(sum(progress_values) / len(progress_values), 2) if progress_values else None
        ),
        "open_alerts": open_alerts,
        "open_interventions": open_interventions,
        "coverage_note": (
            "All figures computed from ingested Flash Report snapshots for this period."
        ),
    }


def _group(db: Session, column, period: str | None = None, limit: int | None = None) -> list[dict]:
    period = period or latest_period(db)
    rows = (
        db.query(
            column.label("key"),
            func.count(ProjectSnapshot.id).label("projects"),
            func.sum(ProjectSnapshot.original_cost).label("original_cost"),
            func.sum(
                func.coalesce(ProjectSnapshot.revised_cost, ProjectSnapshot.original_cost)
            ).label("revised_cost"),
            func.sum(ProjectSnapshot.expenditure).label("expenditure"),
            func.avg(ProjectSnapshot.physical_progress).label("avg_progress"),
            func.avg(ProjectSnapshot.risk_score).label("avg_risk"),
            func.sum(
                func.coalesce(ProjectSnapshot.schedule_delay_months, 0)
            ).label("total_delay_months"),
        )
        .join(Project, Project.id == ProjectSnapshot.project_id)
        .filter(
            ProjectSnapshot.report_period == period,
            ProjectSnapshot.supersedes_id.is_(None),
        )
        .group_by(column)
        .order_by(func.count(ProjectSnapshot.id).desc())
    )
    if limit:
        rows = rows.limit(limit)

    out = []
    for r in rows.all():
        original = float(r.original_cost or 0)
        revised = float(r.revised_cost or 0)
        out.append(
            {
                "key": r.key,
                "projects": r.projects,
                "original_cost_cr": round(original, 2),
                "revised_cost_cr": round(revised, 2),
                "cost_exposure_cr": round(revised - original, 2),
                "expenditure_cr": round(float(r.expenditure or 0), 2),
                "avg_physical_progress": (
                    round(float(r.avg_progress), 2) if r.avg_progress is not None else None
                ),
                "avg_risk_score": (
                    round(float(r.avg_risk), 1) if r.avg_risk is not None else None
                ),
                "avg_delay_months": (
                    round(float(r.total_delay_months or 0) / r.projects, 1) if r.projects else None
                ),
                "reliable": r.projects >= MIN_COHORT,
            }
        )
    return out


def by_state(db: Session, period: str | None = None) -> list[dict]:
    return _group(db, Project.state, period)


def by_sector(db: Session, period: str | None = None) -> list[dict]:
    return _group(db, Project.sector, period)


def by_ministry(db: Session, period: str | None = None) -> list[dict]:
    return _group(db, Project.ministry, period)


def national_trend(db: Session) -> list[dict]:
    """Period-by-period national position. The backbone of the trend charts."""
    rows = (
        db.query(
            ProjectSnapshot.report_period.label("period"),
            func.count(ProjectSnapshot.id).label("projects"),
            func.avg(ProjectSnapshot.risk_score).label("avg_risk"),
            func.avg(ProjectSnapshot.physical_progress).label("avg_progress"),
            func.sum(ProjectSnapshot.original_cost).label("original_cost"),
            func.sum(
                func.coalesce(ProjectSnapshot.revised_cost, ProjectSnapshot.original_cost)
            ).label("revised_cost"),
            func.sum(ProjectSnapshot.expenditure).label("expenditure"),
        )
        .filter(ProjectSnapshot.supersedes_id.is_(None))
        .group_by(ProjectSnapshot.report_period)
        .order_by(ProjectSnapshot.report_period)
        .all()
    )
    out = []
    for r in rows:
        original = float(r.original_cost or 0)
        revised = float(r.revised_cost or 0)
        severe_high = (
            db.query(func.count(ProjectSnapshot.id))
            .filter(
                ProjectSnapshot.report_period == r.period,
                ProjectSnapshot.supersedes_id.is_(None),
                ProjectSnapshot.risk_level.in_([RiskLevel.SEVERE, RiskLevel.HIGH]),
            )
            .scalar()
        )
        out.append(
            {
                "period": r.period,
                "projects": r.projects,
                "avg_risk_score": round(float(r.avg_risk), 1) if r.avg_risk is not None else None,
                "avg_physical_progress": (
                    round(float(r.avg_progress), 2) if r.avg_progress is not None else None
                ),
                "at_risk": severe_high,
                "cost_exposure_cr": round(revised - original, 2),
                "expenditure_cr": round(float(r.expenditure or 0), 2),
            }
        )
    return out


def compare(db: Session, dimension: str, values: list[str], period: str | None = None) -> dict:
    """Side-by-side comparison, e.g. Maharashtra vs Gujarat, or Roads vs Railways."""
    column = {
        "state": Project.state,
        "sector": Project.sector,
        "ministry": Project.ministry,
    }.get(dimension)
    if column is None:
        raise ValueError("dimension must be one of: state, sector, ministry")

    period = period or latest_period(db)
    results = []
    for value in values:
        snaps = (
            db.query(ProjectSnapshot)
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(
                and_(
                    column == value,
                    ProjectSnapshot.report_period == period,
                    ProjectSnapshot.supersedes_id.is_(None),
                )
            )
            .all()
        )
        risks = [s.risk_score for s in snaps if s.risk_score is not None]
        delays = [s.schedule_delay_months for s in snaps if s.schedule_delay_months is not None]
        esc = [s.cost_escalation_pct for s in snaps if s.cost_escalation_pct is not None]
        prog = [s.physical_progress for s in snaps if s.physical_progress is not None]
        results.append(
            {
                "value": value,
                "projects": len(snaps),
                "reliable": len(snaps) >= MIN_COHORT,
                "avg_risk_score": round(sum(risks) / len(risks), 1) if risks else None,
                "at_risk": sum(
                    1 for s in snaps if s.risk_level in (RiskLevel.SEVERE, RiskLevel.HIGH)
                ),
                "avg_delay_months": round(sum(delays) / len(delays), 1) if delays else None,
                "avg_cost_escalation_pct": round(sum(esc) / len(esc), 2) if esc else None,
                "avg_physical_progress": round(sum(prog) / len(prog), 2) if prog else None,
                "cost_exposure_cr": round(
                    sum(
                        ((s.revised_cost if s.revised_cost is not None else s.original_cost) or 0)
                        - (s.original_cost or 0)
                        for s in snaps
                        if s.original_cost is not None
                    ),
                    2,
                ),
            }
        )
    return {
        "dimension": dimension,
        "period": period,
        "results": results,
        "note": (
            f"Cohorts smaller than {MIN_COHORT} projects are marked unreliable; "
            f"averages over such groups are shown but should not be generalised."
        ),
    }


def data_quality_summary(db: Session) -> dict:
    total_projects = db.query(func.count(Project.id)).scalar()
    total_snapshots = (
        db.query(func.count(ProjectSnapshot.id))
        .filter(ProjectSnapshot.supersedes_id.is_(None))
        .scalar()
    )
    avg_completeness = (
        db.query(func.avg(ProjectSnapshot.completeness))
        .filter(ProjectSnapshot.supersedes_id.is_(None))
        .scalar()
    )
    open_issues = (
        db.query(DataQualityIssue.issue_type, func.count(DataQualityIssue.id))
        .filter(DataQualityIssue.status == "OPEN")
        .group_by(DataQualityIssue.issue_type)
        .all()
    )
    review_count = (
        db.query(func.count(ProjectSnapshot.id))
        .filter(
            ProjectSnapshot.data_quality_status == "REVIEW",
            ProjectSnapshot.supersedes_id.is_(None),
        )
        .scalar()
    )
    unknown_sector = (
        db.query(func.count(Project.id)).filter(Project.sector == "UNKNOWN").scalar()
    )
    unknown_state = db.query(func.count(Project.id)).filter(Project.state == "UNKNOWN").scalar()

    completeness_pct = round(float(avg_completeness or 0), 2)
    issue_total = sum(c for _, c in open_issues)
    penalty = min(30.0, 30.0 * issue_total / max(total_snapshots, 1))
    score = round(max(0.0, completeness_pct - penalty), 1)

    return {
        "data_quality_score": score,
        "avg_completeness_pct": completeness_pct,
        "total_projects": total_projects,
        "total_snapshots": total_snapshots,
        "snapshots_awaiting_review": review_count,
        "projects_unknown_sector": unknown_sector,
        "projects_unknown_state": unknown_state,
        "open_issues_total": issue_total,
        "open_issues_by_type": {t: c for t, c in open_issues},
    }
