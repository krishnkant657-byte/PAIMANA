"""Project Explorer, Digital Twin, Escalation Replay, and Provenance."""
from __future__ import annotations

import math

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import (
    Alert,
    AlertStatus,
    DataQualityIssue,
    Intervention,
    Project,
    ProjectSnapshot,
    RiskEvent,
    RiskLevel,
)
from ..schemas import PaginatedProjects, ProjectSummary
from ..services import analytics
from ..services.risk_engine import classify_trend, deterioration_streak, explain_change


RECOMMENDATIONS = {
    "SCHEDULE_DELAY": {
        "priority": "HIGH",
        "action": "Review delayed milestones, contractor mobilisation and the revised completion plan.",
        "reason": "The reported completion date is beyond the original baseline.",
    },
    "COST_ESCALATION": {
        "priority": "HIGH",
        "action": "Trigger a cost-variance review and request justification for the revised estimate.",
        "reason": "Revised cost has moved above the originally approved cost.",
    },
    "PROGRESS_DIVERGENCE": {
        "priority": "HIGH",
        "action": "Review expenditure against physical delivery and verify milestone-level progress.",
        "reason": "Financial progress is materially ahead of physical progress.",
    },
    "PROGRESS_STALL": {
        "priority": "CRITICAL",
        "action": "Escalate the stalled workstream and request a recovery plan with dated milestones.",
        "reason": "Physical progress has stalled or regressed versus the previous report.",
    },
    "TIME_PROGRESS_GAP": {
        "priority": "MEDIUM",
        "action": "Review the remaining schedule against actual physical progress and identify the critical path.",
        "reason": "A large share of the sanctioned timeline has elapsed relative to work completed.",
    },
}

router = APIRouter(prefix="/api/projects", tags=["projects"])


def _latest_snapshot(db: Session, project_id: int) -> ProjectSnapshot | None:
    return (
        db.query(ProjectSnapshot)
        .filter_by(project_id=project_id, supersedes_id=None)
        .order_by(ProjectSnapshot.report_period.desc())
        .first()
    )


def _snapshot_dict(s: ProjectSnapshot) -> dict:
    return {
        "id": s.id,
        "period": s.report_period,
        "original_cost_cr": s.original_cost,
        "revised_cost_cr": s.revised_cost,
        "expenditure_cr": s.expenditure,
        "physical_progress": s.physical_progress,
        "financial_progress": s.financial_progress,
        "cost_escalation_pct": s.cost_escalation_pct,
        "progress_divergence": s.progress_divergence,
        "schedule_delay_months": s.schedule_delay_months,
        "elapsed_time_pct": s.elapsed_time_pct,
        "approval_date": s.approval_date.isoformat() if s.approval_date else None,
        "start_date": s.start_date.isoformat() if s.start_date else None,
        "original_completion": (
            s.original_completion.isoformat() if s.original_completion else None
        ),
        "revised_completion": (
            s.revised_completion.isoformat() if s.revised_completion else None
        ),
        "risk_score": s.risk_score,
        "risk_level": s.risk_level.value if s.risk_level else "UNKNOWN",
        "risk_confidence": s.risk_confidence,
        "risk_drivers": s.risk_drivers or [],
        "data_quality_status": s.data_quality_status,
        "completeness": s.completeness,
        "is_demo": s.is_demo,
        "source": {
            "document": s.source.filename if s.source else None,
            "label": s.source.report_label if s.source else None,
            "page": s.record.page_number if s.record else None,
        },
    }


@router.get("", response_model=PaginatedProjects)
def list_projects(
    db: Session = Depends(get_db),
    q: str | None = Query(None, max_length=200, description="search name, code or agency"),
    state: str | None = None,
    sector: str | None = None,
    ministry: str | None = None,
    risk_level: str | None = None,
    period: str | None = None,
    min_risk: float | None = Query(None, ge=0, le=100),
    sort: str = Query("risk_desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=1, le=200),
):
    """Search, filter, sort and paginate the project register."""
    period = period or analytics.latest_period(db)
    if period is None:
        return PaginatedProjects(
            total=0, page=page, page_size=page_size, pages=0, period=None, items=[]
        )

    base = (
        db.query(ProjectSnapshot, Project)
        .join(Project, Project.id == ProjectSnapshot.project_id)
        .filter(
            ProjectSnapshot.report_period == period,
            ProjectSnapshot.supersedes_id.is_(None),
        )
    )

    if q:
        like = f"%{q.strip()}%"
        base = base.filter(
            or_(
                Project.name.ilike(like),
                Project.project_code.ilike(like),
                Project.agency.ilike(like),
            )
        )
    if state:
        base = base.filter(Project.state == state)
    if sector:
        base = base.filter(Project.sector == sector)
    if ministry:
        base = base.filter(Project.ministry == ministry)
    if risk_level:
        try:
            base = base.filter(ProjectSnapshot.risk_level == RiskLevel(risk_level.upper()))
        except ValueError:
            raise HTTPException(400, f"Unknown risk level: {risk_level}")
    if min_risk is not None:
        base = base.filter(ProjectSnapshot.risk_score >= min_risk)

    total = base.count()

    order = {
        "risk_desc": ProjectSnapshot.risk_score.desc(),
        "risk_asc": ProjectSnapshot.risk_score.asc(),
        "cost_desc": ProjectSnapshot.revised_cost.desc(),
        "progress_asc": ProjectSnapshot.physical_progress.asc(),
        "progress_desc": ProjectSnapshot.physical_progress.desc(),
        "delay_desc": ProjectSnapshot.schedule_delay_months.desc(),
        "name": Project.name.asc(),
    }.get(sort, ProjectSnapshot.risk_score.desc())

    rows = base.order_by(order).offset((page - 1) * page_size).limit(page_size).all()

    alert_counts = dict(
        db.query(Alert.project_id, func.count(Alert.id))
        .filter(Alert.status == AlertStatus.OPEN)
        .group_by(Alert.project_id)
        .all()
    )

    items = []
    for snap, proj in rows:
        history = [
            s.risk_score
            for s in db.query(ProjectSnapshot)
            .filter_by(project_id=proj.id, supersedes_id=None)
            .order_by(ProjectSnapshot.report_period)
            .all()
        ]
        items.append(
            ProjectSummary(
                id=proj.id,
                project_code=proj.project_code,
                name=proj.name,
                agency=proj.agency,
                ministry=proj.ministry,
                sector=proj.sector,
                state=proj.state,
                is_demo=proj.is_demo,
                risk_score=snap.risk_score,
                risk_level=snap.risk_level.value if snap.risk_level else "UNKNOWN",
                risk_confidence=snap.risk_confidence,
                trend=classify_trend(history).value,
                physical_progress=snap.physical_progress,
                financial_progress=snap.financial_progress,
                original_cost_cr=snap.original_cost,
                revised_cost_cr=snap.revised_cost,
                expenditure_cr=snap.expenditure,
                cost_escalation_pct=snap.cost_escalation_pct,
                schedule_delay_months=snap.schedule_delay_months,
                last_period=snap.report_period,
                open_alerts=alert_counts.get(proj.id, 0),
            )
        )

    return PaginatedProjects(
        total=total,
        page=page,
        page_size=page_size,
        pages=math.ceil(total / page_size) if page_size else 0,
        period=period,
        items=items,
    )


@router.get("/filters")
def filter_options(db: Session = Depends(get_db)):
    """Distinct values for the Explorer filter controls."""
    def distinct(col):
        return sorted(
            r[0] for r in db.query(col).distinct().all() if r[0]
        )

    return {
        "states": distinct(Project.state),
        "sectors": distinct(Project.sector),
        "ministries": distinct(Project.ministry),
        "risk_levels": [r.value for r in RiskLevel],
        "periods": analytics.available_periods(db),
    }


@router.get("/{project_code}")
def digital_twin(project_code: str, db: Session = Depends(get_db)):
    """Everything known about one project — the Project Digital Twin payload."""
    project = db.query(Project).filter_by(project_code=project_code).one_or_none()
    if project is None:
        raise HTTPException(404, "Project not found.")

    snaps = (
        db.query(ProjectSnapshot)
        .filter_by(project_id=project.id, supersedes_id=None)
        .order_by(ProjectSnapshot.report_period)
        .all()
    )
    if not snaps:
        raise HTTPException(404, "No snapshots recorded for this project.")

    latest = snaps[-1]
    previous = snaps[-2] if len(snaps) > 1 else None
    scores = [s.risk_score for s in snaps]

    change = explain_change(previous, latest) if previous is not None else None
    active_drivers = latest.risk_drivers or []
    ranked_drivers = sorted(active_drivers, key=lambda d: d.get("contribution", 0), reverse=True)
    recommendations = []
    seen_actions = set()
    for driver in ranked_drivers:
        rec = RECOMMENDATIONS.get(driver.get("code"))
        if rec and driver.get("code") not in seen_actions:
            recommendations.append({
                "driver_code": driver.get("code"),
                "driver": driver.get("label"),
                **rec,
            })
            seen_actions.add(driver.get("code"))
    confidence = {
        "overall": round((latest.completeness or 0) / 100, 3) if latest.completeness is not None else latest.risk_confidence,
        "completeness_pct": latest.completeness,
        "quality_status": latest.data_quality_status,
        "source_verified": bool(latest.source and latest.record),
        "source": {
            "label": latest.source.report_label if latest.source else None,
            "document": latest.source.filename if latest.source else None,
            "page": latest.record.page_number if latest.record else None,
        },
    }

    alerts = (
        db.query(Alert)
        .filter_by(project_id=project.id)
        .order_by(Alert.created_at.desc())
        .all()
    )
    interventions = (
        db.query(Intervention)
        .filter_by(project_id=project.id)
        .order_by(Intervention.created_at.desc())
        .all()
    )
    issues = (
        db.query(DataQualityIssue)
        .filter_by(project_id=project.id, status="OPEN")
        .all()
    )
    events = (
        db.query(RiskEvent)
        .filter_by(project_id=project.id)
        .order_by(RiskEvent.to_period)
        .all()
    )

    return {
        "project": {
            "id": project.id,
            "project_code": project.project_code,
            "name": project.name,
            "agency": project.agency,
            "ministry": project.ministry,
            "sector": project.sector,
            "sector_origin": project.sector_origin.value if project.sector_origin else "UNKNOWN",
            "state": project.state,
            "state_origin": project.state_origin.value if project.state_origin else "UNKNOWN",
            "is_multi_state": project.is_multi_state,
            "states_list": project.states_list or [],
            "legacy_ocms_code": project.legacy_ocms_code,
            "pmgid": project.pmgid,
            "first_seen_period": project.first_seen_period,
            "last_seen_period": project.last_seen_period,
            "is_demo": project.is_demo,
        },
        "current": _snapshot_dict(latest),
        "snapshots": [_snapshot_dict(s) for s in snaps],
        "trend": {
            "direction": classify_trend(scores).value,
            "deterioration_streak": deterioration_streak(scores),
            "previous_period": previous.report_period if previous else None,
            "risk_delta": round(latest.risk_score - previous.risk_score, 2) if previous and None not in (latest.risk_score, previous.risk_score) else None,
            "series": [
                {
                    "period": s.report_period,
                    "risk_score": s.risk_score,
                    "physical_progress": s.physical_progress,
                    "financial_progress": s.financial_progress,
                    "expenditure_cr": s.expenditure,
                    "revised_cost_cr": s.revised_cost,
                    "schedule_delay_months": s.schedule_delay_months,
                }
                for s in snaps
            ],
        },
        "decision_support": {
            "what_changed": change,
            "top_drivers": ranked_drivers[:3],
            "recommendations": recommendations[:3],
            "data_confidence": confidence,
        },
        "risk_events": [
            {
                "from_period": e.from_period,
                "to_period": e.to_period,
                "from_score": e.from_score,
                "to_score": e.to_score,
                "delta": e.delta,
                "direction": e.direction.value if e.direction else None,
                "explanation": e.explanation,
            }
            for e in events
        ],
        "alerts": [
            {
                "id": a.id,
                "code": a.code,
                "title": a.title,
                "description": a.description,
                "severity": a.severity.value,
                "status": a.status.value,
                "trigger": a.trigger,
                "evidence": a.evidence,
                "report_period": a.report_period,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in alerts
        ],
        "interventions": [
            {
                "id": i.id,
                "reference": i.reference,
                "issue": i.issue,
                "severity": i.severity.value,
                "status": i.status.value,
                "owner": i.owner,
                "assigned_authority": i.assigned_authority,
                "due_date": i.due_date.isoformat() if i.due_date else None,
                "action": i.action,
                "remarks": i.remarks,
                "resolution": i.resolution,
                "created_at": i.created_at.isoformat() if i.created_at else None,
                "history": i.history or [],
            }
            for i in interventions
        ],
        "data_quality": [
            {
                "issue_type": d.issue_type,
                "field": d.field,
                "severity": d.severity.value,
                "description": d.description,
                "report_period": d.report_period,
            }
            for d in issues
        ],
    }


@router.get("/{project_code}/replay")
def escalation_replay(project_code: str, db: Session = Depends(get_db)):
    """Risk Escalation Replay: what changed at each step, and why."""
    project = db.query(Project).filter_by(project_code=project_code).one_or_none()
    if project is None:
        raise HTTPException(404, "Project not found.")

    snaps = (
        db.query(ProjectSnapshot)
        .filter_by(project_id=project.id, supersedes_id=None)
        .order_by(ProjectSnapshot.report_period)
        .all()
    )
    if len(snaps) < 2:
        return {
            "project_code": project_code,
            "frames": [_snapshot_dict(s) for s in snaps],
            "transitions": [],
            "note": "At least two reporting periods are required to replay an escalation.",
        }

    transitions = []
    for before, after in zip(snaps, snaps[1:]):
        transitions.append(explain_change(before, after))

    return {
        "project_code": project_code,
        "project_name": project.name,
        "frames": [
            {
                "period": s.report_period,
                "risk_score": s.risk_score,
                "risk_level": s.risk_level.value if s.risk_level else "UNKNOWN",
                "physical_progress": s.physical_progress,
                "financial_progress": s.financial_progress,
                "expenditure_cr": s.expenditure,
                "revised_cost_cr": s.revised_cost,
                "schedule_delay_months": s.schedule_delay_months,
                "drivers": s.risk_drivers or [],
                "source": {
                    "document": s.source.filename if s.source else None,
                    "label": s.source.report_label if s.source else None,
                    "page": s.record.page_number if s.record else None,
                },
            }
            for s in snaps
        ],
        "transitions": transitions,
    }


@router.get("/{project_code}/provenance/{field}")
def provenance(project_code: str, field: str, period: str | None = None,
               db: Session = Depends(get_db)):
    """Click-any-number provenance: trace a metric back to its source page."""
    project = db.query(Project).filter_by(project_code=project_code).one_or_none()
    if project is None:
        raise HTTPException(404, "Project not found.")

    q = db.query(ProjectSnapshot).filter_by(project_id=project.id, supersedes_id=None)
    if period:
        q = q.filter(ProjectSnapshot.report_period == period)
    snap = q.order_by(ProjectSnapshot.report_period.desc()).first()
    if snap is None:
        raise HTTPException(404, "No snapshot for that period.")

    FIELD_MAP = {
        "physical_progress": ("Physical Progress (%)", "col_7", "Percentage normalisation"),
        "original_cost": ("Original Cost (₹ crore)", "col_5", "Numeric normalisation, ₹ crore"),
        "revised_cost": ("Revised Cost (₹ crore)", "col_5", "Numeric normalisation, ₹ crore"),
        "expenditure": ("Cumulative Expenditure (₹ crore)", "col_6",
                        "Numeric normalisation, ₹ crore"),
        "state": ("State", "col_2", "Verbatim; Multi-State values parsed into a list"),
        "approval_date": ("Date of Approval", "col_3", "MM/YYYY to first day of month"),
        "original_completion": ("Original/Target DoC", "col_4",
                                "MM/YYYY to first day of month"),
        "revised_completion": ("Revised DoC", "col_4", "MM/YYYY to first day of month"),
    }

    derived = {
        "financial_progress": "expenditure / revised cost x 100",
        "cost_escalation_pct": "(revised cost - original cost) / original cost x 100",
        "progress_divergence": "financial progress - physical progress",
        "schedule_delay_months": "months between original and revised completion date",
        "risk_score": "composite of deterministic indicators (see risk drivers)",
    }

    record = snap.record
    source = snap.source

    if field in derived:
        return {
            "field": field,
            "value": getattr(snap, field, None),
            "origin": "DERIVED",
            "derivation": derived[field],
            "report_period": snap.report_period,
            "source_document": source.filename if source else None,
            "source_label": source.report_label if source else None,
            "source_page": record.page_number if record else None,
            "engine_version": snap.risk_engine_version,
            "note": "Computed by PAIMANA from extracted values on the cited source page.",
        }

    if field not in FIELD_MAP:
        raise HTTPException(400, f"No provenance mapping for field '{field}'.")

    label, raw_key, transformation = FIELD_MAP[field]
    raw_value = (record.raw_payload or {}).get(raw_key) if record else None
    confidence = None
    if record and record.field_confidence:
        confidence = record.field_confidence.get(field) or record.extraction_confidence

    return {
        "field": field,
        "value": (
            getattr(snap, field).isoformat()
            if hasattr(getattr(snap, field, None), "isoformat")
            else getattr(snap, field, None)
        ),
        "origin": "EXTRACTED",
        "source_document": source.filename if source else None,
        "source_label": source.report_label if source else None,
        "source_page": record.page_number if record else None,
        "source_table": record.table_name if record else None,
        "original_field": label,
        "raw_value": raw_value,
        "transformation": transformation,
        "extraction_confidence": confidence,
        "extraction_date": (
            record.extracted_at.isoformat() if record and record.extracted_at else None
        ),
        "parse_warnings": record.parse_warnings if record else [],
        "report_period": snap.report_period,
    }
