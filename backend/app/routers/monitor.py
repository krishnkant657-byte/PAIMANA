"""National Monitor, Analytics Studio, Early Warnings, Intervention Centre."""
from __future__ import annotations

import datetime as dt
import secrets

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import (
    Alert,
    AlertStatus,
    Intervention,
    InterventionStatus,
    Project,
    ProjectSnapshot,
    RiskLevel,
    Severity,
    User,
)
from ..schemas import AlertUpdate, InterventionCreate, InterventionUpdate
from ..security import require_analyst
from ..services import analytics
from ..services.audit import record as audit_record

router = APIRouter(prefix="/api", tags=["monitor"])


# ---------------------------------------------------------------------------
# National Monitor
# ---------------------------------------------------------------------------
@router.get("/monitor/summary")
def national_summary(period: str | None = None, db: Session = Depends(get_db)):
    return analytics.national_summary(db, period)


@router.get("/monitor/trend")
def national_trend(db: Session = Depends(get_db)):
    return {"series": analytics.national_trend(db)}


@router.get("/monitor/by-state")
def by_state(period: str | None = None, db: Session = Depends(get_db)):
    return {"period": period or analytics.latest_period(db), "groups": analytics.by_state(db, period)}


@router.get("/monitor/by-sector")
def by_sector(period: str | None = None, db: Session = Depends(get_db)):
    return {
        "period": period or analytics.latest_period(db),
        "groups": analytics.by_sector(db, period),
    }


@router.get("/monitor/by-ministry")
def by_ministry(period: str | None = None, db: Session = Depends(get_db)):
    return {
        "period": period or analytics.latest_period(db),
        "groups": analytics.by_ministry(db, period),
    }


@router.get("/monitor/map")
def map_data(period: str | None = None, db: Session = Depends(get_db)):
    """State-level aggregates for the India map.

    Deliberately state-level, not point-level: the Flash Report records a state
    (sometimes a multi-state group) and no coordinates. Plotting pins would
    imply a geographic precision the source does not support.
    """
    period = period or analytics.latest_period(db)
    groups = analytics.by_state(db, period)

    single, multi = [], []
    for g in groups:
        (multi if str(g["key"]).upper().startswith("MULTI-STATES") else single).append(g)

    for g in groups:
        snaps = (
            db.query(func.count(ProjectSnapshot.id))
            .join(Project, Project.id == ProjectSnapshot.project_id)
            .filter(
                Project.state == g["key"],
                ProjectSnapshot.report_period == period,
                ProjectSnapshot.supersedes_id.is_(None),
                ProjectSnapshot.risk_level.in_([RiskLevel.SEVERE, RiskLevel.HIGH]),
            )
            .scalar()
        )
        g["at_risk"] = snaps

    return {
        "period": period,
        "states": sorted(single, key=lambda g: (g.get("avg_risk_score") or 0, g["projects"]), reverse=True),
        "multi_state_groups": sorted(multi, key=lambda g: (g.get("avg_risk_score") or 0, g["projects"]), reverse=True),
        "geographic_note": (
            "Aggregated at state level. The source Flash Report records an administrative "
            "state, not project coordinates, so no point locations are shown."
        ),
    }


# ---------------------------------------------------------------------------
# Analytics Studio
# ---------------------------------------------------------------------------
@router.get("/analytics/compare")
def compare(
    dimension: str = Query(..., pattern="^(state|sector|ministry)$"),
    values: str = Query(..., description="comma-separated values"),
    period: str | None = None,
    db: Session = Depends(get_db),
):
    parsed = [v.strip() for v in values.split(",") if v.strip()][:4]
    if len(parsed) < 2:
        raise HTTPException(400, "Provide at least two values to compare.")
    try:
        return analytics.compare(db, dimension, parsed, period)
    except ValueError as exc:
        raise HTTPException(400, str(exc))


@router.get("/analytics/leaderboard")
def leaderboard(
    metric: str = Query("cost_exposure_cr"),
    dimension: str = Query("state", pattern="^(state|sector|ministry)$"),
    period: str | None = None,
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    fn = {
        "state": analytics.by_state,
        "sector": analytics.by_sector,
        "ministry": analytics.by_ministry,
    }[dimension]
    groups = fn(db, period)
    valid = {
        "cost_exposure_cr", "projects", "avg_risk_score",
        "avg_delay_months", "avg_physical_progress", "expenditure_cr",
    }
    if metric not in valid:
        raise HTTPException(400, f"metric must be one of {sorted(valid)}")
    groups = [g for g in groups if g.get(metric) is not None]
    groups.sort(key=lambda g: g[metric], reverse=True)
    return {"dimension": dimension, "metric": metric, "groups": groups[:limit]}


# ---------------------------------------------------------------------------
# Early Warnings
# ---------------------------------------------------------------------------
@router.get("/alerts")
def list_alerts(
    db: Session = Depends(get_db),
    status: str | None = None,
    severity: str | None = None,
    code: str | None = None,
    period: str | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    q = db.query(Alert, Project).join(Project, Project.id == Alert.project_id)
    if status:
        q = q.filter(Alert.status == AlertStatus(status.upper()))
    if severity:
        q = q.filter(Alert.severity == Severity(severity.upper()))
    if code:
        q = q.filter(Alert.code == code)
    if period:
        q = q.filter(Alert.report_period == period)

    total = q.count()
    severity_order = {
        Severity.CRITICAL: 0, Severity.HIGH: 1, Severity.MEDIUM: 2,
        Severity.LOW: 3, Severity.INFO: 4,
    }
    rows = q.order_by(Alert.created_at.desc()).offset(offset).limit(limit).all()
    items = [
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
            "project": {
                "project_code": p.project_code,
                "name": p.name,
                "state": p.state,
                "sector": p.sector,
            },
        }
        for a, p in rows
    ]
    items.sort(key=lambda i: severity_order.get(Severity(i["severity"]), 9))

    counts = dict(
        db.query(Alert.code, func.count(Alert.id))
        .filter(Alert.status == AlertStatus.OPEN)
        .group_by(Alert.code)
        .all()
    )
    return {"total": total, "items": items, "open_by_code": counts}


@router.patch("/alerts/{alert_id}")
def update_alert(
    alert_id: int,
    payload: AlertUpdate,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst),
):
    alert = db.get(Alert, alert_id)
    if alert is None:
        raise HTTPException(404, "Alert not found.")
    previous = alert.status.value
    alert.status = AlertStatus(payload.status)
    if alert.status == AlertStatus.ACKNOWLEDGED and alert.acknowledged_at is None:
        alert.acknowledged_at = dt.datetime.now(dt.timezone.utc)
    if alert.status == AlertStatus.CLOSED:
        alert.closed_at = dt.datetime.now(dt.timezone.utc)
    db.commit()

    audit_record(
        db, actor=user.username, action="ALERT_STATUS_CHANGED", object_type="Alert",
        object_id=alert.id, previous_value={"status": previous},
        new_value={"status": alert.status.value}, request=request,
    )
    return {"id": alert.id, "status": alert.status.value}


# ---------------------------------------------------------------------------
# Intervention Centre
# ---------------------------------------------------------------------------
def _intervention_dict(i: Intervention, project: Project | None = None) -> dict:
    return {
        "id": i.id,
        "reference": i.reference,
        "issue": i.issue,
        "severity": i.severity.value,
        "status": i.status.value,
        "assigned_authority": i.assigned_authority,
        "owner": i.owner,
        "due_date": i.due_date.isoformat() if i.due_date else None,
        "action": i.action,
        "remarks": i.remarks,
        "resolution": i.resolution,
        "alert_id": i.alert_id,
        "created_by": i.created_by,
        "created_at": i.created_at.isoformat() if i.created_at else None,
        "updated_at": i.updated_at.isoformat() if i.updated_at else None,
        "history": i.history or [],
        "project": (
            {
                "project_code": project.project_code,
                "name": project.name,
                "state": project.state,
                "sector": project.sector,
            }
            if project
            else None
        ),
    }


@router.get("/interventions")
def list_interventions(
    db: Session = Depends(get_db),
    status: str | None = None,
    project_code: str | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    q = db.query(Intervention, Project).join(Project, Project.id == Intervention.project_id)
    if status:
        q = q.filter(Intervention.status == InterventionStatus(status.upper()))
    if project_code:
        q = q.filter(Project.project_code == project_code)
    rows = q.order_by(Intervention.created_at.desc()).limit(limit).all()

    counts = dict(
        db.query(Intervention.status, func.count(Intervention.id))
        .group_by(Intervention.status)
        .all()
    )
    return {
        "total": q.count(),
        "items": [_intervention_dict(i, p) for i, p in rows],
        "by_status": {k.value: v for k, v in counts.items()},
    }


@router.post("/interventions", status_code=201)
def create_intervention(
    payload: InterventionCreate,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst),
):
    project = db.query(Project).filter_by(project_code=payload.project_code).one_or_none()
    if project is None:
        raise HTTPException(404, "Project not found.")

    if payload.alert_id is not None:
        alert = db.get(Alert, payload.alert_id)
        if alert is None or alert.project_id != project.id:
            raise HTTPException(400, "Alert does not belong to this project.")

    reference = f"INT-{dt.datetime.now().strftime('%Y%m')}-{secrets.token_hex(3).upper()}"
    now = dt.datetime.now(dt.timezone.utc).isoformat()

    intervention = Intervention(
        reference=reference,
        project_id=project.id,
        alert_id=payload.alert_id,
        issue=payload.issue,
        severity=Severity(payload.severity),
        assigned_authority=payload.assigned_authority,
        owner=payload.owner,
        due_date=payload.due_date,
        action=payload.action,
        remarks=payload.remarks,
        status=InterventionStatus.ASSIGNED if payload.owner else InterventionStatus.CREATED,
        created_by=user.username,
        history=[
            {
                "at": now,
                "by": user.username,
                "from": None,
                "to": "ASSIGNED" if payload.owner else "CREATED",
                "note": "Intervention raised.",
            }
        ],
    )
    db.add(intervention)
    db.commit()

    audit_record(
        db, actor=user.username, action="INTERVENTION_CREATED", object_type="Intervention",
        object_id=intervention.id, new_value={"reference": reference, "issue": payload.issue},
        meta={"project_code": project.project_code}, request=request,
    )
    return _intervention_dict(intervention, project)


@router.patch("/interventions/{reference}")
def update_intervention(
    reference: str,
    payload: InterventionUpdate,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst),
):
    intervention = db.query(Intervention).filter_by(reference=reference).one_or_none()
    if intervention is None:
        raise HTTPException(404, "Intervention not found.")

    previous = {
        "status": intervention.status.value,
        "owner": intervention.owner,
        "due_date": intervention.due_date.isoformat() if intervention.due_date else None,
    }
    now = dt.datetime.now(dt.timezone.utc)
    history = list(intervention.history or [])

    if payload.status and payload.status != intervention.status.value:
        history.append(
            {
                "at": now.isoformat(),
                "by": user.username,
                "from": intervention.status.value,
                "to": payload.status,
                "note": payload.remarks or "",
            }
        )
        intervention.status = InterventionStatus(payload.status)
        if intervention.status == InterventionStatus.RESOLVED:
            intervention.resolved_at = now
        if intervention.status == InterventionStatus.CLOSED:
            intervention.closed_at = now

    for field in ("owner", "assigned_authority", "due_date", "action", "remarks", "resolution"):
        value = getattr(payload, field)
        if value is not None:
            setattr(intervention, field, value)

    intervention.history = history
    db.commit()

    audit_record(
        db, actor=user.username, action="INTERVENTION_UPDATED", object_type="Intervention",
        object_id=intervention.id, previous_value=previous,
        new_value={"status": intervention.status.value, "owner": intervention.owner},
        request=request,
    )
    project = db.get(Project, intervention.project_id)
    return _intervention_dict(intervention, project)
