"""Auth, Assistant, Scenario Lab, Data Quality, Reports, Audit, Ingestion."""
from __future__ import annotations

import datetime as dt

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from fastapi.responses import FileResponse, Response
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import (
    AuditLog,
    DataQualityIssue,
    DataSource,
    Project,
    ProjectSnapshot,
    Report,
    User,
    UserRole,
)
from ..schemas import (
    AssistantQuery,
    LoginRequest,
    ReportRequest,
    ScenarioRequest,
    SimulatorRequest,
    TokenResponse,
)
from ..security import (
    client_key,
    create_token,
    get_current_user,
    get_optional_user,
    rate_limit,
    require_admin,
    require_analyst,
    safe_filename,
    validate_pdf_bytes,
    verify_password,
)
from ..services import analytics, assistant, reports, scenario
from ..services.audit import record as audit_record

router = APIRouter(prefix="/api", tags=["platform"])


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@router.post("/auth/login", response_model=TokenResponse)
def login(payload: LoginRequest, request: Request, db: Session = Depends(get_db)):
    rate_limit(f"login:{client_key(request)}", limit=10, window_seconds=300)

    user = db.query(User).filter_by(username=payload.username).one_or_none()
    if user is None or not user.is_active or not verify_password(payload.password, user.password_hash):
        # Identical message for both cases — no account enumeration.
        audit_record(
            db, actor=payload.username, action="LOGIN_FAILED",
            object_type="User", request=request,
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid username or password.")

    user.last_login_at = dt.datetime.now(dt.timezone.utc)
    db.commit()
    audit_record(db, actor=user.username, action="LOGIN", object_type="User",
                 object_id=user.id, request=request)

    return TokenResponse(
        access_token=create_token(user),
        username=user.username,
        full_name=user.full_name,
        designation=user.designation,
        role=user.role.value,
        expires_in_minutes=settings.token_ttl_minutes,
    )


@router.get("/auth/me")
def whoami(user: User = Depends(get_current_user)):
    return {
        "username": user.username,
        "full_name": user.full_name,
        "designation": user.designation,
        "role": user.role.value,
    }


# ---------------------------------------------------------------------------
# Assistant
# ---------------------------------------------------------------------------
@router.post("/assistant/ask")
def ask(
    payload: AssistantQuery,
    request: Request,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    rate_limit(
        f"assistant:{client_key(request, user)}",
        limit=settings.assistant_rate_limit,
        window_seconds=settings.assistant_rate_window,
    )

    project = None
    if payload.project_code:
        project = db.query(Project).filter_by(project_code=payload.project_code).one_or_none()
        if project is None:
            raise HTTPException(404, "Project not found.")

    result = assistant.answer(db, payload.question, project_scope=project)

    audit_record(
        db, actor=user.username if user else None, action="ASSISTANT_QUERY",
        object_type="Project" if project else None,
        object_id=project.project_code if project else None,
        meta={"intent": result["intent"], "resolved": result["resolved"]},
        request=request,
    )
    return result


@router.get("/assistant/capabilities")
def capabilities():
    return {
        "supported_questions": assistant.SUPPORTED,
        "llm_enabled": settings.llm_enabled,
        "grounding": (
            "Questions are resolved to structured database queries first. The language "
            "model only rephrases the verified result and cannot introduce figures."
        ),
    }


# ---------------------------------------------------------------------------
# Scenario Lab + Pre-Approval Simulator
# ---------------------------------------------------------------------------
@router.post("/scenario/run")
def run_scenario(payload: ScenarioRequest, db: Session = Depends(get_db)):
    project = db.query(Project).filter_by(project_code=payload.project_code).one_or_none()
    if project is None:
        raise HTTPException(404, "Project not found.")
    result = scenario.run_scenario(
        db,
        project,
        cost_change_pct=payload.cost_change_pct,
        schedule_change_days=payload.schedule_change_days,
        progress_change_pp=payload.progress_change_pp,
        expenditure_change_pct=payload.expenditure_change_pct,
    )
    if "error" in result:
        raise HTTPException(400, result["error"])
    return result


@router.post("/simulator/pre-approval")
def pre_approval(payload: SimulatorRequest, db: Session = Depends(get_db)):
    return scenario.pre_approval_simulation(
        db,
        estimated_cost_cr=payload.estimated_cost_cr,
        duration_months=payload.duration_months,
        sector=payload.sector,
        state=payload.state,
        expected_progress_year1=payload.expected_progress_year1,
    )


# ---------------------------------------------------------------------------
# Data Quality Centre
# ---------------------------------------------------------------------------
@router.get("/data-quality/summary")
def dq_summary(db: Session = Depends(get_db)):
    return analytics.data_quality_summary(db)


@router.get("/data-quality/issues")
def dq_issues(
    db: Session = Depends(get_db),
    issue_type: str | None = None,
    severity: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    q = (
        db.query(DataQualityIssue, Project)
        .outerjoin(Project, Project.id == DataQualityIssue.project_id)
        .filter(DataQualityIssue.status == "OPEN")
    )
    if issue_type:
        q = q.filter(DataQualityIssue.issue_type == issue_type)
    if severity:
        q = q.filter(DataQualityIssue.severity == severity.upper())

    total = q.count()
    rows = q.order_by(DataQualityIssue.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [
            {
                "id": d.id,
                "issue_type": d.issue_type,
                "field": d.field,
                "severity": d.severity.value,
                "description": d.description,
                "detail": d.detail,
                "report_period": d.report_period,
                "project": (
                    {"project_code": p.project_code, "name": p.name} if p else None
                ),
            }
            for d, p in rows
        ],
    }


@router.get("/data-quality/sources")
def dq_sources(db: Session = Depends(get_db)):
    rows = db.query(DataSource).order_by(DataSource.report_period).all()
    out = []
    for s in rows:
        snaps = (
            db.query(func.count(ProjectSnapshot.id)).filter_by(source_id=s.id).scalar()
        )
        out.append(
            {
                "id": s.id,
                "filename": s.filename,
                "sha256": s.sha256,
                "report_period": s.report_period,
                "report_label": s.report_label,
                "issue_number": s.issue_number,
                "page_count": s.page_count,
                "publisher": s.publisher,
                "snapshots": snaps,
                "is_demo": s.is_demo,
                "ingested_at": s.ingested_at.isoformat() if s.ingested_at else None,
                "ingested_by": s.ingested_by,
            }
        )
    return {"sources": out}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
@router.post("/ingest/upload")
async def upload_report(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst),
):
    """Upload a Flash Report PDF for ingestion.

    Validation: extension, MIME type, magic bytes, size cap, and a sanitised
    filename that cannot escape the upload directory.
    """
    from ..services.ingestion import ingest_pdf  # local import keeps startup light

    if file.content_type not in settings.allowed_upload_mime:
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")

    data = await file.read()
    validate_pdf_bytes(data)

    target = settings.upload_dir / safe_filename(file.filename or "report.pdf")
    target.write_bytes(data)

    try:
        result = ingest_pdf(db, target, actor=user.username)
    except Exception:
        target.unlink(missing_ok=True)
        # Never leak a stack trace to the client.
        raise HTTPException(500, "Ingestion failed. The document was not accepted.")

    audit_record(
        db, actor=user.username, action="REPORT_INGESTED", object_type="DataSource",
        object_id=result.get("source_id"), new_value=result, request=request,
    )

    if result.get("status") == "OK":
        from ..services import risk_engine
        from ..services.warnings import generate_all_alerts

        risk_engine.recalculate_all(db)
        generate_all_alerts(db)

    return result


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------
@router.post("/reports/generate")
def generate_report(
    payload: ReportRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(require_analyst),
):
    try:
        path, title = reports.generate(
            db,
            report_type=payload.report_type,
            project_code=payload.project_code,
            period=payload.period,
            generated_by=user.full_name or user.username,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))

    entry = Report(
        report_type=payload.report_type,
        title=title,
        report_period=payload.period,
        filename=path.name,
        parameters=payload.model_dump(),
        generated_by=user.username,
    )
    db.add(entry)
    db.commit()

    audit_record(
        db, actor=user.username, action="REPORT_GENERATED", object_type="Report",
        object_id=entry.id, new_value={"type": payload.report_type, "file": path.name},
        request=request,
    )
    return {"id": entry.id, "title": title, "filename": path.name,
            "download_url": f"/api/reports/{entry.id}/download"}


@router.get("/reports")
def list_reports(db: Session = Depends(get_db), limit: int = Query(50, ge=1, le=200)):
    rows = db.query(Report).order_by(Report.generated_at.desc()).limit(limit).all()
    return {
        "items": [
            {
                "id": r.id,
                "report_type": r.report_type,
                "title": r.title,
                "report_period": r.report_period,
                "filename": r.filename,
                "generated_by": r.generated_by,
                "generated_at": r.generated_at.isoformat() if r.generated_at else None,
                "download_url": f"/api/reports/{r.id}/download",
            }
            for r in rows
        ]
    }


@router.get("/reports/{report_id}/download")
def download_report(
    report_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    entry = db.get(Report, report_id)
    if entry is None:
        raise HTTPException(404, "Report not found.")

    # Resolve inside the reports directory and confirm containment.
    path = (reports.REPORT_DIR / entry.filename).resolve()
    if not str(path).startswith(str(reports.REPORT_DIR.resolve())) or not path.exists():
        raise HTTPException(404, "Report file is no longer available.")

    audit_record(
        db, actor=user.username, action="REPORT_DOWNLOADED", object_type="Report",
        object_id=report_id, request=request,
    )
    return FileResponse(path, media_type="application/pdf", filename=entry.filename)


@router.get("/reports/{report_id}/download_csv")
def download_report_csv(
    report_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    entry = db.get(Report, report_id)
    if entry is None:
        raise HTTPException(404, "Report not found.")

    params = entry.parameters or {}
    csv_content, filename = reports.generate_csv(
        db,
        report_type=entry.report_type,
        project_code=params.get("project_code"),
        period=entry.report_period or params.get("period"),
    )

    audit_record(
        db, actor=user.username, action="REPORT_DOWNLOADED_CSV", object_type="Report",
        object_id=report_id, request=request,
    )
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Audit trail
# ---------------------------------------------------------------------------
@router.get("/audit")
def audit_trail(
    db: Session = Depends(get_db),
    user: User = Depends(require_admin),
    action: str | None = None,
    actor: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    q = db.query(AuditLog)
    if action:
        q = q.filter(AuditLog.action == action)
    if actor:
        q = q.filter(AuditLog.actor == actor)
    total = q.count()
    rows = q.order_by(AuditLog.created_at.desc()).offset(offset).limit(limit).all()
    return {
        "total": total,
        "items": [
            {
                "id": a.id,
                "actor": a.actor,
                "action": a.action,
                "object_type": a.object_type,
                "object_id": a.object_id,
                "previous_value": a.previous_value,
                "new_value": a.new_value,
                "meta": a.meta,
                "ip_address": a.ip_address,
                "created_at": a.created_at.isoformat() if a.created_at else None,
            }
            for a in rows
        ],
    }
