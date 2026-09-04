"""Conversational assistant with attachments.

This router is additive. The original ``POST /api/assistant/ask`` is untouched
and still serves the stateless, verified-only path that the rest of the platform
(and the Digital Twin panel) depends on.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import ChatAttachment, ChatMessage, Conversation, Project, User
from ..schemas import ChatMessageRequest
from ..security import (
    client_key,
    get_optional_user,
    rate_limit,
    resolve_within,
    safe_attachment_name,
    validate_attachment_bytes,
)
from ..services import agent, conversation as convo_service, files
from ..services.audit import record as audit_record

log = logging.getLogger("paimana.chat")
router = APIRouter(prefix="/api/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------
@router.get("/capabilities")
def capabilities():
    from ..services import assistant

    return {
        "llm_enabled": settings.llm_enabled,
        "supported_questions": assistant.SUPPORTED,
        "attachments": {
            "enabled": True,
            "max_files_per_message": settings.max_attachments_per_message,
            "max_file_mb": settings.max_attachment_bytes // (1024 * 1024),
            "accepted_extensions": files.SUPPORTED_EXTENSIONS,
            "fully_analysed_extensions": files.FULLY_ANALYSED_EXTENSIONS,
        },
        "modes": {
            agent.SRC_PAIMANA: "Verified project data from ingested Flash Report snapshots.",
            agent.SRC_GENERAL: "General knowledge from the language model, not PAIMANA data.",
            agent.SRC_FILE: "Analysis of a file you uploaded.",
            agent.SRC_MIXED: "Your file compared against verified PAIMANA data.",
        },
        "note": (
            "General conversation requires a configured language model. Verified PAIMANA "
            "answers and all file analysis are computed by the platform and work without one."
            if not settings.llm_enabled else
            "A language model is configured. It explains computed results; it never produces figures."
        ),
    }


# ---------------------------------------------------------------------------
# Attachments
# ---------------------------------------------------------------------------
def _attachment_payload(att: ChatAttachment) -> dict:
    analysis = att.analysis or {}
    return {
        "id": att.id,
        "filename": att.filename,
        "size_bytes": att.size_bytes,
        "kind": att.detected_kind,
        "label": att.detected_label,
        "status": att.status,
        "error": att.error,
        "recovery": analysis.get("recovery"),
        "warnings": analysis.get("warnings", []),
        "summary_facts": analysis.get("summary_facts", {}),
        "quality": (analysis.get("quality") or {}).get("overall"),
        "extension_mismatch": analysis.get("detection", {}).get(
            "extension_signature_mismatch", False
        ),
        "prompt_injection_detected": bool(analysis.get("prompt_injection_detected")),
        "created_at": att.created_at.isoformat() if att.created_at else None,
    }


@router.post("/attachments", status_code=status.HTTP_201_CREATED)
async def upload_attachment(
    request: Request,
    file: UploadFile = File(...),
    conversation_id: str | None = Form(default=None),
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    """Accept one file, analyse it immediately, and return what was found.

    Analysis happens at upload time so the UI can show the user what the
    platform actually managed to read before they ask a question about it.
    """
    rate_limit(
        f"chat-upload:{client_key(request, user)}",
        limit=settings.upload_rate_limit,
        window_seconds=settings.upload_rate_window,
    )

    convo = convo_service.get_or_create(
        db, conversation_id, owner=user.username if user else None
    )

    live = (
        db.query(ChatAttachment)
        .filter_by(conversation_id=convo.id, deleted_at=None)
        .count()
    )
    if live >= settings.max_attachments_per_message * 4:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This conversation already holds the maximum number of attachments. "
            "Remove some before adding more.",
        )

    data = await file.read()
    validate_attachment_bytes(data, file.filename or "")

    display_name, stored_name = safe_attachment_name(file.filename or "attachment")
    digest = hashlib.sha256(data).hexdigest()

    # Analyse before writing anything, so a file we cannot use never lands on disk.
    try:
        analysis = files.analyse(display_name, data, file.content_type)
    except Exception:
        log.exception("Attachment analysis crashed for %s", display_name)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "The file could not be processed. Please try a different file or format.",
        )

    target = resolve_within(settings.chat_upload_dir, stored_name)
    try:
        target.write_bytes(data)
    except OSError:
        log.exception("Could not persist attachment %s", stored_name)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "The file could not be stored on the server. Please try again.",
        )

    attachment = ChatAttachment(
        conversation_id=convo.id,
        filename=display_name,
        stored_name=stored_name,
        sha256=digest,
        size_bytes=len(data),
        declared_mime=file.content_type,
        detected_kind=analysis.get("detection", {}).get("kind"),
        detected_label=analysis.get("detection", {}).get("label"),
        status=analysis.get("status", files.FAILED),
        error=analysis.get("error"),
        analysis=analysis,
        uploaded_by=user.username if user else None,
    )
    db.add(attachment)
    db.commit()
    db.refresh(attachment)

    audit_record(
        db, actor=user.username if user else None, action="CHAT_ATTACHMENT_UPLOADED",
        object_type="ChatAttachment", object_id=attachment.id,
        meta={"kind": attachment.detected_kind, "status": attachment.status,
              "bytes": attachment.size_bytes},
        request=request,
    )

    return {"conversation_id": convo.public_id, "attachment": _attachment_payload(attachment)}


@router.delete("/attachments/{attachment_id}")
def remove_attachment(
    attachment_id: int,
    request: Request,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    attachment = db.get(ChatAttachment, attachment_id)
    if attachment is None or attachment.deleted_at is not None:
        raise HTTPException(404, "Attachment not found.")

    attachment.deleted_at = dt.datetime.now(dt.timezone.utc)
    try:
        resolve_within(settings.chat_upload_dir, attachment.stored_name).unlink(missing_ok=True)
    except (OSError, HTTPException):
        log.warning("Could not delete attachment file %s", attachment.stored_name)
    db.commit()

    audit_record(
        db, actor=user.username if user else None, action="CHAT_ATTACHMENT_REMOVED",
        object_type="ChatAttachment", object_id=attachment_id, request=request,
    )
    return {"removed": attachment_id}


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------
@router.post("/message")
def send_message(
    payload: ChatMessageRequest,
    request: Request,
    db: Session = Depends(get_db),
    user: User | None = Depends(get_optional_user),
):
    rate_limit(
        f"chat:{client_key(request, user)}",
        limit=settings.chat_rate_limit,
        window_seconds=settings.chat_rate_window,
    )

    convo = convo_service.get_or_create(
        db, payload.conversation_id, owner=user.username if user else None
    )

    attachments: list[ChatAttachment] = []
    if payload.attachment_ids:
        attachments = (
            db.query(ChatAttachment)
            .filter(
                ChatAttachment.id.in_(payload.attachment_ids[:settings.max_attachments_per_message]),
                ChatAttachment.conversation_id == convo.id,
                ChatAttachment.deleted_at.is_(None),
            )
            .all()
        )
        missing = set(payload.attachment_ids) - {a.id for a in attachments}
        if missing:
            raise HTTPException(
                400,
                "One or more attachments are no longer available. Please re-attach them.",
            )

    project_scope = None
    if payload.project_code:
        project_scope = (
            db.query(Project).filter_by(project_code=payload.project_code).one_or_none()
        )
        if project_scope is None:
            raise HTTPException(404, "Project not found.")

    convo_service.record_message(
        db, convo, role="user", content=payload.message,
        attachment_ids=[a.id for a in attachments],
    )

    try:
        outcome = agent.respond(db, convo, payload.message, attachments, project_scope)
    except Exception:
        log.exception("Agent failed on conversation %s", convo.public_id)
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "Something went wrong while working on that. Please try rephrasing, or try "
            "again in a moment.",
        )

    message = convo_service.record_message(
        db, convo, role="assistant", content=outcome["answer"],
        source=outcome["source"], intent=outcome.get("intent"),
        project_codes=outcome.get("project_codes"),
        attachment_ids=[a.id for a in attachments],
        analysis_meta={
            "route": outcome.get("route"),
            "route_reason": outcome.get("route_reason"),
            "resolved": outcome.get("resolved"),
        },
    )

    if not convo.title:
        convo.title = payload.message[:80]
        db.commit()

    audit_record(
        db, actor=user.username if user else None, action="CHAT_MESSAGE",
        object_type="Conversation", object_id=convo.public_id,
        meta={"route": outcome.get("route"), "intent": outcome.get("intent"),
              "resolved": outcome.get("resolved"),
              "attachments": len(attachments)},
        request=request,
    )

    return {
        "conversation_id": convo.public_id,
        "message_id": message.id,
        "answer": outcome["answer"],
        "source": outcome["source"],
        "route": outcome["route"],
        "route_reason": outcome["route_reason"],
        "intent": outcome.get("intent"),
        "resolved": outcome.get("resolved"),
        "grounding_note": outcome.get("grounding_note"),
        "verified_result": outcome.get("verified_result"),
        "cross_check": outcome.get("cross_check"),
        "project_codes": outcome.get("project_codes"),
        "context": {
            "current_project": outcome["context"].get("current_project"),
            "recent_projects": outcome["context"].get("recent_projects", [])[:5],
            "period": outcome["context"].get("period"),
            "attachments": outcome["context"].get("attachments", []),
        },
    }


@router.get("/conversations/{public_id}")
def get_conversation(public_id: str, db: Session = Depends(get_db)):
    convo = db.query(Conversation).filter_by(public_id=public_id).one_or_none()
    if convo is None:
        raise HTTPException(404, "Conversation not found.")

    messages = (
        db.query(ChatMessage)
        .filter_by(conversation_id=convo.id)
        .order_by(ChatMessage.id)
        .all()
    )
    attachments = (
        db.query(ChatAttachment)
        .filter_by(conversation_id=convo.id, deleted_at=None)
        .all()
    )
    return {
        "conversation_id": convo.public_id,
        "title": convo.title,
        "context": convo.context,
        "messages": [
            {
                "id": m.id, "role": m.role, "content": m.content, "source": m.source,
                "intent": m.intent, "project_codes": m.project_codes,
                "attachment_ids": m.attachment_ids,
                "meta": m.analysis_meta,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
        "attachments": [_attachment_payload(a) for a in attachments],
    }


@router.post("/conversations")
def start_conversation(
    db: Session = Depends(get_db), user: User | None = Depends(get_optional_user)
):
    convo = convo_service.get_or_create(db, None, owner=user.username if user else None)
    return {"conversation_id": convo.public_id}
