"""Audit trail helper.

Called from routers on every state-changing action, plus logins, report
generation and assistant queries.
"""
from __future__ import annotations

import datetime as dt

from fastapi import Request
from sqlalchemy.orm import Session

from ..models import AuditLog


def _jsonable(value):
    if value is None:
        return None
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if hasattr(value, "value"):          # Enum
        return value.value
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def record(
    db: Session,
    *,
    actor: str | None,
    action: str,
    object_type: str | None = None,
    object_id: str | int | None = None,
    previous_value=None,
    new_value=None,
    meta: dict | None = None,
    request: Request | None = None,
    commit: bool = True,
) -> AuditLog:
    entry = AuditLog(
        actor=actor or "anonymous",
        action=action,
        object_type=object_type,
        object_id=str(object_id) if object_id is not None else None,
        previous_value=_jsonable(previous_value),
        new_value=_jsonable(new_value),
        meta=_jsonable(meta),
        ip_address=(request.client.host if request and request.client else None),
    )
    db.add(entry)
    if commit:
        db.commit()
    return entry
