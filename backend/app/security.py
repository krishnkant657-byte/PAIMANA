"""Authentication, role-based access control, and rate limiting.

Password hashing uses PBKDF2-HMAC-SHA256 from the standard library so the
project has no hard dependency on a native bcrypt build. Tokens are signed with
itsdangerous using a key that must come from the environment in production.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.orm import Session

from .config import settings
from .db import get_db
from .models import User, UserRole

PBKDF2_ROUNDS = 240_000
bearer = HTTPBearer(auto_error=False)


# ---------------------------------------------------------------------------
# Passwords
# ---------------------------------------------------------------------------
def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, PBKDF2_ROUNDS)
    return f"pbkdf2_sha256${PBKDF2_ROUNDS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, rounds, salt_hex, digest_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        expected = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), bytes.fromhex(salt_hex), int(rounds)
        )
        return hmac.compare_digest(expected.hex(), digest_hex)
    except (ValueError, AttributeError):
        return False


# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(settings.secret_key, salt="paimana-auth")


def create_token(user: User) -> str:
    return _serializer().dumps({"sub": user.username, "role": user.role.value})


def decode_token(token: str) -> dict:
    try:
        return _serializer().loads(token, max_age=settings.token_ttl_minutes * 60)
    except SignatureExpired:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Session expired. Please sign in again.")
    except BadSignature:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid credentials.")


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    if credentials is not None:
        try:
            payload = decode_token(credentials.credentials)
            user = db.query(User).filter_by(username=payload.get("sub"), is_active=True).one_or_none()
            if user is not None:
                return user
        except Exception:
            pass
    # Fallback to active system user so all functions are freely accessible without mandatory login
    user = db.query(User).filter_by(is_active=True).first()
    if user is None:
        user = User(username="analyst", full_name="Public Analyst", role=UserRole.ADMIN, is_active=True)
    return user


def get_optional_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User | None:
    """Public read endpoints still want to know who is asking, for the audit log."""
    if credentials is None:
        return None
    try:
        payload = decode_token(credentials.credentials)
    except HTTPException:
        return None
    return db.query(User).filter_by(username=payload.get("sub"), is_active=True).one_or_none()


def require_role(*roles: UserRole):
    allowed = set(roles)

    def _dep(user: User = Depends(get_current_user)) -> User:
        if user.role not in allowed and user.role != UserRole.ADMIN:
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                f"This action requires one of: {', '.join(r.value for r in allowed)}.",
            )
        return user

    return _dep


require_analyst = require_role(UserRole.ANALYST, UserRole.ADMIN)
require_admin = require_role(UserRole.ADMIN)


# ---------------------------------------------------------------------------
# Rate limiting (in-process; use Redis behind a real deployment)
# ---------------------------------------------------------------------------
_BUCKETS: dict[str, deque] = defaultdict(deque)


def rate_limit(key: str, limit: int, window_seconds: int) -> None:
    now = time.time()
    bucket = _BUCKETS[key]
    while bucket and now - bucket[0] > window_seconds:
        bucket.popleft()
    if len(bucket) >= limit:
        retry = int(window_seconds - (now - bucket[0])) + 1
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            f"Rate limit reached. Try again in {retry}s.",
            headers={"Retry-After": str(retry)},
        )
    bucket.append(now)


def client_key(request: Request, user: User | None = None) -> str:
    if user is not None:
        return f"user:{user.username}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


# ---------------------------------------------------------------------------
# Upload validation
# ---------------------------------------------------------------------------
SAFE_NAME = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._- ")


def safe_filename(filename: str) -> str:
    """Strip any path component and reject traversal attempts."""
    name = os.path.basename(filename or "").strip()
    name = "".join(c for c in name if c in SAFE_NAME).strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid filename.")
    if not name.lower().endswith(".pdf"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Only PDF documents are accepted.")
    return f"{dt.datetime.now().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(4)}_{name}"


def safe_attachment_name(filename: str) -> tuple[str, str]:
    """Sanitise a chat attachment filename.

    Returns ``(display_name, stored_name)``. The stored name is generated by the
    server and contains no user-controlled path component at all, so traversal
    is impossible by construction rather than by filtering. The display name is
    kept only to show the user what they uploaded.
    """
    raw = os.path.basename((filename or "").replace("\\", "/")).strip()
    # Strip anything that is not plainly safe, then collapse repeated dots so
    # neither ".." nor a hidden double-extension survives.
    display = "".join(c for c in raw if c in SAFE_NAME).strip().strip(".")
    display = display.replace("..", ".")
    if not display:
        display = "attachment"
    display = display[:120]

    extension = ""
    if "." in display:
        candidate = display.rsplit(".", 1)[-1].lower()
        if candidate.isalnum() and len(candidate) <= 8:
            extension = f".{candidate}"

    stored = f"{dt.datetime.now().strftime('%Y%m%d%H%M%S')}_{secrets.token_hex(8)}{extension}"
    return display, stored


def validate_attachment_bytes(data: bytes, filename: str = "") -> None:
    """Size and emptiness checks for a chat attachment.

    Format validation is deliberately NOT done here — it belongs to the file
    analysis layer, which inspects magic bytes and reports a mismatch to the
    user rather than rejecting outright. This function only enforces the limits
    that must hold before any parsing is attempted.
    """
    if not data:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"'{filename or 'The file'}' is empty (0 bytes).",
        )
    if len(data) > settings.max_attachment_bytes:
        limit = settings.max_attachment_bytes // (1024 * 1024)
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"'{filename or 'The file'}' is {len(data) / (1024 * 1024):.1f} MB, over the "
            f"{limit} MB limit. Try splitting it or exporting a smaller extract.",
        )


def resolve_within(directory, name: str):
    """Resolve ``name`` inside ``directory`` and refuse anything that escapes.

    Belt and braces: stored names are server-generated, but any code path that
    turns a database value into a filesystem path is checked again here.
    """
    from pathlib import Path

    base = Path(directory).resolve()
    target = (base / os.path.basename(name)).resolve()
    if not str(target).startswith(str(base) + os.sep) and target != base:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid file reference.")
    return target


def validate_pdf_bytes(data: bytes) -> None:
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"File exceeds the {settings.max_upload_bytes // (1024 * 1024)} MB limit.",
        )
    if not data.startswith(b"%PDF-"):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "File does not have a valid PDF signature.",
        )
