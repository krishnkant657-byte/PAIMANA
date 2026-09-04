"""Environment-based configuration. No secret is ever hardcoded here."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")


class Settings:
    # --- paths -------------------------------------------------------------
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    frontend_dir: Path = BASE_DIR / "frontend"
    upload_dir: Path = BASE_DIR / "data" / "uploads"

    # --- database ----------------------------------------------------------
    # SQLite for development / demo, PostgreSQL for production.
    database_url: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'paimana.db'}")

    # --- auth --------------------------------------------------------------
    secret_key: str = os.getenv("PAIMANA_SECRET_KEY", "")
    token_ttl_minutes: int = int(os.getenv("PAIMANA_TOKEN_TTL_MINUTES", "480"))

    # --- CORS --------------------------------------------------------------
    # Explicit origins only. Never "*" — a wildcard with credentials is both a
    # vulnerability and rejected by browsers per the CORS specification.
    cors_origins: list[str] = [
        o.strip()
        for o in os.getenv(
            "PAIMANA_CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
        ).split(",")
        if o.strip()
    ]

    # --- LLM ---------------------------------------------------------------
    llm_provider: str = os.getenv("PAIMANA_LLM_PROVIDER", "gemini")
    llm_api_key: str = os.getenv("GEMINI_API_KEY", "")
    llm_models: list[str] = [
        m.strip()
        for m in os.getenv(
            "PAIMANA_LLM_MODELS", "gemini-1.5-flash,gemini-2.0-flash,gemini-2.5-flash,gemini-flash-latest"
        ).split(",")
        if m.strip()
    ]
    llm_timeout_seconds: int = int(os.getenv("PAIMANA_LLM_TIMEOUT", "30"))

    # --- uploads -----------------------------------------------------------
    # Flash Report ingestion remains PDF-only and analyst-gated. This is a
    # separate, stricter path from chat attachments below.
    max_upload_bytes: int = int(os.getenv("PAIMANA_MAX_UPLOAD_MB", "40")) * 1024 * 1024
    allowed_upload_mime: tuple[str, ...] = ("application/pdf",)

    # --- chat attachments --------------------------------------------------
    chat_upload_dir: Path = BASE_DIR / "data" / "chat_uploads"
    max_attachment_bytes: int = int(os.getenv("PAIMANA_MAX_ATTACHMENT_MB", "25")) * 1024 * 1024
    max_attachments_per_message: int = int(os.getenv("PAIMANA_MAX_ATTACHMENTS", "5"))
    #: Attachments older than this are removed by the cleanup pass.
    attachment_retention_hours: int = int(os.getenv("PAIMANA_ATTACHMENT_RETENTION_HOURS", "24"))
    chat_rate_limit: int = int(os.getenv("PAIMANA_CHAT_RATE_LIMIT", "40"))
    chat_rate_window: int = int(os.getenv("PAIMANA_CHAT_RATE_WINDOW", "300"))
    upload_rate_limit: int = int(os.getenv("PAIMANA_UPLOAD_RATE_LIMIT", "30"))
    upload_rate_window: int = int(os.getenv("PAIMANA_UPLOAD_RATE_WINDOW", "600"))

    # --- rate limiting -----------------------------------------------------
    assistant_rate_limit: int = int(os.getenv("PAIMANA_ASSISTANT_RATE_LIMIT", "20"))
    assistant_rate_window: int = int(os.getenv("PAIMANA_ASSISTANT_RATE_WINDOW", "300"))

    # --- product -----------------------------------------------------------
    environment: str = os.getenv("PAIMANA_ENV", "development")

    @property
    def llm_enabled(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    s.chat_upload_dir.mkdir(parents=True, exist_ok=True)
    if s.is_production and not s.secret_key:
        raise RuntimeError(
            "PAIMANA_SECRET_KEY must be set in production. Refusing to start with an "
            "ephemeral signing key."
        )
    if not s.secret_key:
        # Development only: ephemeral key, tokens do not survive a restart.
        import secrets

        s.secret_key = secrets.token_urlsafe(48)
    return s


settings = get_settings()
