"""Pydantic schemas. These define the public API contract."""
from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field, field_validator


# --- auth -------------------------------------------------------------------
class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=256)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    full_name: str | None = None
    designation: str | None = None
    role: str
    expires_in_minutes: int


# --- projects ---------------------------------------------------------------
class ProjectSummary(BaseModel):
    id: int
    project_code: str
    name: str
    agency: str
    ministry: str
    sector: str
    state: str
    is_demo: bool
    risk_score: float | None = None
    risk_level: str = "UNKNOWN"
    risk_confidence: float | None = None
    trend: str = "INSUFFICIENT_HISTORY"
    physical_progress: float | None = None
    financial_progress: float | None = None
    original_cost_cr: float | None = None
    revised_cost_cr: float | None = None
    expenditure_cr: float | None = None
    cost_escalation_pct: float | None = None
    schedule_delay_months: int | None = None
    last_period: str | None = None
    open_alerts: int = 0


class PaginatedProjects(BaseModel):
    total: int
    page: int
    page_size: int
    pages: int
    period: str | None
    items: list[ProjectSummary]


# --- interventions ----------------------------------------------------------
class InterventionCreate(BaseModel):
    project_code: str
    alert_id: int | None = None
    issue: str = Field(min_length=5, max_length=2000)
    severity: str = "MEDIUM"
    assigned_authority: str | None = Field(default=None, max_length=255)
    owner: str | None = Field(default=None, max_length=255)
    due_date: dt.date | None = None
    action: str | None = Field(default=None, max_length=4000)
    remarks: str | None = Field(default=None, max_length=4000)

    @field_validator("severity")
    @classmethod
    def _sev(cls, v: str) -> str:
        allowed = {"INFO", "LOW", "MEDIUM", "HIGH", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"severity must be one of {sorted(allowed)}")
        return v.upper()


class InterventionUpdate(BaseModel):
    status: str | None = None
    owner: str | None = Field(default=None, max_length=255)
    assigned_authority: str | None = Field(default=None, max_length=255)
    due_date: dt.date | None = None
    action: str | None = Field(default=None, max_length=4000)
    remarks: str | None = Field(default=None, max_length=4000)
    resolution: str | None = Field(default=None, max_length=4000)

    @field_validator("status")
    @classmethod
    def _status(cls, v: str | None) -> str | None:
        if v is None:
            return v
        allowed = {"CREATED", "ASSIGNED", "IN_PROGRESS", "ESCALATED", "RESOLVED", "CLOSED"}
        if v.upper() not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return v.upper()


class AlertUpdate(BaseModel):
    status: str

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        allowed = {"OPEN", "ACKNOWLEDGED", "IN_REVIEW", "CLOSED"}
        if v.upper() not in allowed:
            raise ValueError(f"status must be one of {sorted(allowed)}")
        return v.upper()


# --- assistant / scenario ---------------------------------------------------
class AssistantQuery(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    project_code: str | None = None


class ChatMessageRequest(BaseModel):
    """A turn in the conversational assistant.

    `attachment_ids` must already have been uploaded to the same conversation via
    POST /api/chat/attachments; the router verifies ownership before use.
    """

    message: str = Field(min_length=1, max_length=4000)
    conversation_id: str | None = Field(default=None, max_length=64)
    attachment_ids: list[int] = Field(default_factory=list, max_length=10)
    project_code: str | None = Field(default=None, max_length=64)

    @field_validator("message")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("message must not be blank")
        return v.strip()


class ScenarioRequest(BaseModel):
    project_code: str
    cost_change_pct: float = Field(default=0.0, ge=-90, le=500)
    schedule_change_days: int = Field(default=0, ge=-3650, le=3650)
    progress_change_pp: float = Field(default=0.0, ge=-100, le=100)
    expenditure_change_pct: float = Field(default=0.0, ge=-90, le=500)


class SimulatorRequest(BaseModel):
    estimated_cost_cr: float = Field(gt=0, le=1_000_000)
    duration_months: int = Field(gt=0, le=360)
    sector: str | None = None
    state: str | None = None
    expected_progress_year1: float | None = Field(default=None, ge=0, le=100)


# --- reports ----------------------------------------------------------------
class ReportRequest(BaseModel):
    report_type: str
    project_code: str | None = None
    period: str | None = None

    @field_validator("report_type")
    @classmethod
    def _type(cls, v: str) -> str:
        allowed = {"PROJECT", "MONTHLY_MONITORING", "EXECUTIVE_BRIEF", "RISK"}
        if v.upper() not in allowed:
            raise ValueError(f"report_type must be one of {sorted(allowed)}")
        return v.upper()
