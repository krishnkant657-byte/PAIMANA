"""Canonical PAIMANA data model.

Design rule that drives this whole file: a project is NOT a mutable row. A
project is a stable identity plus an append-only series of snapshots, each of
which is traceable back to the page of the document it came from.
"""
from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

UNKNOWN = "UNKNOWN"
REVIEW = "DATA QUALITY REVIEW"


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class RiskLevel(str, enum.Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    SEVERE = "SEVERE"
    UNKNOWN = "UNKNOWN"


class TrendDirection(str, enum.Enum):
    IMPROVING = "IMPROVING"
    STABLE = "STABLE"
    DETERIORATING = "DETERIORATING"
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"


class Severity(str, enum.Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertStatus(str, enum.Enum):
    OPEN = "OPEN"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    IN_REVIEW = "IN_REVIEW"
    CLOSED = "CLOSED"


class InterventionStatus(str, enum.Enum):
    CREATED = "CREATED"
    ASSIGNED = "ASSIGNED"
    IN_PROGRESS = "IN_PROGRESS"
    ESCALATED = "ESCALATED"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class DataOrigin(str, enum.Enum):
    """Where a value came from. Rendered in the UI. Never silently mixed."""

    EXTRACTED = "EXTRACTED"          # parsed directly from a source document
    DERIVED = "DERIVED"              # computed from extracted values
    MAPPED = "MAPPED"                # resolved via a reviewed lookup table
    DEMO = "DEMO"                    # simulated, must be labelled in the UI
    UNKNOWN = "UNKNOWN"              # not available — shown as UNKNOWN


class UserRole(str, enum.Enum):
    VIEWER = "VIEWER"
    ANALYST = "ANALYST"
    ADMIN = "ADMIN"


# ---------------------------------------------------------------------------
# Provenance layer
# ---------------------------------------------------------------------------
class DataSource(Base):
    """One ingested source document (e.g. a monthly Flash Report PDF)."""

    __tablename__ = "data_sources"

    id = Column(Integer, primary_key=True)
    filename = Column(String(255), nullable=False)
    sha256 = Column(String(64), nullable=False, unique=True, index=True)
    document_type = Column(String(64), default="FLASH_REPORT")
    report_period = Column(String(7), index=True)       # YYYY-MM
    report_label = Column(String(64))                    # "JULY 2026"
    issue_number = Column(String(32))
    page_count = Column(Integer)
    publisher = Column(String(255), default=UNKNOWN)
    ingested_at = Column(DateTime, default=utcnow)
    ingested_by = Column(String(120), default="system")
    is_demo = Column(Boolean, default=False, nullable=False)
    notes = Column(Text)

    records = relationship("ExtractedRecord", back_populates="source")
    snapshots = relationship("ProjectSnapshot", back_populates="source")


class ExtractedRecord(Base):
    """A single raw row as it appeared in the source, before normalisation.

    Keeping this separate from the snapshot is what makes provenance real: the
    UI can show the original string alongside the normalised number.
    """

    __tablename__ = "extracted_records"

    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False, index=True)
    page_number = Column(Integer, nullable=False)
    table_name = Column(String(120))
    row_index = Column(Integer)
    raw_payload = Column(JSON, nullable=False)      # original cell strings
    normalised_payload = Column(JSON)               # parsed values
    extraction_confidence = Column(Float, default=0.0)
    field_confidence = Column(JSON)                 # per-field confidence
    parse_warnings = Column(JSON)
    extracted_at = Column(DateTime, default=utcnow)

    source = relationship("DataSource", back_populates="records")
    snapshot = relationship("ProjectSnapshot", back_populates="record", uselist=False)


# ---------------------------------------------------------------------------
# Project identity + history
# ---------------------------------------------------------------------------
class Project(Base):
    """Stable project identity. Slow-changing attributes only.

    Anything that varies month to month belongs on ProjectSnapshot, not here.
    """

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True)
    project_code = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(Text, nullable=False)
    agency = Column(String(512), default=UNKNOWN)
    ministry = Column(String(255), default=UNKNOWN)
    sector = Column(String(255), default=UNKNOWN)
    state = Column(String(512), default=UNKNOWN)
    is_multi_state = Column(Boolean, default=False)
    states_list = Column(JSON)                       # parsed from Multi-States (...)
    legacy_ocms_code = Column(String(64))
    pmgid = Column(String(64))

    # Provenance of the slow-changing attributes themselves.
    sector_origin = Column(Enum(DataOrigin), default=DataOrigin.UNKNOWN)
    state_origin = Column(Enum(DataOrigin), default=DataOrigin.UNKNOWN)
    is_demo = Column(Boolean, default=False, nullable=False, index=True)

    first_seen_period = Column(String(7))
    last_seen_period = Column(String(7), index=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    snapshots = relationship(
        "ProjectSnapshot", back_populates="project", order_by="ProjectSnapshot.report_period"
    )
    milestones = relationship("ProjectMilestone", back_populates="project")
    alerts = relationship("Alert", back_populates="project")
    interventions = relationship("Intervention", back_populates="project")
    quality_issues = relationship("DataQualityIssue", back_populates="project")

    __table_args__ = (Index("ix_projects_state_sector", "state", "sector"),)


class ProjectSnapshot(Base):
    """Immutable point-in-time record of a project, one per reporting period.

    Never updated in place. A correction creates a superseding snapshot with
    `supersedes_id` set, so the original observation survives.
    """

    __tablename__ = "project_snapshots"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    source_id = Column(Integer, ForeignKey("data_sources.id"), nullable=False)
    record_id = Column(Integer, ForeignKey("extracted_records.id"))
    report_period = Column(String(7), nullable=False, index=True)   # YYYY-MM

    # Cost (₹ crore)
    original_cost = Column(Float)
    revised_cost = Column(Float)
    expenditure = Column(Float)

    # Progress (%)
    physical_progress = Column(Float)
    financial_progress = Column(Float)      # derived

    # Schedule
    approval_date = Column(Date)
    start_date = Column(Date)
    original_completion = Column(Date)
    revised_completion = Column(Date)

    # Derived indicators
    cost_escalation_pct = Column(Float)
    progress_divergence = Column(Float)     # financial − physical
    schedule_delay_months = Column(Integer)
    elapsed_time_pct = Column(Float)

    # Risk (written by the risk engine, versioned with the snapshot)
    risk_score = Column(Float)
    risk_level = Column(Enum(RiskLevel), default=RiskLevel.UNKNOWN)
    risk_confidence = Column(Float)
    risk_drivers = Column(JSON)             # list of {code,label,detail,weight,evidence}
    risk_engine_version = Column(String(32))

    data_quality_status = Column(String(64), default="OK")
    completeness = Column(Float)
    is_demo = Column(Boolean, default=False, nullable=False)
    supersedes_id = Column(Integer, ForeignKey("project_snapshots.id"))
    created_at = Column(DateTime, default=utcnow)

    project = relationship("Project", back_populates="snapshots")
    source = relationship("DataSource", back_populates="snapshots")
    record = relationship("ExtractedRecord", back_populates="snapshot")

    __table_args__ = (
        UniqueConstraint("project_id", "report_period", "supersedes_id", name="uq_snapshot_period"),
        Index("ix_snapshot_period_risk", "report_period", "risk_score"),
    )


class ProjectMilestone(Base):
    __tablename__ = "project_milestones"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    code = Column(String(32))
    name = Column(String(255))
    planned_date = Column(Date)
    forecast_date = Column(Date)
    actual_date = Column(Date)
    status = Column(String(64), default=UNKNOWN)
    origin = Column(Enum(DataOrigin), default=DataOrigin.DERIVED)
    notes = Column(Text)

    project = relationship("Project", back_populates="milestones")


# ---------------------------------------------------------------------------
# Risk events, alerts, interventions
# ---------------------------------------------------------------------------
class RiskEvent(Base):
    """A material change in risk between two consecutive snapshots."""

    __tablename__ = "risk_events"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    from_period = Column(String(7))
    to_period = Column(String(7))
    from_score = Column(Float)
    to_score = Column(Float)
    delta = Column(Float)
    direction = Column(Enum(TrendDirection))
    explanation = Column(JSON)
    created_at = Column(DateTime, default=utcnow)


class Alert(Base):
    """Early warning. Every alert carries the evidence that triggered it."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    code = Column(String(64), nullable=False, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text)
    severity = Column(Enum(Severity), default=Severity.MEDIUM, index=True)
    status = Column(Enum(AlertStatus), default=AlertStatus.OPEN, index=True)
    trigger = Column(Text)
    evidence = Column(JSON)
    report_period = Column(String(7), index=True)
    snapshot_id = Column(Integer, ForeignKey("project_snapshots.id"))
    created_at = Column(DateTime, default=utcnow)
    acknowledged_at = Column(DateTime)
    closed_at = Column(DateTime)

    project = relationship("Project", back_populates="alerts")
    interventions = relationship("Intervention", back_populates="alert")


class Intervention(Base):
    __tablename__ = "interventions"

    id = Column(Integer, primary_key=True)
    reference = Column(String(32), unique=True, index=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=False, index=True)
    alert_id = Column(Integer, ForeignKey("alerts.id"))
    issue = Column(Text, nullable=False)
    severity = Column(Enum(Severity), default=Severity.MEDIUM)
    assigned_authority = Column(String(255))
    owner = Column(String(255))
    due_date = Column(Date)
    action = Column(Text)
    status = Column(Enum(InterventionStatus), default=InterventionStatus.CREATED, index=True)
    remarks = Column(Text)
    resolution = Column(Text)
    created_by = Column(String(120))
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    resolved_at = Column(DateTime)
    closed_at = Column(DateTime)
    history = Column(JSON, default=list)

    project = relationship("Project", back_populates="interventions")
    alert = relationship("Alert", back_populates="interventions")


# ---------------------------------------------------------------------------
# Data quality, reports, audit, users
# ---------------------------------------------------------------------------
class DataQualityIssue(Base):
    __tablename__ = "data_quality_issues"

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey("projects.id"), index=True)
    source_id = Column(Integer, ForeignKey("data_sources.id"))
    record_id = Column(Integer, ForeignKey("extracted_records.id"))
    issue_type = Column(String(64), nullable=False, index=True)
    field = Column(String(64))
    severity = Column(Enum(Severity), default=Severity.LOW, index=True)
    description = Column(Text)
    detail = Column(JSON)
    status = Column(String(32), default="OPEN", index=True)
    report_period = Column(String(7))
    created_at = Column(DateTime, default=utcnow)
    resolved_at = Column(DateTime)

    project = relationship("Project", back_populates="quality_issues")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True)
    report_type = Column(String(64), nullable=False)
    title = Column(String(255))
    project_id = Column(Integer, ForeignKey("projects.id"))
    report_period = Column(String(7))
    filename = Column(String(255))
    parameters = Column(JSON)
    generated_by = Column(String(120))
    generated_at = Column(DateTime, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True)
    actor = Column(String(120), index=True)
    action = Column(String(120), nullable=False, index=True)
    object_type = Column(String(64))
    object_id = Column(String(64))
    previous_value = Column(JSON)
    new_value = Column(JSON)
    meta = Column(JSON)
    ip_address = Column(String(64))
    created_at = Column(DateTime, default=utcnow, index=True)


# ---------------------------------------------------------------------------
# Assistant conversations
# ---------------------------------------------------------------------------
class Conversation(Base):
    """One assistant thread.

    `context` holds the compact structured state the agent reasons over —
    current project, recently listed projects, active filters, attachment
    references. It is deliberately small: the agent resolves references from
    this object rather than replaying the transcript.
    """

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    public_id = Column(String(64), unique=True, nullable=False, index=True)
    owner = Column(String(120), index=True)          # NULL for anonymous sessions
    title = Column(String(255))
    context = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    messages = relationship(
        "ChatMessage", back_populates="conversation", order_by="ChatMessage.id",
        cascade="all, delete-orphan",
    )
    attachments = relationship(
        "ChatAttachment", back_populates="conversation", cascade="all, delete-orphan"
    )


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    role = Column(String(16), nullable=False)        # user | assistant
    content = Column(Text, nullable=False)
    source = Column(String(48))                       # provenance label for the answer
    intent = Column(String(64))
    project_codes = Column(JSON, default=list)
    attachment_ids = Column(JSON, default=list)
    analysis_meta = Column(JSON, default=dict)
    created_at = Column(DateTime, default=utcnow, index=True)

    conversation = relationship("Conversation", back_populates="messages")


class ChatAttachment(Base):
    """An uploaded file attached to an assistant conversation.

    `analysis` stores the deterministic result computed at upload time, so a
    follow-up question about the same file does not re-parse it.
    """

    __tablename__ = "chat_attachments"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False, index=True,
    )
    filename = Column(String(255), nullable=False)      # sanitised display name
    stored_name = Column(String(255), nullable=False)   # on-disk name, never user-controlled
    sha256 = Column(String(64), index=True)
    size_bytes = Column(Integer, nullable=False)
    declared_mime = Column(String(128))
    detected_kind = Column(String(32))
    detected_label = Column(String(128))
    status = Column(String(24), default="PENDING")      # ANALYSED | IDENTIFIED | FAILED
    error = Column(Text)
    analysis = Column(JSON)
    uploaded_by = Column(String(120))
    created_at = Column(DateTime, default=utcnow)
    deleted_at = Column(DateTime)

    conversation = relationship("Conversation", back_populates="attachments")


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(120), unique=True, nullable=False, index=True)
    full_name = Column(String(255))
    designation = Column(String(255))
    password_hash = Column(String(255), nullable=False)
    role = Column(Enum(UserRole), default=UserRole.VIEWER, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=utcnow)
    last_login_at = Column(DateTime)
