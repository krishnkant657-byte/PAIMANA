"""Flash Report ingestion.

Pipeline: UPLOAD -> VALIDATE -> EXTRACT -> NORMALISE -> VALIDATE -> MATCH
          -> CREATE SNAPSHOT -> RECALCULATE RISK -> UPDATE PLATFORM

Two rules govern everything here:

1. Every value keeps a pointer back to the document and page it came from.
2. Nothing is invented. A field that cannot be parsed becomes None and raises a
   DataQualityIssue; it never becomes a plausible default.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import pdfplumber
from sqlalchemy.orm import Session

from ..models import (
    DataOrigin,
    DataQualityIssue,
    DataSource,
    ExtractedRecord,
    Project,
    ProjectSnapshot,
    Severity,
)
from .agency_map import resolve_agency

log = logging.getLogger(__name__)

UNKNOWN = "UNKNOWN"

MONTHS = {
    "JANUARY": 1, "FEBRUARY": 2, "MARCH": 3, "APRIL": 4, "MAY": 5, "JUNE": 6,
    "JULY": 7, "AUGUST": 8, "SEPTEMBER": 9, "OCTOBER": 10, "NOVEMBER": 11, "DECEMBER": 12,
}

PROJECT_TABLE_HEADER = "Sl.No"
TARGET_TABLE = "All Ongoing Projects"
NE_TABLE = "Ongoing Projects of North-East Region"

_MMYYYY = re.compile(r"^(0[1-9]|1[0-2])/(\d{4})$")
_PAREN = re.compile(r"^\((.*)\)$")
_CODE_LINE = re.compile(r"^\((\d{3,10})\)$")
_NUM = re.compile(r"^-?[\d,]*\.?\d+$")


# ---------------------------------------------------------------------------
# Value normalisers — each returns (value, confidence, warning)
# ---------------------------------------------------------------------------
def norm_number(raw: str | None) -> tuple[float | None, float, str | None]:
    """Parse a ₹-crore or percentage figure. Returns None rather than 0.

    Returning 0 for a missing cost is how the old pipeline produced nonsense
    ratios, so a blank stays blank all the way to the UI, where it renders as
    UNKNOWN.
    """
    if raw is None:
        return None, 0.0, "missing"
    s = str(raw).strip().replace("\n", " ")
    if s in {"", "-", "(-)", "NA", "N/A", "--"}:
        return None, 1.0, None                      # explicitly absent in source
    s = s.replace(",", "").replace("₹", "").strip()
    if not _NUM.match(s):
        cleaned = re.sub(r"[^\d.]", "", s)
        if not cleaned or cleaned.count(".") > 1:
            return None, 0.0, f"unparseable_number:{raw!r}"
        try:
            return float(cleaned), 0.55, f"lossy_number_parse:{raw!r}"
        except ValueError:
            return None, 0.0, f"unparseable_number:{raw!r}"
    try:
        return float(s), 1.0, None
    except ValueError:
        return None, 0.0, f"unparseable_number:{raw!r}"


def norm_percent(raw: str | None) -> tuple[float | None, float, str | None]:
    value, conf, warn = norm_number(raw)
    if value is None:
        return None, conf, warn
    if value < 0 or value > 100:
        return value, 0.4, f"percent_out_of_range:{value}"
    return value, conf, warn


def norm_month_year(raw: str | None) -> tuple[dt.date | None, float, str | None]:
    """MM/YYYY -> first day of that month. '(-)' means genuinely not set."""
    if raw is None:
        return None, 0.0, "missing"
    s = str(raw).strip()
    if s in {"", "-", "(-)", "NA"}:
        return None, 1.0, None
    m = _MMYYYY.match(s)
    if not m:
        return None, 0.0, f"unparseable_date:{raw!r}"
    month, year = int(m.group(1)), int(m.group(2))
    if not (1990 <= year <= 2100):
        return None, 0.3, f"date_out_of_range:{raw!r}"
    return dt.date(year, month, 1), 1.0, None


def split_paren_pair(raw: str | None) -> tuple[str | None, str | None]:
    """'03/2029\\n(-)' -> ('03/2029', '-'); primary value on top, revision below."""
    if raw is None:
        return None, None
    lines = [ln.strip() for ln in str(raw).split("\n") if ln.strip()]
    if not lines:
        return None, None
    primary = lines[0]
    secondary = None
    if len(lines) > 1:
        m = _PAREN.match(lines[1])
        secondary = m.group(1).strip() if m else lines[1]
    return primary, secondary


def parse_state(raw: str | None) -> tuple[str, list[str], bool]:
    """Handle both 'West Bengal' and 'Multi-States (A, B, C)'."""
    if not raw or not str(raw).strip():
        return UNKNOWN, [], False
    s = " ".join(str(raw).split())
    if s.upper().startswith("MULTI-STATES"):
        inner = re.search(r"\((.*)\)", s)
        parts = [p.strip() for p in inner.group(1).split(",")] if inner else []
        return s, [p for p in parts if p], True
    return s, [s], False


def parse_project_cell(raw: str | None) -> dict:
    """Split the composite first column.

    Layout in source:
        <project name, possibly wrapped over several lines>
        (Agency)
        (Project Code)
        (Legacy OCMS Code) (PMGID)
    """
    out = {
        "name": None, "agency": None, "project_code": None,
        "legacy_ocms_code": None, "pmgid": None, "confidence": 0.0, "warnings": [],
    }
    if not raw:
        out["warnings"].append("empty_project_cell")
        return out

    lines = [ln.strip() for ln in str(raw).split("\n") if ln.strip()]
    if not lines:
        out["warnings"].append("empty_project_cell")
        return out

    code_idx = next(
        (i for i in range(len(lines) - 1, -1, -1) if _CODE_LINE.match(lines[i])), None
    )

    if code_idx is not None and code_idx >= 1:
        out["project_code"] = _CODE_LINE.match(lines[code_idx]).group(1)
        agency_line = lines[code_idx - 1]
        m = _PAREN.match(agency_line)
        out["agency"] = (m.group(1) if m else agency_line).strip()
        out["name"] = " ".join(lines[: code_idx - 1]).strip() or None
        tail = lines[code_idx + 1:]
        if tail:
            ids = re.findall(r"\(([^)]*)\)", " ".join(tail))
            if len(ids) >= 1 and ids[0].strip() not in {"-", ""}:
                out["legacy_ocms_code"] = ids[0].strip()
            if len(ids) >= 2 and ids[1].strip() not in {"-", ""}:
                out["pmgid"] = ids[1].strip()
        out["confidence"] = 1.0 if out["name"] else 0.5
        if not out["name"]:
            out["warnings"].append("missing_project_name")
    else:
        # Degraded layout — keep what we can and flag it for review.
        out["name"] = lines[0]
        out["confidence"] = 0.25
        out["warnings"].append("project_code_not_found")
        for ln in lines[1:]:
            m = _PAREN.match(ln)
            if m and not out["agency"]:
                out["agency"] = m.group(1).strip()
    return out


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
@dataclass
class ParsedRow:
    page_number: int
    table_name: str
    row_index: int
    raw: list
    parsed: dict = field(default_factory=dict)
    field_confidence: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    @property
    def confidence(self) -> float:
        if not self.field_confidence:
            return 0.0
        return round(sum(self.field_confidence.values()) / len(self.field_confidence), 4)


def detect_report_period(pdf) -> tuple[str | None, str | None, str | None]:
    """Read the reporting month off the cover page. Returns (YYYY-MM, label, issue)."""
    text = (pdf.pages[0].extract_text() or "") if pdf.pages else ""
    label, period, issue = None, None, None
    m = re.search(r"\b(" + "|".join(MONTHS) + r")\s+(\d{4})\b", text.upper())
    if m:
        label = f"{m.group(1)} {m.group(2)}"
        period = f"{int(m.group(2)):04d}-{MONTHS[m.group(1)]:02d}"
    issue_m = re.search(r"^\s*(\d{3,4})\s*$", text.split("\n")[0] if text else "")
    if issue_m:
        issue = issue_m.group(1)
    return period, label, issue


def _current_table_name(page_text: str) -> str | None:
    head = (page_text or "").split("\n")[0].strip()
    if head.startswith(TARGET_TABLE):
        return TARGET_TABLE
    if head.startswith(NE_TABLE):
        return NE_TABLE
    return None


def extract_rows(pdf_path: Path, progress_every: int = 25) -> tuple[list[ParsedRow], dict]:
    """Walk the PDF and return every parseable project row, with provenance."""
    rows: list[ParsedRow] = []
    meta: dict = {}

    with pdfplumber.open(pdf_path) as pdf:
        period, label, issue = detect_report_period(pdf)
        meta = {
            "report_period": period,
            "report_label": label,
            "issue_number": issue,
            "page_count": len(pdf.pages),
        }

        for page_index, page in enumerate(pdf.pages):
            page_number = page_index + 1
            if progress_every and page_number % progress_every == 0:
                log.info("  ...page %s/%s", page_number, len(pdf.pages))

            text = page.extract_text() or ""
            table_name = _current_table_name(text)
            if table_name is None:
                continue

            for table in page.extract_tables() or []:
                if not table or len(table[0]) != 8:
                    continue
                header = " ".join(c or "" for c in table[0])
                if PROJECT_TABLE_HEADER not in header:
                    continue

                for row_index, row in enumerate(table[1:], start=1):
                    if not row or not row[0] or not str(row[0]).strip().isdigit():
                        continue
                    rows.append(_parse_row(page_number, table_name, row_index, row))

    return rows, meta


def _parse_row(page_number: int, table_name: str, row_index: int, row: list) -> ParsedRow:
    pr = ParsedRow(page_number=page_number, table_name=table_name, row_index=row_index, raw=list(row))

    proj = parse_project_cell(row[1])
    pr.parsed.update(
        {
            "name": proj["name"],
            "agency": proj["agency"],
            "project_code": proj["project_code"],
            "legacy_ocms_code": proj["legacy_ocms_code"],
            "pmgid": proj["pmgid"],
        }
    )
    pr.field_confidence["project_identity"] = proj["confidence"]
    pr.warnings.extend(proj["warnings"])

    state, states_list, multi = parse_state(row[2])
    pr.parsed.update({"state": state, "states_list": states_list, "is_multi_state": multi})
    pr.field_confidence["state"] = 1.0 if state != UNKNOWN else 0.0
    if state == UNKNOWN:
        pr.warnings.append("missing_state")

    approval_raw, start_raw = split_paren_pair(row[3])
    approval, c1, w1 = norm_month_year(approval_raw)
    start, c2, w2 = norm_month_year(start_raw)
    pr.parsed.update({"approval_date": approval, "start_date": start})
    pr.field_confidence["approval_date"] = c1
    pr.field_confidence["start_date"] = c2
    pr.warnings.extend(w for w in (w1, w2) if w)

    orig_doc_raw, rev_doc_raw = split_paren_pair(row[4])
    orig_doc, c3, w3 = norm_month_year(orig_doc_raw)
    rev_doc, c4, w4 = norm_month_year(rev_doc_raw)
    pr.parsed.update({"original_completion": orig_doc, "revised_completion": rev_doc})
    pr.field_confidence["original_completion"] = c3
    pr.field_confidence["revised_completion"] = c4
    pr.warnings.extend(w for w in (w3, w4) if w)

    orig_cost_raw, rev_cost_raw = split_paren_pair(row[5])
    orig_cost, c5, w5 = norm_number(orig_cost_raw)
    rev_cost, c6, w6 = norm_number(rev_cost_raw)
    pr.parsed.update({"original_cost": orig_cost, "revised_cost": rev_cost})
    pr.field_confidence["original_cost"] = c5
    pr.field_confidence["revised_cost"] = c6
    pr.warnings.extend(w for w in (w5, w6) if w)

    expenditure, c7, w7 = norm_number(row[6])
    pr.parsed["expenditure"] = expenditure
    pr.field_confidence["expenditure"] = c7
    if w7:
        pr.warnings.append(w7)

    progress, c8, w8 = norm_percent(row[7])
    pr.parsed["physical_progress"] = progress
    pr.field_confidence["physical_progress"] = c8
    if w8:
        pr.warnings.append(w8)

    return pr


# ---------------------------------------------------------------------------
# Derived indicators
# ---------------------------------------------------------------------------
def compute_indicators(p: dict, as_of: dt.date | None = None) -> dict:
    """Deterministic Layer-1 indicators. None in, None out — never a fabricated 0."""
    out: dict = {
        "financial_progress": None,
        "cost_escalation_pct": None,
        "progress_divergence": None,
        "schedule_delay_months": None,
        "elapsed_time_pct": None,
    }

    orig, rev = p.get("original_cost"), p.get("revised_cost")
    exp, phys = p.get("expenditure"), p.get("physical_progress")

    effective_cost = rev if rev not in (None, 0) else orig
    if exp is not None and effective_cost not in (None, 0):
        out["financial_progress"] = round(100.0 * exp / effective_cost, 2)

    if orig not in (None, 0) and rev is not None:
        out["cost_escalation_pct"] = round(100.0 * (rev - orig) / orig, 2)

    if out["financial_progress"] is not None and phys is not None:
        out["progress_divergence"] = round(out["financial_progress"] - phys, 2)

    orig_doc, rev_doc = p.get("original_completion"), p.get("revised_completion")
    if orig_doc and rev_doc:
        out["schedule_delay_months"] = (rev_doc.year - orig_doc.year) * 12 + (
            rev_doc.month - orig_doc.month
        )

    start = p.get("start_date") or p.get("approval_date")
    target = rev_doc or orig_doc
    ref = as_of or dt.date.today()
    if start and target and target > start:
        total = (target.year - start.year) * 12 + (target.month - start.month)
        elapsed = (ref.year - start.year) * 12 + (ref.month - start.month)
        if total > 0:
            out["elapsed_time_pct"] = round(max(0.0, min(100.0 * elapsed / total, 999.0)), 2)

    return out


def completeness(p: dict) -> float:
    keys = [
        "name", "agency", "project_code", "state", "original_cost", "revised_cost",
        "expenditure", "physical_progress", "approval_date", "original_completion",
    ]
    present = sum(1 for k in keys if p.get(k) not in (None, "", UNKNOWN, []))
    return round(100.0 * present / len(keys), 2)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _period_to_date(period: str) -> dt.date:
    y, m = period.split("-")
    return dt.date(int(y), int(m), 1)


def ingest_pdf(
    db: Session,
    pdf_path: Path,
    *,
    actor: str = "system",
    is_demo: bool = False,
    allow_reingest: bool = False,
) -> dict:
    """Ingest one Flash Report. Idempotent by file hash."""
    pdf_path = Path(pdf_path)
    digest = sha256_file(pdf_path)

    existing = db.query(DataSource).filter_by(sha256=digest).one_or_none()
    if existing and not allow_reingest:
        return {
            "status": "DUPLICATE",
            "message": f"Already ingested as source #{existing.id} ({existing.report_label}).",
            "source_id": existing.id,
            "projects_created": 0,
            "snapshots_created": 0,
        }

    log.info("Extracting %s", pdf_path.name)
    rows, meta = extract_rows(pdf_path)

    period = meta.get("report_period")
    if not period:
        return {
            "status": "REJECTED",
            "message": "Could not determine the reporting period from the cover page. "
                       "Refusing to ingest rather than guessing a date.",
            "projects_created": 0,
            "snapshots_created": 0,
        }

    source = DataSource(
        filename=pdf_path.name,
        sha256=digest,
        document_type="FLASH_REPORT",
        report_period=period,
        report_label=meta.get("report_label"),
        issue_number=meta.get("issue_number"),
        page_count=meta.get("page_count"),
        publisher="Ministry of Statistics and Programme Implementation (published report)",
        ingested_by=actor,
        is_demo=is_demo,
    )
    db.add(source)
    db.flush()

    as_of = _period_to_date(period)
    stats = {
        "projects_created": 0, "projects_updated": 0, "snapshots_created": 0,
        "rows_parsed": len(rows), "rows_rejected": 0, "quality_issues": 0,
        "duplicate_rows": 0, "unmapped_agencies": 0,
    }
    seen_codes: dict[str, int] = {}

    for pr in rows:
        p = pr.parsed
        code = p.get("project_code")

        if not code or not p.get("name"):
            stats["rows_rejected"] += 1
            db.add(
                DataQualityIssue(
                    source_id=source.id,
                    issue_type="UNIDENTIFIABLE_ROW",
                    severity=Severity.MEDIUM,
                    description="Row could not be matched to a project identity "
                                "(missing project code or name). Not ingested.",
                    detail={"page": pr.page_number, "raw": pr.raw, "warnings": pr.warnings},
                    report_period=period,
                )
            )
            stats["quality_issues"] += 1
            continue

        # --- record raw + normalised payload (provenance) ------------------
        record = ExtractedRecord(
            source_id=source.id,
            page_number=pr.page_number,
            table_name=pr.table_name,
            row_index=pr.row_index,
            raw_payload={f"col_{i}": v for i, v in enumerate(pr.raw)},
            normalised_payload={
                k: (v.isoformat() if isinstance(v, dt.date) else v) for k, v in p.items()
            },
            extraction_confidence=pr.confidence,
            field_confidence=pr.field_confidence,
            parse_warnings=pr.warnings,
        )
        db.add(record)
        db.flush()

        # --- duplicate row within the same report --------------------------
        if code in seen_codes:
            stats["duplicate_rows"] += 1
            db.add(
                DataQualityIssue(
                    source_id=source.id,
                    record_id=record.id,
                    issue_type="DUPLICATE_ROW_IN_REPORT",
                    severity=Severity.LOW,
                    description=f"Project code {code} appears more than once in "
                                f"{source.report_label}. Kept the first occurrence "
                                f"(it also appears in the North-East regional table).",
                    detail={"page": pr.page_number, "first_page": seen_codes[code]},
                    report_period=period,
                )
            )
            stats["quality_issues"] += 1
            continue
        seen_codes[code] = pr.page_number

        # --- project identity ---------------------------------------------
        ministry, sector, matched = resolve_agency(p.get("agency"))
        if not matched:
            stats["unmapped_agencies"] += 1

        project = db.query(Project).filter_by(project_code=code).one_or_none()
        if project is None:
            project = Project(
                project_code=code,
                name=p["name"],
                agency=p.get("agency") or UNKNOWN,
                ministry=ministry,
                sector=sector,
                state=p.get("state") or UNKNOWN,
                is_multi_state=p.get("is_multi_state", False),
                states_list=p.get("states_list") or [],
                legacy_ocms_code=p.get("legacy_ocms_code"),
                pmgid=p.get("pmgid"),
                sector_origin=DataOrigin.MAPPED if matched else DataOrigin.UNKNOWN,
                state_origin=(
                    DataOrigin.EXTRACTED if p.get("state") != UNKNOWN else DataOrigin.UNKNOWN
                ),
                is_demo=is_demo,
                first_seen_period=period,
                last_seen_period=period,
            )
            db.add(project)
            db.flush()
            stats["projects_created"] += 1
        else:
            # Conflicting attribute across reports -> flag, do not silently overwrite.
            if project.state != (p.get("state") or UNKNOWN) and p.get("state") != UNKNOWN:
                db.add(
                    DataQualityIssue(
                        project_id=project.id,
                        source_id=source.id,
                        record_id=record.id,
                        issue_type="CONFLICTING_VALUE",
                        field="state",
                        severity=Severity.MEDIUM,
                        description="State differs from the value recorded in an earlier "
                                    "report. Existing value retained pending review.",
                        detail={"existing": project.state, "incoming": p.get("state")},
                        report_period=period,
                    )
                )
                stats["quality_issues"] += 1
            if project.last_seen_period is None or period > project.last_seen_period:
                project.last_seen_period = period
            stats["projects_updated"] += 1

        # --- snapshot -------------------------------------------------------
        already = (
            db.query(ProjectSnapshot)
            .filter_by(project_id=project.id, report_period=period, supersedes_id=None)
            .one_or_none()
        )
        if already is not None:
            continue

        ind = compute_indicators(p, as_of=as_of)
        comp = completeness(p)

        snapshot = ProjectSnapshot(
            project_id=project.id,
            source_id=source.id,
            record_id=record.id,
            report_period=period,
            original_cost=p.get("original_cost"),
            revised_cost=p.get("revised_cost"),
            expenditure=p.get("expenditure"),
            physical_progress=p.get("physical_progress"),
            approval_date=p.get("approval_date"),
            start_date=p.get("start_date"),
            original_completion=p.get("original_completion"),
            revised_completion=p.get("revised_completion"),
            data_quality_status="OK" if comp >= 70 and pr.confidence >= 0.7 else "REVIEW",
            completeness=comp,
            is_demo=is_demo,
            **ind,
        )
        db.add(snapshot)
        stats["snapshots_created"] += 1

        # --- field-level quality issues -------------------------------------
        if pr.confidence < 0.7:
            db.add(
                DataQualityIssue(
                    project_id=project.id, source_id=source.id, record_id=record.id,
                    issue_type="LOW_EXTRACTION_CONFIDENCE", severity=Severity.MEDIUM,
                    description=f"Row parsed with confidence {pr.confidence:.0%}.",
                    detail={"warnings": pr.warnings, "page": pr.page_number},
                    report_period=period,
                )
            )
            stats["quality_issues"] += 1

        for fname in ("original_cost", "physical_progress", "state"):
            if p.get(fname) in (None, UNKNOWN):
                db.add(
                    DataQualityIssue(
                        project_id=project.id, source_id=source.id, record_id=record.id,
                        issue_type="MISSING_FIELD", field=fname, severity=Severity.LOW,
                        description=f"{fname.replace('_', ' ').title()} not present in source. "
                                    f"Displayed as UNKNOWN.",
                        detail={"page": pr.page_number},
                        report_period=period,
                    )
                )
                stats["quality_issues"] += 1

        if not matched and p.get("agency"):
            db.add(
                DataQualityIssue(
                    project_id=project.id, source_id=source.id, record_id=record.id,
                    issue_type="UNMAPPED_AGENCY", field="sector", severity=Severity.LOW,
                    description="Agency is not in the reviewed ministry/sector mapping. "
                                "Sector shown as UNKNOWN rather than guessed.",
                    detail={"agency": p.get("agency")},
                    report_period=period,
                )
            )
            stats["quality_issues"] += 1

    db.commit()
    stats["status"] = "OK"
    stats["source_id"] = source.id
    stats["report_period"] = period
    stats["report_label"] = source.report_label
    return stats
