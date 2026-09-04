"""Compare an uploaded dataset against verified PAIMANA snapshots.

Two independent records of the same project will disagree. The useful question
is not "which one is right" — this service cannot know that, and says so — but
"where do they disagree, and by how much".

Every comparison row is labelled with all four provenance classes the platform
uses: the uploaded value, the PAIMANA value, the derived difference, and any
interpretation. Interpretation is offered as a *possible* explanation and never
as a finding.
"""
from __future__ import annotations

import math
import re
from typing import Any

from sqlalchemy.orm import Session

from ..models import Project, ProjectSnapshot
from . import analytics

#: Column name patterns mapped onto PAIMANA snapshot fields. Matching is on the
#: normalised header text, so "Physical Progress (%)" and "physical_progress"
#: both land on the same field.
FIELD_PATTERNS: list[tuple[str, str, str]] = [
    ("physical_progress", r"physical.*progress|progress.*physical|^progress$|%\s*complete|completion\s*%", "percentage points"),
    ("financial_progress", r"financial.*progress|financial.*%", "percentage points"),
    ("original_cost", r"original.*cost|approved.*cost|sanctioned.*cost|initial.*cost", "₹ crore"),
    ("revised_cost", r"revised.*cost|current.*cost|latest.*cost|anticipated.*cost", "₹ crore"),
    ("expenditure", r"expenditure|expense|spent|amount.*spent|actual.*cost", "₹ crore"),
    ("schedule_delay_months", r"delay|slippage|months.*late|time.*overrun", "months"),
    ("cost_escalation_pct", r"cost.*escalat|cost.*overrun|escalation.*%", "percentage points"),
    ("risk_score", r"risk.*score|risk.*rating", "points"),
]

FIELD_LABELS = {
    "physical_progress": "Physical progress",
    "financial_progress": "Financial progress",
    "original_cost": "Original cost",
    "revised_cost": "Revised cost",
    "expenditure": "Expenditure",
    "schedule_delay_months": "Schedule delay",
    "cost_escalation_pct": "Cost escalation",
    "risk_score": "Risk score",
}

#: Below this the two sources are treated as agreeing. Rounding and reporting
#: conventions produce differences this small routinely.
TOLERANCE = {
    "physical_progress": 0.5,
    "financial_progress": 0.5,
    "cost_escalation_pct": 0.5,
    "risk_score": 1.0,
    "schedule_delay_months": 0.5,
    "original_cost": 1.0,
    "revised_cost": 1.0,
    "expenditure": 1.0,
}

CODE_COLUMN_HINTS = ("project_code", "projectcode", "project code", "code", "project_id",
                     "project id", "projectid", "pmgid", "ocms")
NAME_COLUMN_HINTS = ("project name", "project_name", "projectname", "name", "project",
                     "title", "description")


def _norm(text: Any) -> str:
    return re.sub(r"[^a-z0-9%]+", " ", str(text or "").lower()).strip()


def map_columns(column_names: list[str]) -> dict[str, str]:
    """Map spreadsheet headers onto PAIMANA snapshot fields."""
    mapping: dict[str, str] = {}
    for column in column_names:
        normalised = _norm(column)
        if not normalised:
            continue
        for field, pattern, _unit in FIELD_PATTERNS:
            if field in mapping.values():
                continue
            if re.search(pattern, normalised):
                mapping[column] = field
                break
    return mapping


def find_key_columns(column_names: list[str]) -> tuple[str | None, str | None]:
    """Pick the identifier and the human-readable name column.

    Specific hints are matched before loose ones, and the column already chosen
    as the code is excluded from the name search. Without both guards, a header
    set of ["Project Code", "Project Name"] resolves the name to "Project Code",
    because "project code" starts with the loose hint "project".
    """
    lowered = [(c, _norm(c)) for c in column_names]

    code_column = None
    for hints in (("project code", "project_code", "projectcode", "project id",
                   "project_id", "projectid", "pmgid", "ocms"), ("code", "id")):
        for original, normalised in lowered:
            if normalised in hints:
                code_column = original
                break
        if code_column:
            break
    if code_column is None:
        for original, normalised in lowered:
            if any(h in normalised for h in CODE_COLUMN_HINTS):
                code_column = original
                break

    name_column = None
    for hints in (("project name", "project_name", "projectname"),
                  ("name", "title", "description"), ("project",)):
        for original, normalised in lowered:
            if original == code_column:
                continue
            if normalised in hints:
                name_column = original
                break
        if name_column:
            break
    if name_column is None:
        for original, normalised in lowered:
            if original == code_column:
                continue
            if any(normalised.startswith(h) for h in NAME_COLUMN_HINTS):
                name_column = original
                break

    return code_column, name_column


def _to_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None if isinstance(value, float) and math.isnan(value) else float(value)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "-", "n/a", "na", "unknown"}:
        return None
    text = re.sub(r"[₹,%\s]", "", text)
    text = re.sub(r"(cr|crore|crores|rs\.?|inr)$", "", text, flags=re.I)
    try:
        return float(text)
    except ValueError:
        return None


def match_project(db: Session, code: str | None, name: str | None) -> Project | None:
    """Resolve a row to a PAIMANA project. Exact code first, then name."""
    if code:
        cleaned = str(code).strip()
        hit = db.query(Project).filter_by(project_code=cleaned).one_or_none()
        if hit:
            return hit
        digits = re.sub(r"\D", "", cleaned)
        if digits and digits != cleaned:
            hit = db.query(Project).filter_by(project_code=digits).one_or_none()
            if hit:
                return hit
    if name:
        cleaned = str(name).strip()
        if len(cleaned) >= 4:
            hit = (
                db.query(Project)
                .filter(Project.name.ilike(cleaned))
                .first()
            )
            if hit:
                return hit
            hit = (
                db.query(Project)
                .filter(Project.name.ilike(f"%{cleaned[:60]}%"))
                .first()
            )
            if hit:
                return hit
    return None


def latest_snapshot(db: Session, project: Project, period: str | None = None) -> ProjectSnapshot | None:
    query = (
        db.query(ProjectSnapshot)
        .filter_by(project_id=project.id, supersedes_id=None)
    )
    if period:
        query = query.filter_by(report_period=period)
    return query.order_by(ProjectSnapshot.report_period.desc()).first()


def compare_rows(db: Session, rows: list[dict], column_map: dict[str, str],
                 code_column: str | None, name_column: str | None,
                 period: str | None = None, max_rows: int = 200) -> dict:
    """Compare uploaded rows against verified snapshots.

    Returns matched projects with a per-field comparison, plus the rows that
    could not be matched at all. Nothing is asserted about which source is
    correct.
    """
    matched: list[dict] = []
    unmatched: list[dict] = []
    examined = 0

    for row in rows[:max_rows]:
        examined += 1
        code = row.get(code_column) if code_column else None
        name = row.get(name_column) if name_column else None
        project = match_project(db, code, name)

        if project is None:
            unmatched.append({
                "identifier": str(code or name or "(no identifier column)")[:80],
            })
            continue

        snapshot = latest_snapshot(db, project, period)
        if snapshot is None:
            unmatched.append({
                "identifier": str(code or name)[:80],
                "reason": f"Matched PAIMANA project {project.project_code} but it has no "
                          "snapshot on record.",
            })
            continue

        comparisons = []
        for column, field in column_map.items():
            uploaded = _to_number(row.get(column))
            verified = getattr(snapshot, field, None)
            if uploaded is None or verified is None:
                comparisons.append({
                    "field": FIELD_LABELS.get(field, field),
                    "uploaded_column": column,
                    "uploaded_value": uploaded,
                    "paimana_value": verified,
                    "status": "INCOMPARABLE",
                    "note": "One of the two sources has no value for this field.",
                })
                continue

            difference = round(uploaded - float(verified), 3)
            unit = next((u for f, _p, u in FIELD_PATTERNS if f == field), "")
            agrees = abs(difference) <= TOLERANCE.get(field, 0.5)
            comparisons.append({
                "field": FIELD_LABELS.get(field, field),
                "uploaded_column": column,
                "uploaded_value": uploaded,
                "paimana_value": round(float(verified), 3),
                "difference": difference,
                "unit": unit,
                "status": "AGREES" if agrees else "DIFFERS",
            })

        discrepancies = [c for c in comparisons if c["status"] == "DIFFERS"]
        matched.append({
            "project_code": project.project_code,
            "project_name": project.name,
            "state": project.state,
            "sector": project.sector,
            "paimana_period": snapshot.report_period,
            "comparisons": comparisons,
            "discrepancy_count": len(discrepancies),
        })

    with_discrepancies = [m for m in matched if m["discrepancy_count"]]

    return {
        "rows_examined": examined,
        "rows_available": len(rows),
        "matched_count": len(matched),
        "unmatched_count": len(unmatched),
        "matched": matched,
        "unmatched_examples": unmatched[:10],
        "projects_with_discrepancies": len(with_discrepancies),
        "discrepancies": with_discrepancies[:25],
        "paimana_period": period or analytics.latest_period(db),
        "interpretation_note": (
            "A difference is not evidence that either source is wrong. The uploaded file "
            "and PAIMANA may use different reporting dates, different measurement "
            "definitions, or different revision states of the same figure."
        ),
        "labels": {
            "UPLOADED FILE": "values read from the file you attached",
            "PAIMANA VERIFIED DATA": "values from ingested Flash Report snapshots",
            "DERIVED CALCULATION": "the difference, computed by the platform",
            "AI INTERPRETATION": "any suggested explanation, which is not a finding",
        },
    }


def cross_check(db: Session, analysis: dict, period: str | None = None) -> dict:
    """Entry point: cross-check one analysed file against PAIMANA."""
    content = analysis.get("content") or {}
    if content.get("kind") != "tabular" or not content.get("tables"):
        return {
            "applicable": False,
            "reason": ("Cross-checking against PAIMANA needs tabular data with a project "
                       "identifier column. This file is not a table."),
        }

    if analytics.latest_period(db) is None:
        return {
            "applicable": False,
            "reason": ("There is no ingested PAIMANA data to compare against yet, so I "
                       "cannot cross-check this file."),
        }

    best: dict | None = None
    for table in content["tables"]:
        columns = table.get("column_names", [])
        mapping = map_columns(columns)
        code_column, name_column = find_key_columns(columns)
        if not mapping or (code_column is None and name_column is None):
            continue
        candidate = {"table": table, "mapping": mapping,
                     "code_column": code_column, "name_column": name_column}
        if best is None or len(mapping) > len(best["mapping"]):
            best = candidate

    if best is None:
        return {
            "applicable": False,
            "reason": ("I could not find both a project identifier column and any "
                       "comparable metric column (progress, cost, expenditure, delay) in "
                       "this file."),
            "columns_seen": [
                c for t in content["tables"] for c in t.get("column_names", [])
            ][:40],
        }

    # The stored profile keeps only a small sample of rows; re-read the file for
    # a full comparison when the caller supplies it.
    rows = best["table"].get("full_rows") or best["table"].get("sample") or []

    result = compare_rows(
        db, rows, best["mapping"], best["code_column"], best["name_column"], period
    )
    result["applicable"] = True
    result["sheet"] = best["table"].get("sheet")
    result["column_mapping"] = {k: FIELD_LABELS.get(v, v) for k, v in best["mapping"].items()}
    result["identifier_column"] = best["code_column"] or best["name_column"]
    result["rows_in_table"] = best["table"].get("rows")
    if result["rows_available"] < (best["table"].get("rows") or 0):
        result["sampling_note"] = (
            f"The comparison ran on {result['rows_available']} row(s) held in the stored "
            f"profile out of {best['table'].get('rows')} in the file."
        )
    return result
