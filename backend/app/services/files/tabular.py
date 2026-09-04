"""Deterministic profiling and quality assessment of tabular data.

Every number produced here comes from pandas, not from a language model. The
model is only ever shown the finished profile. That is the whole point: an LLM
asked to "count the duplicates in this spreadsheet" will guess, and the guess
will look plausible.

The quality verdict is rule-based and the rules are stated in the output, so a
user can disagree with the threshold rather than with an opaque judgement.
"""
from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd

GOOD = "GOOD"
NEEDS_ATTENTION = "NEEDS ATTENTION"
POOR = "POOR"
INSUFFICIENT = "INSUFFICIENT DATA"

MAX_PROFILE_ROWS = 200_000
MAX_PROFILE_COLS = 300

#: Column names that plausibly hold a project identifier, used for cross-checks.
ID_HINTS = ("project_code", "projectcode", "project id", "project_id", "projectid",
            "code", "id", "pmgid", "ocms")
NAME_HINTS = ("project name", "project_name", "projectname", "name", "title",
              "description", "project")


def _finite(value: Any) -> float | None:
    """pandas returns numpy scalars and NaN; JSON needs plain floats or None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) or math.isinf(f) else round(f, 4)


def _series_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "boolean"
    if pd.api.types.is_numeric_dtype(s):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(s):
        return "date"
    return "text"


def _maybe_numeric(s: pd.Series) -> pd.Series | None:
    """Detect a numeric column stored as text (a very common export defect)."""
    if pd.api.types.is_numeric_dtype(s):
        return None
    text = s.dropna().astype(str).str.strip()
    if text.empty:
        return None
    stripped = text.str.replace(r"[,\s₹%]", "", regex=True)
    coerced = pd.to_numeric(stripped, errors="coerce")
    if coerced.notna().mean() >= 0.9:
        return coerced
    return None


def profile_column(name: str, s: pd.Series) -> dict:
    total = len(s)
    missing = int(s.isna().sum())
    non_null = s.dropna()
    kind = _series_kind(s)

    col: dict[str, Any] = {
        "name": str(name),
        "detected_type": kind,
        "non_null": int(len(non_null)),
        "missing": missing,
        "missing_pct": round(missing / total * 100, 2) if total else 0.0,
        "unique": int(non_null.nunique()) if len(non_null) else 0,
    }

    numeric = non_null if kind == "numeric" else _maybe_numeric(s)
    if numeric is not None and len(numeric.dropna()):
        if kind != "numeric":
            col["note"] = "Stored as text but the values are numeric."
            col["detected_type"] = "numeric (stored as text)"
        nums = numeric.dropna()
        col["min"] = _finite(nums.min())
        col["max"] = _finite(nums.max())
        col["mean"] = _finite(nums.mean())
        col["median"] = _finite(nums.median())
        col["sum"] = _finite(nums.sum())
        # Outliers by the IQR rule — reported, never removed.
        if len(nums) >= 8:
            q1, q3 = nums.quantile(0.25), nums.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                outliers = nums[(nums < low) | (nums > high)]
                col["outliers"] = int(len(outliers))
                if len(outliers):
                    col["outlier_examples"] = [_finite(v) for v in outliers.head(5)]
                    col["outlier_bounds"] = [_finite(low), _finite(high)]
    elif kind == "text" and len(non_null):
        as_text = non_null.astype(str)
        col["most_common"] = [
            {"value": str(k)[:120], "count": int(v)}
            for k, v in as_text.value_counts().head(5).items()
        ]
        col["max_length"] = int(as_text.str.len().max())
        # Inconsistent casing / stray whitespace: same value, different spelling.
        normalised = as_text.str.strip().str.lower()
        if normalised.nunique() < as_text.nunique():
            col["inconsistent_formatting"] = int(as_text.nunique() - normalised.nunique())
    return col


def _find_id_column(df: pd.DataFrame) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for hint in ID_HINTS:
        for low, original in lowered.items():
            if low == hint:
                return original
    for hint in ID_HINTS:
        for low, original in lowered.items():
            if hint in low:
                return original
    return None


def profile_frame(df: pd.DataFrame, sheet_name: str | None = None) -> dict:
    """Full deterministic profile of one table."""
    truncated_rows = False
    if len(df) > MAX_PROFILE_ROWS:
        df = df.head(MAX_PROFILE_ROWS)
        truncated_rows = True
    truncated_cols = False
    if df.shape[1] > MAX_PROFILE_COLS:
        df = df.iloc[:, :MAX_PROFILE_COLS]
        truncated_cols = True

    rows, cols = df.shape
    columns = [profile_column(c, df[c]) for c in df.columns]

    total_cells = rows * cols
    missing_cells = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum()) if rows else 0

    empty_columns = [c["name"] for c in columns if c["non_null"] == 0]
    constant_columns = [
        c["name"] for c in columns if c["unique"] == 1 and c["non_null"] > 1
    ]
    unnamed_columns = [
        str(c) for c in df.columns
        if re.match(r"^unnamed:?\s*\d*$", str(c).strip(), re.I) or str(c).strip() == ""
    ]

    id_column = _find_id_column(df)
    duplicate_ids = 0
    duplicate_id_examples: list[str] = []
    if id_column is not None and rows:
        ids = df[id_column].dropna().astype(str).str.strip()
        counts = ids.value_counts()
        repeated = counts[counts > 1]
        duplicate_ids = int(repeated.sum() - len(repeated))
        duplicate_id_examples = [str(v)[:60] for v in repeated.index[:5]]

    return {
        "sheet": sheet_name,
        "rows": rows,
        "columns": cols,
        "column_names": [str(c) for c in df.columns],
        "profile": columns,
        "totals": {
            "cells": total_cells,
            "missing_cells": missing_cells,
            "missing_pct": round(missing_cells / total_cells * 100, 2) if total_cells else 0.0,
            "duplicate_rows": duplicate_rows,
            "duplicate_row_pct": round(duplicate_rows / rows * 100, 2) if rows else 0.0,
        },
        "identifier_column": str(id_column) if id_column is not None else None,
        "duplicate_identifiers": duplicate_ids,
        "duplicate_identifier_examples": duplicate_id_examples,
        "empty_columns": empty_columns,
        "constant_columns": constant_columns,
        "unnamed_columns": unnamed_columns,
        "truncated_rows": truncated_rows,
        "truncated_columns": truncated_cols,
        "sample": _sample_rows(df),
    }


def _sample_rows(df: pd.DataFrame, n: int = 5) -> list[dict]:
    """A few real rows, stringified and length-capped, for the model to see."""
    out = []
    for _, row in df.head(n).iterrows():
        out.append({
            str(k): (None if pd.isna(v) else str(v)[:120])
            for k, v in row.items()
        })
    return out


# ---------------------------------------------------------------------------
# Quality verdict
# ---------------------------------------------------------------------------
def assess_quality(profiles: list[dict]) -> dict:
    """Evidence-based quality verdict across one or more tables.

    Each finding carries the count that produced it, so nothing is asserted
    without the number behind it.
    """
    if not profiles:
        return {
            "overall": INSUFFICIENT,
            "reason": "No readable table was found in the file.",
            "findings": [], "strengths": [], "recommendation":
                "Export the data again as CSV or XLSX with a single header row.",
        }

    total_rows = sum(p["rows"] for p in profiles)
    total_cols = sum(p["columns"] for p in profiles)

    if total_rows == 0 or total_cols == 0:
        return {
            "overall": INSUFFICIENT,
            "reason": "The file parsed successfully but contains no data rows.",
            "findings": [], "strengths": [],
            "metrics": {"rows": total_rows, "columns": total_cols},
            "recommendation": "Check that the export completed and includes data below the header.",
        }

    findings: list[dict] = []
    strengths: list[str] = []
    penalty = 0.0

    missing_cells = sum(p["totals"]["missing_cells"] for p in profiles)
    total_cells = sum(p["totals"]["cells"] for p in profiles) or 1
    missing_pct = missing_cells / total_cells * 100

    # --- completeness ------------------------------------------------------
    if missing_pct >= 30:
        findings.append({"area": "Completeness", "severity": "HIGH",
                         "detail": f"{missing_pct:.1f}% of all cells are empty "
                                   f"({missing_cells:,} of {total_cells:,})."})
        penalty += 35
    elif missing_pct >= 10:
        findings.append({"area": "Completeness", "severity": "MEDIUM",
                         "detail": f"{missing_pct:.1f}% of all cells are empty "
                                   f"({missing_cells:,} of {total_cells:,})."})
        penalty += 15
    elif missing_pct >= 2:
        findings.append({"area": "Completeness", "severity": "LOW",
                         "detail": f"{missing_pct:.1f}% of cells are empty."})
        penalty += 5
    else:
        strengths.append(f"Only {missing_pct:.1f}% of cells are empty.")

    # Columns that are badly incomplete individually.
    sparse = [
        (p.get("sheet"), c["name"], c["missing_pct"])
        for p in profiles for c in p["profile"]
        if c["missing_pct"] >= 40 and c["non_null"] > 0
    ]
    if sparse:
        listed = ", ".join(f"{name} ({pct:.0f}% empty)" for _, name, pct in sparse[:5])
        findings.append({"area": "Completeness", "severity": "MEDIUM",
                         "detail": f"{len(sparse)} column(s) are largely empty: {listed}"
                                   + (" …" if len(sparse) > 5 else "")})
        penalty += min(15, 3 * len(sparse))

    empty_cols = [c for p in profiles for c in p["empty_columns"]]
    if empty_cols:
        findings.append({"area": "Structure", "severity": "LOW",
                         "detail": f"{len(empty_cols)} column(s) contain no values at all: "
                                   + ", ".join(empty_cols[:5])})
        penalty += min(8, 2 * len(empty_cols))

    # --- uniqueness --------------------------------------------------------
    dup_rows = sum(p["totals"]["duplicate_rows"] for p in profiles)
    if dup_rows:
        pct = dup_rows / total_rows * 100
        sev = "HIGH" if pct >= 10 else "MEDIUM" if pct >= 2 else "LOW"
        findings.append({"area": "Uniqueness", "severity": sev,
                         "detail": f"{dup_rows:,} fully duplicated row(s) ({pct:.1f}% of rows)."})
        penalty += min(30, pct * 2)
    else:
        strengths.append("No fully duplicated rows.")

    dup_ids = sum(p["duplicate_identifiers"] for p in profiles)
    if dup_ids:
        examples = [e for p in profiles for e in p["duplicate_identifier_examples"]][:4]
        id_col = next((p["identifier_column"] for p in profiles if p["identifier_column"]), "ID")
        findings.append({"area": "Uniqueness", "severity": "HIGH",
                         "detail": f"{dup_ids:,} repeated value(s) in the identifier column "
                                   f"'{id_col}'" + (f" — e.g. {', '.join(examples)}" if examples else "")})
        penalty += 20
    elif any(p["identifier_column"] for p in profiles):
        strengths.append("Identifier column values are unique.")

    # --- consistency -------------------------------------------------------
    inconsistent = [
        (c["name"], c["inconsistent_formatting"])
        for p in profiles for c in p["profile"] if c.get("inconsistent_formatting")
    ]
    if inconsistent:
        listed = ", ".join(f"{n} ({k} variant spellings)" for n, k in inconsistent[:4])
        findings.append({"area": "Consistency", "severity": "MEDIUM",
                         "detail": f"Case or whitespace inconsistency in {len(inconsistent)} "
                                   f"text column(s): {listed}"})
        penalty += min(12, 3 * len(inconsistent))

    text_numeric = [
        c["name"] for p in profiles for c in p["profile"]
        if c.get("detected_type") == "numeric (stored as text)"
    ]
    if text_numeric:
        findings.append({"area": "Consistency", "severity": "MEDIUM",
                         "detail": f"{len(text_numeric)} numeric column(s) are stored as text, "
                                   f"which breaks sorting and aggregation: "
                                   + ", ".join(text_numeric[:5])})
        penalty += min(12, 3 * len(text_numeric))

    unnamed = [c for p in profiles for c in p["unnamed_columns"]]
    if unnamed:
        findings.append({"area": "Structure", "severity": "MEDIUM",
                         "detail": f"{len(unnamed)} column(s) have no header name. The header "
                                   "row may be misplaced or the export may contain merged cells."})
        penalty += min(12, 4 * len(unnamed))

    # --- plausibility ------------------------------------------------------
    outlier_cols = [
        (c["name"], c["outliers"]) for p in profiles for c in p["profile"]
        if c.get("outliers")
    ]
    if outlier_cols:
        total_out = sum(k for _, k in outlier_cols)
        listed = ", ".join(f"{n} ({k})" for n, k in outlier_cols[:4])
        findings.append({"area": "Plausibility", "severity": "LOW",
                         "detail": f"{total_out} statistical outlier(s) across "
                                   f"{len(outlier_cols)} numeric column(s): {listed}. "
                                   "These may be legitimate — they are flagged, not corrected."})
        penalty += min(8, len(outlier_cols))

    # Percentage-looking columns outside 0–100.
    for p in profiles:
        for c in p["profile"]:
            if not re.search(r"(progress|percent|pct|%|completion)", c["name"], re.I):
                continue
            lo, hi = c.get("min"), c.get("max")
            if lo is None or hi is None:
                continue
            if lo < 0 or hi > 100:
                findings.append({"area": "Plausibility", "severity": "HIGH",
                                 "detail": f"Column '{c['name']}' looks like a percentage but "
                                           f"ranges from {lo} to {hi}, outside 0–100."})
                penalty += 10

    if total_rows < 5:
        findings.append({"area": "Volume", "severity": "MEDIUM",
                         "detail": f"Only {total_rows} data row(s). Statistics from a table this "
                                   "small are not reliable."})
        penalty += 10

    score = max(0.0, 100.0 - penalty)
    if total_rows < 3:
        overall = INSUFFICIENT
    elif score >= 85:
        overall = GOOD
    elif score >= 55:
        overall = NEEDS_ATTENTION
    else:
        overall = POOR

    high = [f for f in findings if f["severity"] == "HIGH"]
    if overall == GOOD and high:
        overall = NEEDS_ATTENTION

    recommendation = {
        GOOD: "The dataset is fit for analysis as it stands.",
        NEEDS_ATTENTION: "Resolve the issues listed above before relying on this dataset "
                         "for project decisions. Most are fixable in the source export.",
        POOR: "This dataset has enough structural problems that analysis built on it would "
              "be unreliable. Correct it at source and re-export.",
        INSUFFICIENT: "There is not enough data here to judge quality.",
    }[overall]

    return {
        "overall": overall,
        "quality_score": round(score, 1),
        "scoring_note": "Score starts at 100 and is reduced by each finding below. "
                        "GOOD ≥ 85, NEEDS ATTENTION ≥ 55, POOR below 55.",
        "findings": findings,
        "strengths": strengths,
        "recommendation": recommendation,
        "metrics": {
            "tables": len(profiles),
            "rows": total_rows,
            "columns": total_cols,
            "missing_pct": round(missing_pct, 2),
            "duplicate_rows": dup_rows,
            "duplicate_identifiers": dup_ids,
        },
    }
