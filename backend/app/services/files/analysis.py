"""Upload analysis orchestrator.

Takes raw bytes plus a filename and returns one normalised structure regardless
of format. Callers never branch on file type; they read ``status`` and the
sections that are present.

Three states are possible and they are kept strictly distinct:

* ``ANALYSED``   — content was genuinely extracted.
* ``IDENTIFIED`` — the format was recognised but cannot be parsed. Said plainly.
* ``FAILED``     — parsing was attempted and failed. The reason is returned.

There is no fourth state where the assistant pretends. That is the whole design
constraint.
"""
from __future__ import annotations

import time

from . import detect as det
from . import extractors as ex
from . import sanitize
from .tabular import assess_quality

ANALYSED = "ANALYSED"
IDENTIFIED = "IDENTIFIED"
FAILED = "FAILED"

#: Hard cap on how long one file may occupy a worker.
TIMEOUT_SECONDS = 45


def analyse(filename: str, data: bytes, declared_mime: str | None = None,
            max_text_chars: int = sanitize.MAX_CHARS_DEFAULT) -> dict:
    """Analyse one uploaded file. Never raises."""
    started = time.monotonic()
    detection = det.detect(filename, data, declared_mime)

    base: dict = {
        "filename": filename,
        "size_bytes": len(data),
        "detection": detection.to_dict(),
        "status": FAILED,
        "summary_facts": {},
        "warnings": list(detection.notes),
    }

    if detection.is_empty:
        base["status"] = FAILED
        base["error"] = "The file is empty (0 bytes)."
        base["recovery"] = "Check the export completed, then upload it again."
        return base

    try:
        payload = _dispatch(detection, data)
    except Exception as exc:  # defence in depth — extractors already catch
        base["status"] = FAILED
        base["error"] = (f"The file could not be analysed ({type(exc).__name__}). "
                         "It may be corrupted or in an unexpected format.")
        base["recovery"] = "Try re-exporting the file and uploading it again."
        return base

    base["elapsed_seconds"] = round(time.monotonic() - started, 2)

    if not payload.get("ok"):
        base["status"] = IDENTIFIED if detection.identified_only else FAILED
        base["error"] = payload.get("reason")
        base["recovery"] = payload.get("recovery")
        for key in ("identified_as", "sheet_names", "sheet_errors"):
            if key in payload:
                base[key] = payload[key]
        return base

    base["status"] = ANALYSED
    base["content"] = payload
    if payload.get("summary_facts"):
        base["summary_facts"] = payload["summary_facts"]

    # A .drp that we could only *name* must not be reported as analysed. The
    # extractor returns ok=True because identification succeeded, but the user-
    # facing status has to reflect whether any project information was actually
    # recovered — otherwise the card says ANALYSED over a file we read nothing from.
    if detection.kind == det.DRP and payload.get("parseable") is False:
        base["status"] = IDENTIFIED
        base["error"] = payload.get("summary")
        base["recovery"] = ex.DRP_HANDOFF
    elif detection.kind == det.ZIP and payload.get("expansion_refused"):
        base["warnings"].append(
            "Archive contents were listed but not expanded, so nothing inside was analysed."
        )

    # --- untrusted text handling ------------------------------------------
    raw_text = payload.get("text") or payload.get("preview") or ""
    if raw_text:
        cleaned = sanitize.clean_for_analysis(raw_text, max_text_chars)
        base["text"] = cleaned["text"]
        base["text_truncated"] = cleaned["truncated"]
        base["original_text_chars"] = cleaned["original_chars"]
        if cleaned["injection_attempts"]:
            base["prompt_injection_detected"] = cleaned["injection_attempts"]
            base["warnings"].append(
                f"This file contains {len(cleaned['injection_attempts'])} passage(s) written "
                "as if instructing an AI system. They have been neutralised and are treated "
                "purely as document content."
            )

    # --- tabular quality ---------------------------------------------------
    if payload.get("kind") == "tabular" and payload.get("tables"):
        base["quality"] = assess_quality(payload["tables"])

    base["summary_facts"] = _summary_facts(detection, payload, base)
    base["warnings"].extend(_content_warnings(detection, payload))
    return base


def _dispatch(detection: det.Detection, data: bytes) -> dict:
    kind = detection.kind

    if kind == det.PDF:
        return ex.extract_pdf(data)
    if kind == det.DOCX:
        return ex.extract_docx(data)
    if kind == det.PPTX:
        return ex.extract_pptx(data)
    if kind == det.XLSX:
        return ex.extract_xlsx(data)
    if kind == det.CSV:
        return ex.extract_delimited(data, ",")
    if kind == det.TSV:
        return ex.extract_delimited(data, "\t")
    if kind == det.JSON:
        return ex.extract_json(data)
    if kind == det.XML:
        return ex.extract_xml(data)
    if kind in (det.TXT, det.MD):
        return ex.extract_text(data)
    if kind in det.IMAGE_KINDS:
        return ex.extract_image(data)
    if kind == det.ZIP:
        return ex.inspect_archive(data)
    if kind == det.DRP:
        return ex.inspect_drp(data, detection)
    if kind in det.IDENTIFIED_NOT_PARSEABLE:
        return ex.unsupported(detection)

    return ex._fail(
        "I could not recognise this file's format from its contents, so I cannot "
        "analyse it.",
        "Upload it as PDF, DOCX, XLSX, CSV, JSON, XML, TXT, MD, PPTX or an image "
        "(PNG/JPG/WEBP).",
    )


def _summary_facts(detection: det.Detection, payload: dict, base: dict) -> dict:
    """A small, flat set of extracted facts — the only figures shown to the model."""
    kind = detection.kind
    facts: dict = {"format": detection.label}

    if kind == det.PDF:
        facts.update({
            "pages": payload.get("page_count"),
            "pages_analysed": payload.get("pages_read"),
            "characters_extracted": payload.get("character_count"),
            "tables_detected": payload.get("table_count"),
            "extraction_method": payload.get("extraction_method"),
            "appears_scanned": payload.get("appears_scanned"),
        })
        if payload.get("headings"):
            facts["headings"] = payload["headings"][:15]
        if payload.get("metadata", {}).get("Title"):
            facts["document_title"] = payload["metadata"]["Title"]
    elif kind == det.DOCX:
        facts.update({"paragraphs": payload.get("paragraph_count"),
                      "tables": payload.get("table_count"),
                      "characters_extracted": payload.get("character_count"),
                      "headings": payload.get("headings", [])[:15]})
    elif kind == det.PPTX:
        facts.update({"slides": payload.get("slide_count"),
                      "slide_titles": payload.get("headings", [])[:15],
                      "characters_extracted": payload.get("character_count")})
    elif payload.get("kind") == "tabular":
        tables = payload.get("tables", [])
        facts.update({
            "tables": len(tables),
            "total_rows": sum(t["rows"] for t in tables),
            "total_columns": sum(t["columns"] for t in tables),
            "sheets": payload.get("sheet_names"),
            "columns_by_table": [
                {"sheet": t.get("sheet"), "columns": t["column_names"][:60]}
                for t in tables
            ],
        })
        if base.get("quality"):
            facts["data_quality"] = base["quality"]["overall"]
    elif kind in det.IMAGE_KINDS:
        ocr = payload.get("ocr", {})
        facts.update({
            "dimensions": f"{payload.get('width')}×{payload.get('height')} px",
            "megapixels": payload.get("megapixels"),
            "image_quality": payload.get("quality", {}).get("verdict"),
            "text_found": bool(payload.get("text") and ocr.get("ok")),
            "ocr_available": bool(ocr.get("ok")),
            "ocr_word_count": ocr.get("word_count"),
            "ocr_mean_confidence": ocr.get("mean_confidence"),
            "ocr_reliable": ocr.get("reliable"),
            "legibility": _legibility(payload),
        })
    elif kind == det.XML:
        facts.update({"root_element": payload.get("root_tag"),
                      "total_elements": payload.get("total_elements"),
                      "element_counts": payload.get("element_counts")})
    elif kind == det.JSON:
        facts["structure"] = payload.get("structure", {}).get("type")
        facts["top_level_keys"] = payload.get("structure", {}).get("keys")
    elif kind in (det.TXT, det.MD):
        facts.update({"lines": payload.get("line_count"),
                      "words": payload.get("word_count"),
                      "characters_extracted": payload.get("character_count")})
    elif kind == det.ZIP:
        facts.update({"entries": payload.get("entry_count"),
                      "uncompressed_bytes": payload.get("total_uncompressed_bytes"),
                      "expansion_refused": payload.get("expansion_refused", False)})
    elif kind == det.DRP:
        facts.update({"container_format": payload.get("container_format"),
                      "parseable": payload.get("parseable"),
                      "executed": False})

    raw_text = payload.get("text") or payload.get("preview") or ""
    metrics = extract_key_metrics(raw_text, payload)
    if any(metrics.values()):
        facts["extracted_metrics"] = metrics

    return facts


def extract_key_metrics(text: str, payload: dict) -> dict:
    """Extract key numerical metrics, project codes, costs, progress %, and delays."""
    import re
    metrics = {
        "project_codes": [],
        "cost_figures": [],
        "percentages": [],
        "delays": [],
    }
    txt = text or ""
    if not txt and isinstance(payload, dict):
        txt = str(payload.get("tables") or "")

    codes = re.findall(r"(?:code|id|project|#)\s*[:#]?\s*(\d{5,7})\b", txt, re.IGNORECASE)
    if not codes:
        codes = re.findall(r"\b(\d{6})\b", txt)
    metrics["project_codes"] = list(dict.fromkeys(codes))[:5]

    costs = re.findall(r"(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?)\s*(?:cr|crore|crores)\b", txt, re.IGNORECASE)
    metrics["cost_figures"] = [float(c.replace(",", "")) for c in costs if c][:10]

    pcts = re.findall(r"([\d,]+(?:\.\d+)?)\s*%", txt)
    metrics["percentages"] = [float(p.replace(",", "")) for p in pcts if p][:10]

    delays = re.findall(r"(\d+)\s*(?:months?|mth|mths)\b", txt, re.IGNORECASE)
    metrics["delays"] = [int(d) for d in delays if d][:5]

    return metrics


def _legibility(payload: dict) -> str:
    """One verdict on whether values in an image can be trusted.

    OCR confidence is the stronger signal when OCR actually ran, because it
    measures the thing we care about — can the characters be read — rather than
    a proxy for it. Pixel metrics decide only when OCR is unavailable or silent.
    """
    ocr = payload.get("ocr", {}) or {}
    pixel = (payload.get("quality") or {}).get("verdict")

    if ocr.get("ok") and ocr.get("word_count"):
        if ocr.get("reliable"):
            return "READABLE"
        return "PARTIALLY_READABLE"
    if ocr.get("ok"):
        return "NO_TEXT_FOUND" if pixel != "LOW" else "UNREADABLE"
    if not ocr.get("ok"):
        return "OCR_UNAVAILABLE" if pixel != "LOW" else "UNREADABLE"
    return "UNKNOWN"


def _content_warnings(detection: det.Detection, payload: dict) -> list[str]:
    warnings: list[str] = []

    if payload.get("warning"):
        warnings.append(payload["warning"])

    if detection.kind == det.PDF:
        if payload.get("pages_truncated"):
            warnings.append(
                f"Only the first {payload['pages_read']} of {payload['page_count']} pages "
                "were analysed. Ask about a specific section if you need the rest."
            )
        if payload.get("appears_scanned") and payload.get("extraction_method") == "OCR":
            warnings.append(
                "This PDF had no text layer, so the content was read by OCR. OCR "
                "misreads characters, so treat exact figures with caution."
            )

    if detection.kind in det.IMAGE_KINDS:
        quality = payload.get("quality", {})
        if quality.get("verdict") == "LOW":
            warnings.extend(quality.get("notes", []))
        ocr = payload.get("ocr", {})
        if ocr.get("ok") and ocr.get("reliable") is False:
            warnings.append(
                f"OCR confidence averaged {ocr.get('mean_confidence')}% with "
                f"{ocr.get('low_confidence_words')} low-confidence word(s). Some values in "
                "this image cannot be read reliably and are not reported as facts."
            )
        elif not ocr.get("ok") and ocr.get("reason"):
            warnings.append(ocr["reason"])

    for table in payload.get("tables", []) or []:
        if isinstance(table, dict) and table.get("truncated_rows"):
            warnings.append(
                f"Sheet '{table.get('sheet') or 'data'}' was profiled on its first "
                f"{table['rows']:,} rows only."
            )
    return warnings


def read_rows(data: bytes, filename: str, sheet: str | None = None,
              max_rows: int = 1000) -> list[dict]:
    """Re-read a tabular upload as plain rows, for cross-checking.

    The stored analysis keeps only a five-row preview on purpose — persisting
    thousands of rows of JSON alongside every attachment would bloat the
    database for no benefit. When a full comparison is actually requested the
    original bytes are re-read instead.
    """
    detection = det.detect(filename, data)
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return []

    frames: list = []
    try:
        if detection.kind == det.XLSX:
            book = pd.ExcelFile(__import__("io").BytesIO(data), engine="openpyxl")
            names = [sheet] if sheet and sheet in book.sheet_names else book.sheet_names[:4]
            frames = [book.parse(n) for n in names]
        elif detection.kind in (det.CSV, det.TSV):
            text, _ = ex._decode(data)
            delimiter = ex._sniff_delimiter(text[:8192], "\t" if detection.kind == det.TSV else ",")
            frames = [pd.read_csv(__import__("io").StringIO(text), sep=delimiter,
                                  engine="python", on_bad_lines="skip")]
        elif detection.kind == det.JSON:
            import json as _json
            parsed = _json.loads(data.decode("utf-8", errors="replace"))
            if isinstance(parsed, list) and parsed and all(isinstance(i, dict) for i in parsed):
                frames = [pd.json_normalize(parsed)]
    except Exception:
        return []

    rows: list[dict] = []
    for frame in frames:
        for _, row in frame.head(max_rows - len(rows)).iterrows():
            rows.append({
                str(k): (None if pd.isna(v) else v) for k, v in row.items()
            })
        if len(rows) >= max_rows:
            break
    return rows


# ---------------------------------------------------------------------------
# Model-facing rendering
# ---------------------------------------------------------------------------
def to_prompt_block(analysis: dict, include_text: bool = True,
                    text_budget: int = 6000) -> str:
    """Render one analysis into a compact, clearly-delimited prompt section.

    Only the structured profile and a bounded text excerpt are sent. A 40 MB
    spreadsheet never reaches the model; its computed profile does.
    """
    import json as _json

    name = analysis.get("filename", "file")
    status = analysis.get("status")

    if status == IDENTIFIED:
        return (f"FILE: {name}\nSTATUS: identified but not parseable\n"
                f"REASON: {analysis.get('error')}\n")
    if status == FAILED:
        return (f"FILE: {name}\nSTATUS: could not be analysed\n"
                f"REASON: {analysis.get('error')}\n")

    parts = [
        f"FILE: {name}",
        f"STATUS: analysed",
        "EXTRACTED FACTS (deterministic — computed by the platform, not by you):",
        _json.dumps(analysis.get("summary_facts", {}), indent=2, default=str)[:3500],
    ]

    if analysis.get("quality"):
        parts.append("DETERMINISTIC DATA-QUALITY ASSESSMENT:")
        parts.append(_json.dumps(analysis["quality"], indent=2, default=str)[:3000])

    for table in (analysis.get("content", {}).get("tables") or [])[:3]:
        if isinstance(table, dict) and table.get("sample"):
            parts.append(f"SAMPLE ROWS from '{table.get('sheet') or 'data'}':")
            parts.append(_json.dumps(table["sample"], indent=2, default=str)[:1800])

    if analysis.get("warnings"):
        parts.append("LIMITATIONS THAT MUST BE STATED IF RELEVANT:")
        parts.extend(f"- {w}" for w in analysis["warnings"][:8])

    if include_text and analysis.get("text"):
        excerpt = analysis["text"][:text_budget]
        parts.append(sanitize.wrap_untrusted(name, excerpt))

    return "\n".join(parts)
