"""Per-format content extraction.

Each extractor returns a dict and never raises. A failure is a *result* — a
structured explanation of what went wrong — because the caller has to tell the
user something useful, and a traceback is not useful.

Nothing in this module executes, evaluates, or resolves anything inside an
uploaded file. Archives are inspected by listing, never by extracting to disk.
XML is parsed with defusedxml so entity-expansion attacks cannot land.
"""
from __future__ import annotations

import csv
import io
import json
import zipfile
from typing import Any

from . import detect as det
from .tabular import profile_frame

# Optional dependencies are imported lazily so a missing one degrades one format
# rather than breaking the whole upload path.
MAX_PDF_PAGES = 60
MAX_ARCHIVE_ENTRIES = 500
MAX_ARCHIVE_RATIO = 100          # compressed:uncompressed guard against ZIP bombs
MAX_ARCHIVE_UNCOMPRESSED = 500 * 1024 * 1024


def _fail(reason: str, recovery: str | None = None, **extra) -> dict:
    return {"ok": False, "reason": reason, "recovery": recovery, **extra}


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def extract_pdf(data: bytes) -> dict:
    try:
        import pdfplumber
    except ImportError:  # pragma: no cover
        return _fail("PDF support is not installed on the server.",
                     "Contact the administrator, or upload the content as text.")

    try:
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            total_pages = len(pdf.pages)
            pages_read = min(total_pages, MAX_PDF_PAGES)
            texts: list[str] = []
            tables: list[dict] = []
            headings: list[str] = []

            for i in range(pages_read):
                page = pdf.pages[i]
                try:
                    text = page.extract_text() or ""
                except Exception:
                    text = ""
                texts.append(text)

                for line in text.splitlines():
                    stripped = line.strip()
                    if not (4 <= len(stripped) <= 90):
                        continue
                    letters = [c for c in stripped if c.isalpha()]
                    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.75:
                        if stripped not in headings:
                            headings.append(stripped)

                if len(tables) < 12:
                    try:
                        for raw in page.extract_tables() or []:
                            if raw and len(raw) > 1:
                                tables.append({
                                    "page": i + 1,
                                    "rows": len(raw),
                                    "columns": max(len(r) for r in raw),
                                    "header": [str(c)[:60] if c else "" for c in raw[0]],
                                    "preview": [
                                        [str(c)[:40] if c else "" for c in row]
                                        for row in raw[1:4]
                                    ],
                                })
                    except Exception:
                        pass

            metadata = {}
            try:
                metadata = {
                    k: str(v)[:200] for k, v in (pdf.metadata or {}).items() if v
                }
            except Exception:
                pass

        full_text = "\n\n".join(t for t in texts if t).strip()
        char_count = len(full_text)
        # A PDF whose pages yield almost no characters is a scan.
        chars_per_page = char_count / pages_read if pages_read else 0
        is_scanned = chars_per_page < 40

        result = {
            "ok": True,
            "text": full_text,
            "page_count": total_pages,
            "pages_read": pages_read,
            "pages_truncated": total_pages > pages_read,
            "headings": headings[:40],
            "tables": tables,
            "table_count": len(tables),
            "metadata": metadata,
            "character_count": char_count,
            "appears_scanned": is_scanned,
        }

        if is_scanned:
            ocr = _ocr_pdf(data, pages_read)
            result["ocr"] = ocr
            if ocr.get("ok") and ocr.get("text"):
                result["text"] = ocr["text"]
                result["character_count"] = len(ocr["text"])
                result["extraction_method"] = "OCR"
            else:
                result["extraction_method"] = "none"
                result["warning"] = (
                    "This PDF contains almost no extractable text layer, which means it is "
                    "very likely a scan or a set of page images. "
                    + (ocr.get("reason") or "OCR is not available on this server.")
                )
        else:
            result["extraction_method"] = "text layer"
        return result

    except Exception as exc:
        return _fail(
            f"The PDF could not be read ({type(exc).__name__}). It may be corrupted, "
            "encrypted, or password-protected.",
            "Try re-exporting or re-saving the PDF, or remove the password and upload again.",
        )


def _ocr_pdf(data: bytes, max_pages: int) -> dict:
    """OCR fallback for scanned PDFs. Optional — absent tooling is reported."""
    try:
        import pypdfium2 as pdfium
        import pytesseract
        from PIL import Image
    except ImportError:
        return _fail("OCR libraries are not installed on this server.")

    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return _fail("The Tesseract OCR engine is not installed on this server.")

    try:
        pdf = pdfium.PdfDocument(io.BytesIO(data))
        pages = min(len(pdf), max_pages, 12)
        chunks = []
        for i in range(pages):
            bitmap = pdf[i].render(scale=2)
            image: Image.Image = bitmap.to_pil()
            chunks.append(pytesseract.image_to_string(image))
        text = "\n\n".join(c for c in chunks if c.strip()).strip()
        return {"ok": True, "text": text, "pages_ocred": pages,
                "engine": "tesseract"} if text else _fail(
            "OCR ran but found no readable text on the scanned pages.")
    except Exception as exc:
        return _fail(f"OCR failed ({type(exc).__name__}).")


# ---------------------------------------------------------------------------
# Word / PowerPoint
# ---------------------------------------------------------------------------
def extract_docx(data: bytes) -> dict:
    try:
        import docx
    except ImportError:  # pragma: no cover
        return _fail("Word document support is not installed on this server.")
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        return _fail(f"The Word document could not be opened ({type(exc).__name__}). "
                     "It may be corrupted or in the older .doc format.",
                     "Re-save it as .docx and upload again.")

    paragraphs, headings = [], []
    for p in document.paragraphs:
        text = p.text.strip()
        if not text:
            continue
        paragraphs.append(text)
        if (p.style.name or "").lower().startswith("heading"):
            headings.append(text)

    tables = []
    for t in document.tables[:15]:
        rows = [[c.text.strip()[:60] for c in r.cells] for r in t.rows[:4]]
        tables.append({"rows": len(t.rows), "columns": len(t.columns),
                       "header": rows[0] if rows else [], "preview": rows[1:4]})

    text = "\n\n".join(paragraphs)
    return {
        "ok": True, "text": text, "paragraph_count": len(paragraphs),
        "headings": headings[:40], "tables": tables, "table_count": len(document.tables),
        "character_count": len(text), "extraction_method": "docx XML",
    }


def extract_pptx(data: bytes) -> dict:
    try:
        from pptx import Presentation
    except ImportError:  # pragma: no cover
        return _fail("PowerPoint support is not installed on this server.")
    try:
        deck = Presentation(io.BytesIO(data))
    except Exception as exc:
        return _fail(f"The presentation could not be opened ({type(exc).__name__}). "
                     "It may be corrupted or in the older .ppt format.",
                     "Re-save it as .pptx and upload again.")

    slides, headings = [], []
    for index, slide in enumerate(deck.slides, start=1):
        parts = []
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = shape.text_frame.text.strip()
                if text:
                    parts.append(text)
        title = ""
        try:
            if slide.shapes.title is not None:
                title = (slide.shapes.title.text or "").strip()
        except Exception:
            pass
        if title:
            headings.append(title)
        notes = ""
        try:
            if slide.has_notes_slide:
                notes = (slide.notes_slide.notes_text_frame.text or "").strip()
        except Exception:
            pass
        slides.append({"slide": index, "title": title,
                       "text": "\n".join(parts)[:2000], "notes": notes[:800]})

    text = "\n\n".join(
        f"--- Slide {s['slide']}" + (f": {s['title']}" if s["title"] else "") + f" ---\n{s['text']}"
        for s in slides
    )
    return {"ok": True, "text": text, "slide_count": len(slides), "slides": slides,
            "headings": headings[:40], "character_count": len(text),
            "extraction_method": "pptx XML"}


# ---------------------------------------------------------------------------
# Spreadsheets and delimited text
# ---------------------------------------------------------------------------
def extract_xlsx(data: bytes) -> dict:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return _fail("Spreadsheet support is not installed on this server.")
    try:
        book = pd.ExcelFile(io.BytesIO(data), engine="openpyxl")
    except Exception as exc:
        return _fail(f"The workbook could not be opened ({type(exc).__name__}). It may be "
                     "corrupted, password-protected, or saved in the legacy .xls format.",
                     "Open it in Excel and re-save as .xlsx, or export the sheet as CSV.")

    tables, errors = [], []
    for name in book.sheet_names[:12]:
        try:
            frame = book.parse(name)
        except Exception as exc:
            errors.append({"sheet": name, "error": type(exc).__name__})
            continue
        if frame.empty and frame.shape[1] == 0:
            errors.append({"sheet": name, "error": "empty sheet"})
            continue
        tables.append(profile_frame(frame, sheet_name=name))

    if not tables:
        return _fail("The workbook opened but no sheet contained readable tabular data.",
                     "Check that the sheets have a header row and data beneath it.",
                     sheet_names=book.sheet_names, sheet_errors=errors)

    return {"ok": True, "kind": "tabular", "sheet_names": book.sheet_names,
            "sheet_count": len(book.sheet_names), "tables": tables,
            "sheet_errors": errors, "extraction_method": "openpyxl + pandas"}


def _sniff_delimiter(sample: str, default: str = ",") -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return default


def extract_delimited(data: bytes, default_delimiter: str = ",") -> dict:
    try:
        import pandas as pd
    except ImportError:  # pragma: no cover
        return _fail("Tabular support is not installed on this server.")

    text, encoding = _decode(data)
    if not text.strip():
        return _fail("The file is empty or contains only whitespace.")

    delimiter = _sniff_delimiter(text[:8192], default_delimiter)
    try:
        frame = pd.read_csv(io.StringIO(text), sep=delimiter, engine="python",
                            on_bad_lines="warn")
    except Exception as exc:
        return _fail(f"The delimited file could not be parsed ({type(exc).__name__}). "
                     "Rows may have inconsistent column counts, or the header may be missing.",
                     "Re-export it from the source system as a clean CSV or XLSX.")

    if frame.empty and frame.shape[1] == 0:
        return _fail("The file parsed but contained no columns.")

    return {"ok": True, "kind": "tabular", "tables": [profile_frame(frame)],
            "delimiter": delimiter, "encoding": encoding,
            "extraction_method": "pandas"}


def _decode(data: bytes) -> tuple[str, str]:
    for encoding in ("utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding), encoding
        except (UnicodeDecodeError, UnicodeError):
            continue
    return data.decode("utf-8", errors="replace"), "utf-8 (with replacements)"


# ---------------------------------------------------------------------------
# Structured text
# ---------------------------------------------------------------------------
def extract_json(data: bytes) -> dict:
    text, encoding = _decode(data)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return _fail(f"The JSON is malformed: {exc.msg} at line {exc.lineno}, "
                     f"column {exc.colno}.",
                     "Validate the file in a JSON linter and upload the corrected version.")

    structure = _describe_json(parsed)
    result = {"ok": True, "text": json.dumps(parsed, indent=2, default=str)[:20000],
              "encoding": encoding, "structure": structure,
              "extraction_method": "json"}

    # A list of flat objects is really a table — profile it as one.
    if isinstance(parsed, list) and parsed and all(isinstance(i, dict) for i in parsed):
        try:
            import pandas as pd
            frame = pd.json_normalize(parsed)
            result["kind"] = "tabular"
            result["tables"] = [profile_frame(frame)]
        except Exception:
            pass
    return result


def _describe_json(value: Any, depth: int = 0) -> dict:
    if depth > 4:
        return {"type": "…"}
    if isinstance(value, dict):
        return {"type": "object", "keys": list(value.keys())[:40],
                "key_count": len(value),
                "children": {k: _describe_json(v, depth + 1)
                             for k, v in list(value.items())[:12]}}
    if isinstance(value, list):
        return {"type": "array", "length": len(value),
                "item": _describe_json(value[0], depth + 1) if value else None}
    return {"type": type(value).__name__}


def extract_xml(data: bytes) -> dict:
    try:
        from defusedxml import ElementTree as SafeET
    except ImportError:
        return _fail("Safe XML parsing is not available on this server. XML is not parsed "
                     "with the standard library here because of entity-expansion risk.")
    try:
        root = SafeET.fromstring(data)
    except Exception as exc:
        return _fail(f"The XML could not be parsed ({type(exc).__name__}). It may be "
                     "malformed, or it may use constructs that are blocked for safety "
                     "(external entities, DTDs).",
                     "Validate the XML and remove any DOCTYPE/entity declarations.")

    counts: dict[str, int] = {}
    texts: list[str] = []

    def walk(node, depth=0):
        tag = str(node.tag).split("}")[-1]
        counts[tag] = counts.get(tag, 0) + 1
        if node.text and node.text.strip() and len(texts) < 800:
            texts.append(f"{tag}: {node.text.strip()[:200]}")
        if depth < 12:
            for child in list(node)[:400]:
                walk(child, depth + 1)

    walk(root)
    return {"ok": True, "text": "\n".join(texts),
            "root_tag": str(root.tag).split("}")[-1],
            "element_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:30]),
            "total_elements": sum(counts.values()),
            "character_count": sum(len(t) for t in texts),
            "extraction_method": "defusedxml"}


def extract_text(data: bytes) -> dict:
    text, encoding = _decode(data)
    lines = text.splitlines()
    headings = [
        line.strip() for line in lines
        if line.strip().startswith("#") or (
            4 <= len(line.strip()) <= 80
            and line.strip() == line.strip().upper()
            and any(c.isalpha() for c in line)
        )
    ]
    return {"ok": True, "text": text, "encoding": encoding, "line_count": len(lines),
            "word_count": len(text.split()), "character_count": len(text),
            "headings": headings[:40], "extraction_method": "plain text"}


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def extract_image(data: bytes) -> dict:
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        return _fail("Image support is not installed on this server.")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as exc:
        return _fail(f"The image could not be decoded ({type(exc).__name__}). "
                     "It may be truncated or corrupted.",
                     "Re-export or re-save the image and upload again.")

    width, height = image.size
    quality_info = _image_quality(image, width, height)
    ocr_info = _ocr_image(image)

    ocr_available = bool(ocr_info.get("ok"))
    ocr_text = ocr_info.get("text", "") if ocr_available else ""
    text_found = bool(ocr_text and ocr_available)

    if ocr_text:
        extracted_text = ocr_text
    else:
        extracted_text = (
            f"[Uploaded Image Asset: {image.format or 'Image'} ({width}×{height} px, "
            f"{round(width * height / 1_000_000, 2)} MP, {image.mode} color). "
            f"Quality Assessment: {quality_info.get('verdict', 'OK')}. "
            f"Doc-type: {'Document/Report Screenshot' if quality_info.get('document_like') else 'Photo/Visual Asset'}.]"
        )

    result: dict[str, Any] = {
        "ok": True,
        "width": width,
        "height": height,
        "format": image.format or "PNG",
        "mode": image.mode,
        "megapixels": round(width * height / 1_000_000, 2),
        "extraction_method": "pillow",
        "quality": quality_info,
        "ocr": ocr_info,
        "text": extracted_text,
        "character_count": len(extracted_text),
        "summary_facts": {
            "format": f"{image.format or 'PNG'} image",
            "dimensions": f"{width}×{height} px",
            "megapixels": round(width * height / 1_000_000, 2),
            "image_quality": quality_info.get("verdict", "OK"),
            "text_found": text_found,
            "ocr_available": ocr_available,
            "ocr_word_count": ocr_info.get("word_count") if ocr_available else None,
            "ocr_mean_confidence": ocr_info.get("mean_confidence") if ocr_available else None,
            "ocr_reliable": ocr_info.get("reliable") if ocr_available else None,
            "legibility": "READABLE" if (text_found and ocr_info.get("reliable")) else ("OCR_UNAVAILABLE" if not ocr_available else "UNREADABLE"),
        },
    }
    return result


def _image_quality(image, width: int, height: int) -> dict:
    """Assess whether this image can be read reliably.

    A screenshot of a report is mostly white, so global mean brightness and
    global standard deviation are meaningless for it — judged that way, every
    clean document scores as "washed out and low contrast". So the image is
    first classified as document-like or photographic, and only the metrics that
    actually mean something for that class are applied.

    Blur is measured as the variance of the Laplacian over the *ink* region
    rather than the whole frame, for the same reason.
    """
    notes: list[str] = []
    metrics: dict[str, Any] = {"pixels": f"{width}×{height}"}
    verdict = "OK"

    try:
        import numpy as np
    except ImportError:  # pragma: no cover
        return {"verdict": "UNKNOWN", "metrics": metrics,
                "notes": ["Image quality metrics unavailable (numpy not installed)."],
                "document_like": None}

    try:
        grey = image.convert("L")
        if max(grey.size) > 1600:
            ratio = 1600 / max(grey.size)
            grey = grey.resize((max(1, int(grey.width * ratio)),
                                max(1, int(grey.height * ratio))))
        a = np.asarray(grey, dtype="float32")

        near_white = float((a >= 225).mean())
        document_like = near_white >= 0.55
        metrics["background_fraction"] = round(near_white, 3)
        metrics["document_like"] = document_like

        # Laplacian (4-neighbour) — the standard blur metric.
        lap = (
            -4 * a[1:-1, 1:-1]
            + a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]
        )
        if document_like:
            # Restrict to the region that actually carries ink.
            ink_mask = a[1:-1, 1:-1] < 200
            ink_fraction = float(ink_mask.mean())
            metrics["ink_fraction"] = round(ink_fraction, 4)
            region = lap[ink_mask] if ink_mask.any() else lap
            sharpness = float(region.var()) if region.size else 0.0
            # How dark is the ink against the paper? This is the real contrast
            # measure for a document.
            ink = a[a < 200]
            paper = a[a >= 225]
            if ink.size and paper.size:
                separation = float(paper.mean() - ink.mean())
                metrics["ink_paper_separation"] = round(separation, 1)
                if separation < 60:
                    notes.append("Text and background are very close in tone, which makes "
                                 "characters hard to separate.")
                    verdict = "LOW"
            if ink_fraction < 0.0015:
                notes.append("The image is almost entirely blank — there is very little "
                             "content to analyse.")
                verdict = "LOW"
        else:
            sharpness = float(lap.var())
            mean, std = float(a.mean()), float(a.std())
            metrics["mean_brightness"] = round(mean, 1)
            metrics["contrast"] = round(std, 1)
            if mean < 45:
                notes.append("The image is very dark.")
                verdict = "LOW"
            elif mean > 235:
                notes.append("The image is very bright or over-exposed.")
                verdict = "LOW"
            if std < 18:
                notes.append("Overall contrast is very low.")
                verdict = "LOW"

        metrics["sharpness"] = round(sharpness, 1)
        blur_floor = 120.0 if document_like else 60.0
        if sharpness < blur_floor:
            notes.append("The image is blurry or lacks fine detail, so small text may not "
                         "be readable.")
            verdict = "LOW"

        # Resolution matters for legibility, but only flag it when it is
        # genuinely marginal for text.
        if width * height < 90_000:          # smaller than roughly 300×300
            notes.append(f"Low resolution ({width}×{height} px). Small text is unlikely "
                         "to be legible.")
            verdict = "LOW"
    except Exception:
        notes.append("Quality metrics could not be computed for this image.")
        return {"verdict": "UNKNOWN", "metrics": metrics, "notes": notes,
                "document_like": None}

    return {"verdict": verdict, "metrics": metrics, "notes": notes,
            "document_like": metrics.get("document_like")}


def _ocr_image(image) -> dict:
    try:
        import pytesseract
    except ImportError:
        return _fail("OCR is not installed on this server, so text inside the image "
                     "cannot be read.")
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        return _fail("The Tesseract OCR engine is not installed on this server, so text "
                     "inside the image cannot be read.")

    try:
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words, confidences = [], []
        low_confidence = 0
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = (text or "").strip()
            if not text:
                continue
            try:
                confidence = float(conf)
            except (TypeError, ValueError):
                continue
            if confidence < 0:
                continue
            words.append(text)
            confidences.append(confidence)
            if confidence < 60:
                low_confidence += 1

        if not words:
            return {"ok": True, "text": "", "word_count": 0,
                    "note": "OCR ran successfully but found no readable text in the image."}

        mean_conf = sum(confidences) / len(confidences)
        return {
            "ok": True,
            "text": pytesseract.image_to_string(image).strip(),
            "word_count": len(words),
            "mean_confidence": round(mean_conf, 1),
            "low_confidence_words": low_confidence,
            "reliable": mean_conf >= 70 and low_confidence / len(words) < 0.3,
            "engine": "tesseract",
        }
    except Exception as exc:
        return _fail(f"OCR failed on this image ({type(exc).__name__}).")


# ---------------------------------------------------------------------------
# Archives — listed only, never extracted to disk
# ---------------------------------------------------------------------------
def inspect_archive(data: bytes) -> dict:
    """List archive contents with traversal and expansion-bomb guards.

    Nothing is written to disk and nothing is executed. Entry names are checked
    for path traversal and the compression ratio is checked before any read.
    """
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return _fail("The archive is not a valid ZIP file, or it is corrupted.",
                     "Re-create the archive and upload it again.")
    except Exception as exc:
        return _fail(f"The archive could not be opened ({type(exc).__name__}).")

    try:
        infos = archive.infolist()
    except Exception as exc:
        return _fail(f"The archive index could not be read ({type(exc).__name__}).")

    entries, unsafe = [], []
    compressed = uncompressed = 0

    for info in infos[:MAX_ARCHIVE_ENTRIES]:
        name = info.filename
        if name.startswith("/") or ".." in name.replace("\\", "/").split("/") or (
            len(name) > 1 and name[1] == ":"
        ):
            unsafe.append(name[:120])
            continue
        compressed += info.compress_size
        uncompressed += info.file_size
        entries.append({
            "name": name[:160],
            "size": info.file_size,
            "compressed_size": info.compress_size,
            "is_directory": name.endswith("/"),
        })

    ratio = (uncompressed / compressed) if compressed else 0
    bomb = ratio > MAX_ARCHIVE_RATIO or uncompressed > MAX_ARCHIVE_UNCOMPRESSED

    result = {
        "ok": True,
        "entry_count": len(infos),
        "entries_listed": len(entries),
        "entries_truncated": len(infos) > MAX_ARCHIVE_ENTRIES,
        "total_uncompressed_bytes": uncompressed,
        "compression_ratio": round(ratio, 1),
        "unsafe_entries": unsafe,
        "entries": entries[:120],
        "extraction_method": "listing only — no entry was extracted or executed",
    }
    if unsafe:
        result["warning"] = (
            f"{len(unsafe)} entry name(s) attempt to escape the archive root (path "
            "traversal). They were refused and not listed among the contents."
        )
    if bomb:
        result["warning"] = (
            f"This archive expands to {uncompressed / 1_048_576:.0f} MB at a "
            f"{ratio:.0f}:1 compression ratio. It was listed but deliberately not "
            "expanded."
        )
        result["expansion_refused"] = True
    return result


# ---------------------------------------------------------------------------
# DRP — identified honestly, never executed
# ---------------------------------------------------------------------------
DRP_HANDOFF = (
    "If you export the relevant information as PDF, CSV, XLSX, XML or a screenshot, "
    "I can analyse that fully."
)


def inspect_drp(data: bytes, detection: det.Detection) -> dict:
    """Identify what a .drp file actually is, without executing it.

    ``.drp`` is not one format. It is used by several unrelated products
    (Digital Rebar provisioning bundles, DaVinci Resolve project archives,
    various in-house planning tools). The only honest approach is to look at the
    container, say what it is, and parse only what can genuinely be parsed.
    """
    inner = detection.signature_kind or det.signature_kind(data)

    base = {
        "ok": True,
        "declared_as": "DRP project file",
        "container_format": det.LABELS.get(inner, "unrecognised binary container"),
        "executed": False,
        "note": "The file was inspected as data. Nothing inside it was executed.",
    }

    if inner in (det.ZIP, det.DOCX, det.XLSX, det.PPTX):
        listing = inspect_archive(data)
        base["archive"] = listing
        if listing.get("ok"):
            names = [e["name"] for e in listing.get("entries", [])]
            readable = [n for n in names if n.lower().endswith(
                (".xml", ".json", ".csv", ".txt", ".md"))]
            base["parseable"] = bool(readable)
            base["readable_entries"] = readable[:40]
            base["summary"] = (
                f"This .drp file is a ZIP container holding {listing['entry_count']} "
                f"entries. I can list its structure. "
                + (f"{len(readable)} entry name(s) look like readable data files, but I "
                   "cannot map their internal schema to project fields with confidence. "
                   if readable else
                   "None of the entries are in a format whose project schema I can "
                   "interpret. ")
                + DRP_HANDOFF
            )
        else:
            base["parseable"] = False
            base["summary"] = (
                "This .drp file looks like a ZIP container but the index could not be "
                f"read. {DRP_HANDOFF}"
            )
        return base

    if inner in (det.JSON, det.XML, det.CSV, det.TSV, det.TXT, det.MD) or det._looks_like_text(data):
        text, encoding = _decode(data)
        base["container_format"] = "text-based"
        base["encoding"] = encoding
        base["character_count"] = len(text)
        base["preview"] = text[:2000]
        base["parseable"] = "partially"
        base["summary"] = (
            "This .drp file is text-based, so I can read its raw contents, but its "
            "internal schema is proprietary and I cannot reliably map it to project "
            f"fields such as progress, cost or dates. {DRP_HANDOFF}"
        )
        # Attempt a real structured parse only where the text genuinely is JSON/XML.
        if text.lstrip().startswith(("{", "[")):
            parsed = extract_json(data)
            if parsed.get("ok"):
                base["structured"] = {"format": "JSON", "structure": parsed["structure"]}
                base["summary"] = (
                    "This .drp file contains valid JSON. I have read its structure and can "
                    "answer questions about the values actually present in it, but I cannot "
                    "assume its fields mean the same thing as PAIMANA's project fields."
                )
                base["parseable"] = True
                base["text"] = parsed["text"]
        elif text.lstrip().startswith("<"):
            parsed = extract_xml(data)
            if parsed.get("ok"):
                base["structured"] = {"format": "XML",
                                      "elements": parsed["element_counts"],
                                      "root": parsed["root_tag"]}
                base["summary"] = (
                    "This .drp file contains valid XML. I have read its element structure "
                    "and can answer questions about the values present, but I cannot assume "
                    "its fields map onto PAIMANA's project fields."
                )
                base["parseable"] = True
                base["text"] = parsed["text"]
        return base

    base["parseable"] = False
    base["summary"] = (
        "I detected this as a .drp project file, but the current analyser cannot reliably "
        f"extract its internal project information — it is a binary format with no "
        f"recognisable structure I can decode. {DRP_HANDOFF}"
    )
    return base


def unsupported(detection: det.Detection) -> dict:
    """Honest response for a format we can name but not parse."""
    label = detection.label
    extra = ""
    if detection.kind == det.XLS:
        extra = " Save it as .xlsx or export the sheet as CSV and I can analyse it fully."
    elif detection.kind == det.OLE:
        extra = (" Save it in the modern Office format (.docx, .xlsx, .pptx) and I can "
                 "analyse it fully.")
    return _fail(
        f"I can identify this file as a {label}, but this format isn't currently "
        f"supported for direct analysis.{extra}",
        recovery=extra.strip() or "Convert it to a supported format and upload again.",
        identified_as=detection.kind,
    )
