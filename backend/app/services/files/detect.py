"""File type detection.

The extension is a hint, never a decision. Every upload is triangulated across
three independent signals — declared extension, declared MIME type, and the
actual leading bytes — and the byte signature always wins. A ``.pdf`` that does
not begin with ``%PDF-`` is reported as a mismatch, not parsed as a PDF.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from io import BytesIO

# --- canonical kinds --------------------------------------------------------
PDF = "PDF"
DOCX = "DOCX"
XLSX = "XLSX"
XLS = "XLS"
PPTX = "PPTX"
CSV = "CSV"
TSV = "TSV"
JSON = "JSON"
XML = "XML"
TXT = "TXT"
MD = "MD"
PNG = "PNG"
JPEG = "JPEG"
WEBP = "WEBP"
GIF = "GIF"
BMP = "BMP"
ZIP = "ZIP"
DRP = "DRP"
OLE = "OLE"          # legacy Microsoft compound document (.doc/.xls/.ppt/.msg)
UNKNOWN = "UNKNOWN"

IMAGE_KINDS = {PNG, JPEG, WEBP, GIF, BMP}
TEXT_KINDS = {CSV, TSV, JSON, XML, TXT, MD}
OFFICE_OOXML = {DOCX, XLSX, PPTX}

#: Kinds the analyser can genuinely extract content from.
SUPPORTED = {
    PDF, DOCX, XLSX, PPTX, CSV, TSV, JSON, XML, TXT, MD,
    PNG, JPEG, WEBP, GIF, BMP, ZIP,
}

#: Kinds we can name confidently but cannot parse. Reported honestly.
IDENTIFIED_NOT_PARSEABLE = {XLS, OLE, DRP}

EXTENSION_MAP = {
    ".pdf": PDF, ".docx": DOCX, ".xlsx": XLSX, ".xlsm": XLSX, ".xltx": XLSX,
    ".xls": XLS, ".pptx": PPTX, ".csv": CSV, ".tsv": TSV, ".json": JSON,
    ".xml": XML, ".txt": TXT, ".text": TXT, ".log": TXT, ".md": MD,
    ".markdown": MD, ".png": PNG, ".jpg": JPEG, ".jpeg": JPEG, ".webp": WEBP,
    ".gif": GIF, ".bmp": BMP, ".zip": ZIP, ".drp": DRP,
    ".doc": OLE, ".ppt": OLE, ".msg": OLE,
}

LABELS = {
    PDF: "PDF document", DOCX: "Word document", XLSX: "Excel workbook",
    XLS: "Legacy Excel workbook (BIFF)", PPTX: "PowerPoint presentation",
    CSV: "CSV data file", TSV: "Tab-separated data file", JSON: "JSON data file",
    XML: "XML document", TXT: "Plain text file", MD: "Markdown document",
    PNG: "PNG image", JPEG: "JPEG image", WEBP: "WebP image", GIF: "GIF image",
    BMP: "BMP image", ZIP: "ZIP archive", DRP: "DRP project file",
    OLE: "Legacy Microsoft Office document", UNKNOWN: "Unrecognised file",
}


@dataclass
class Detection:
    kind: str = UNKNOWN
    label: str = "Unrecognised file"
    extension: str = ""
    declared_mime: str | None = None
    signature_kind: str | None = None
    mismatch: bool = False
    notes: list[str] = field(default_factory=list)
    is_empty: bool = False

    @property
    def parseable(self) -> bool:
        return self.kind in SUPPORTED

    @property
    def identified_only(self) -> bool:
        return self.kind in IDENTIFIED_NOT_PARSEABLE

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "extension": self.extension,
            "declared_mime": self.declared_mime,
            "signature_kind": self.signature_kind,
            "extension_signature_mismatch": self.mismatch,
            "parseable": self.parseable,
            "notes": self.notes,
        }


def _ooxml_kind(data: bytes) -> str:
    """A DOCX/XLSX/PPTX is a ZIP. Look inside to tell which."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as zf:
            names = set(zf.namelist())
    except (zipfile.BadZipFile, OSError):
        return ZIP
    if any(n.startswith("word/") for n in names):
        return DOCX
    if any(n.startswith("xl/") for n in names):
        return XLSX
    if any(n.startswith("ppt/") for n in names):
        return PPTX
    return ZIP


def _looks_like_text(data: bytes) -> bool:
    sample = data[:4096]
    if b"\x00" in sample:
        return False
    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        try:
            sample.decode("latin-1")
            # latin-1 decodes anything; require mostly printable
            printable = sum(1 for b in sample if 9 <= b <= 13 or 32 <= b < 127 or b >= 160)
            return printable / max(len(sample), 1) > 0.9
        except UnicodeDecodeError:
            return False


def _text_kind(data: bytes, hint: str) -> str:
    try:
        head = data[:8192].decode("utf-8", errors="replace").lstrip()
    except Exception:  # pragma: no cover — decode with replace cannot raise
        return TXT
    if head.startswith(("{", "[")):
        return JSON
    if head.startswith("<?xml") or (head.startswith("<") and ">" in head[:400]):
        return XML
    if hint in {CSV, TSV, JSON, XML, MD}:
        return hint
    first = head.splitlines()[0] if head.splitlines() else ""
    if first.count("\t") >= 2:
        return TSV
    if first.count(",") >= 2:
        return CSV
    return TXT


def signature_kind(data: bytes) -> str:
    """Identify a file purely from its leading bytes."""
    if not data:
        return UNKNOWN
    if data.startswith(b"%PDF-"):
        return PDF
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return PNG
    if data.startswith(b"\xff\xd8\xff"):
        return JPEG
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return WEBP
    if data.startswith((b"GIF87a", b"GIF89a")):
        return GIF
    if data.startswith(b"BM"):
        return BMP
    if data.startswith(b"PK\x03\x04"):
        return _ooxml_kind(data)
    if data.startswith(b"PK") and data[2:4] in (b"\x05\x06", b"\x07\x08"):
        return ZIP
    if data.startswith(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"):
        return OLE               # covers legacy .xls/.doc/.ppt
    if data.startswith(b"SQLite format 3\x00"):
        return UNKNOWN
    if _looks_like_text(data):
        return "TEXTLIKE"
    return UNKNOWN


def detect(filename: str, data: bytes, declared_mime: str | None = None) -> Detection:
    """Triangulate extension, declared MIME and magic bytes.

    The signature is authoritative. Where the extension disagrees the detection
    records a mismatch so the caller can surface it to the user.
    """
    ext = ""
    if "." in (filename or ""):
        ext = "." + filename.rsplit(".", 1)[-1].lower()
    ext_kind = EXTENSION_MAP.get(ext, UNKNOWN)

    det = Detection(extension=ext, declared_mime=declared_mime)

    if not data:
        det.is_empty = True
        det.kind = ext_kind
        det.label = LABELS.get(ext_kind, LABELS[UNKNOWN])
        det.notes.append("The file is empty (0 bytes).")
        return det

    sig = signature_kind(data)
    det.signature_kind = None if sig == "TEXTLIKE" else sig

    if sig == "TEXTLIKE":
        # Text files have no magic bytes; classify by content shape + extension.
        resolved = _text_kind(data, ext_kind)
        # A .drp that happens to be text is still a .drp — do not silently
        # promote it to XML/JSON without saying so.
        if ext_kind == DRP:
            det.kind = DRP
            det.notes.append(
                f"Contents look like {LABELS.get(resolved, 'text')}, which may be "
                "readable. Handled by the DRP analyser."
            )
            det.label = LABELS[DRP]
            return det
        det.kind = resolved
        if ext_kind in SUPPORTED and ext_kind not in TEXT_KINDS:
            det.mismatch = True
            det.notes.append(
                f"The extension says {LABELS.get(ext_kind, ext)} but the contents are "
                f"plain text. Treating it as {LABELS.get(resolved, 'text')}."
            )
    elif sig == UNKNOWN:
        det.kind = ext_kind if ext_kind in IDENTIFIED_NOT_PARSEABLE else UNKNOWN
        if ext_kind == DRP:
            det.kind = DRP
        elif ext_kind != UNKNOWN:
            det.mismatch = True
            det.notes.append(
                f"The extension says {LABELS.get(ext_kind, ext)}, but the file's leading "
                "bytes do not match that format. It may be renamed or corrupted."
            )
    else:
        det.kind = sig
        if ext_kind != UNKNOWN and ext_kind != sig:
            # ZIP-vs-OOXML and CSV-vs-TSV are not real mismatches worth alarming over.
            benign = {ext_kind, sig} <= (OFFICE_OOXML | {ZIP}) or {ext_kind, sig} <= TEXT_KINDS
            if ext_kind == DRP:
                det.kind = DRP
                det.notes.append(
                    f"The .drp file is internally a {LABELS.get(sig, sig)}. "
                    "The DRP analyser will inspect it without executing it."
                )
                det.signature_kind = sig
            elif not benign:
                det.mismatch = True
                det.notes.append(
                    f"Extension/content mismatch: named {LABELS.get(ext_kind, ext)} but the "
                    f"contents are a {LABELS.get(sig, sig)}. Analysing it as the latter."
                )

    det.label = LABELS.get(det.kind, LABELS[UNKNOWN])
    return det
