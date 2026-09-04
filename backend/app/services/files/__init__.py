"""Uploaded-file analysis.

Uploads are untrusted data. Nothing here executes, evaluates, or resolves
anything found inside a file: archives are listed rather than extracted, XML is
parsed with entity expansion disabled, and text that reads like an instruction
to a model is neutralised before it goes anywhere near a prompt.

Public surface:

    analyse(filename, data, mime) -> dict     one normalised analysis
    to_prompt_block(analysis)     -> str      compact, delimited model input
    SUPPORTED_EXTENSIONS                      what the UI should accept
"""
from __future__ import annotations

from .analysis import ANALYSED, FAILED, IDENTIFIED, analyse, read_rows, to_prompt_block
# NOTE: the `detect` *function* is deliberately not re-exported here. Doing so
# rebinds `app.services.files.detect` from the submodule to the function, so
# `from app.services.files import detect` silently yields a function where a
# module was expected. Import it from `.detect` directly instead.
from .detect import Detection, LABELS
from .tabular import GOOD, INSUFFICIENT, NEEDS_ATTENTION, POOR, assess_quality

#: Extensions offered in the file picker. Presence here means "we will try and
#: tell you honestly what happened", not "guaranteed full analysis".
SUPPORTED_EXTENSIONS = [
    ".pdf", ".docx", ".txt", ".md", ".csv", ".tsv", ".xlsx", ".xlsm", ".json",
    ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".xml", ".zip",
    ".drp", ".xls",
]

FULLY_ANALYSED_EXTENSIONS = [
    ".pdf", ".docx", ".txt", ".md", ".csv", ".tsv", ".xlsx", ".xlsm", ".json",
    ".pptx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".xml",
]

__all__ = [
    "ANALYSED", "IDENTIFIED", "FAILED", "analyse", "read_rows", "to_prompt_block",
    "Detection", "LABELS", "assess_quality",
    "GOOD", "NEEDS_ATTENTION", "POOR", "INSUFFICIENT",
    "SUPPORTED_EXTENSIONS", "FULLY_ANALYSED_EXTENSIONS",
]
