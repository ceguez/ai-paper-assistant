"""PDF -> structured Markdown with page markers.

Headings are what the skeleton tree is built from, so the parser's main job is to keep
them (plus tables and figure captions). It also emits `<!-- page:N -->` before each page's
text so every section knows which PDF page to open in the UI.

Order for PA_PARSER=auto: Docling if installed (best tables and layout; first run downloads
its layout models and is slower), else PyMuPDF4LLM (fast, CPU only).
PyMuPDF is AGPL-licensed: fine for personal and internal use, check before redistributing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ParsedDoc:
    markdown: str
    parser: str
    page_count: int = 0
    warnings: list[str] = field(default_factory=list)


def page_marker(n: int) -> str:
    return f"<!-- page:{n} -->"


def _docling(path: str) -> ParsedDoc:
    from docling.document_converter import DocumentConverter  # type: ignore

    doc = DocumentConverter().convert(path).document
    n_pages = len(getattr(doc, "pages", {}) or {})
    parts: list[str] = []
    try:
        for p in range(1, n_pages + 1):
            parts += [page_marker(p), doc.export_to_markdown(page_no=p)]
        md = "\n\n".join(parts)
    except TypeError:  # older docling-core without page_no
        md = doc.export_to_markdown()
    return ParsedDoc(md, "docling", n_pages)


def _pymupdf4llm(path: str) -> ParsedDoc:
    import pymupdf4llm  # type: ignore

    chunks = pymupdf4llm.to_markdown(path, page_chunks=True)
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts += [page_marker(i), chunk.get("text", "")]
    return ParsedDoc("\n\n".join(parts), "pymupdf4llm", len(chunks),
                     ["layout-light parser: complex tables may be degraded"])


def parse_pdf(path: str, preference: str = "auto") -> ParsedDoc:
    order = {"docling": [_docling], "pymupdf4llm": [_pymupdf4llm]}.get(preference, [_docling, _pymupdf4llm])
    errors = []
    for fn in order:
        try:
            parsed = fn(path)
            if parsed.markdown.strip():
                return parsed
            errors.append(f"{fn.__name__}: empty output (scanned PDF without OCR?)")
        except ImportError:
            errors.append(f"{fn.__name__}: not installed")
        except Exception as e:  # a parser crashing on one odd PDF must not kill the worker
            log.warning("%s failed on %s: %s", fn.__name__, path, e)
            errors.append(f"{fn.__name__}: {e}")
    raise RuntimeError("could not parse PDF: " + "; ".join(errors))
