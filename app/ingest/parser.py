"""PDF -> structured Markdown. Headings are what the skeleton tree is built from, so the
parser's one job is to preserve them (plus tables and figure captions).

Fallback chain: Docling (best structure, tables, figures) -> PyMuPDF4LLM (fast) -> error.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class ParsedDoc:
    markdown: str
    parser: str
    warnings: list[str] = field(default_factory=list)


def parse_pdf(path: str) -> ParsedDoc:
    try:
        from docling.document_converter import DocumentConverter  # type: ignore

        result = DocumentConverter().convert(path)
        return ParsedDoc(result.document.export_to_markdown(), "docling")
    except ImportError:
        log.info("docling not installed; falling back")
    except Exception as e:  # layout model failure on a weird PDF
        log.warning("docling failed on %s: %s", path, e)

    try:
        import pymupdf4llm  # type: ignore

        return ParsedDoc(pymupdf4llm.to_markdown(path), "pymupdf4llm",
                         ["fallback parser: tables/figures may be degraded"])
    except ImportError as e:
        raise RuntimeError("install docling or pymupdf4llm to parse PDFs") from e
