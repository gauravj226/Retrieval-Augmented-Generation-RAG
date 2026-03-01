"""
Document complexity classifier.
Returns one of:
  - "text"
  - "structured"
  - "visual"
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from ..config import settings

logger = logging.getLogger(__name__)

_ALWAYS_VISUAL = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}
_LIKELY_STRUCTURED = {".xlsx", ".xls", ".csv"}
_SCAN_TEXT_RATIO_THRESHOLD = 0.4
_VISUAL_MAX_PDF_PAGES = int(os.getenv("VISUAL_MAX_PDF_PAGES", str(settings.VLM_MAX_PAGES)))
_VISUAL_MAX_PDF_MB = int(os.getenv("VISUAL_MAX_PDF_MB", "4"))


def classify_document(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext in _ALWAYS_VISUAL:
        logger.info("[classifier] %s -> visual (image file)", path.name)
        return "visual"

    if ext in _LIKELY_STRUCTURED:
        logger.info("[classifier] %s -> structured (spreadsheet)", path.name)
        return "structured"

    if ext == ".pdf":
        return _classify_pdf(file_path)

    if ext in (".docx", ".pptx"):
        return _classify_office(file_path)

    return "text"


def _classify_pdf(file_path: str) -> str:
    try:
        import pdfplumber

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            if total_pages == 0:
                return "visual"

            text_pages = 0
            table_pages = 0
            sample_pages = min(total_pages, 6)
            for page in pdf.pages[:sample_pages]:
                text = page.extract_text() or ""
                text_pages += 1 if len(text.strip()) > 50 else 0
                table_pages += 1 if page.extract_tables() else 0

            text_ratio = text_pages / sample_pages
            table_ratio = table_pages / sample_pages
            file_mb = Path(file_path).stat().st_size / (1024 * 1024)

            if text_ratio < _SCAN_TEXT_RATIO_THRESHOLD:
                if total_pages > _VISUAL_MAX_PDF_PAGES or file_mb > _VISUAL_MAX_PDF_MB:
                    logger.info(
                        "[classifier] -> structured (scan too large for VLM, pages=%s, size=%.1fMB)",
                        total_pages,
                        file_mb,
                    )
                    return "structured"
                logger.info("[classifier] -> visual (scanned PDF, text_ratio=%.2f)", text_ratio)
                return "visual"

            if table_ratio >= 0.3:
                logger.info("[classifier] -> structured (table-heavy PDF, table_ratio=%.2f)", table_ratio)
                return "structured"

            logger.info("[classifier] -> text (standard PDF)")
            return "text"
    except Exception as e:
        logger.warning("[classifier] PDF classification failed: %s; defaulting to text", e)
        return "text"


def _classify_office(file_path: str) -> str:
    try:
        from docx import Document as DocxDoc

        doc = DocxDoc(file_path)
        total = len(doc.paragraphs) + 1
        tables = len(doc.tables)
        if tables / total > 0.1 or tables >= 3:
            return "structured"
        return "text"
    except Exception:
        return "text"

