"""
Docling-based layout-aware parser.
Used for: PDFs with tables, financial documents, DOCX with complex layouts.

Multi-vector strategy:
  - Embeds a SUMMARY of each chunk  → what ChromaDB searches
  - Stores RAW content in metadata  → what the LLM receives at generation time

This prevents lossy chunking from destroying table structure.
"""
from __future__ import annotations
import logging
import os
from pathlib import Path
from typing import List

from langchain.schema import Document

logger = logging.getLogger(__name__)


def _extract_key_items(text: str) -> str:
    """Extract key literals (paths/commands/URLs) so embeddings keep critical tokens."""
    import re

    items = []

    # UNC paths like \\server\share\folder
    unc = re.findall(r"\\\\[A-Za-z0-9._-]+\\[A-Za-z0-9$._\\-]+(?:\\[A-Za-z0-9$._\\-]+)*", text)
    items.extend(unc[:5])

    # net use commands
    commands = re.findall(r"net use\s+[A-Za-z]:\s+\S+", text, flags=re.IGNORECASE)
    items.extend(commands[:3])

    # URLs
    urls = re.findall(r"https?://\S+", text)
    items.extend(urls[:3])

    if not items:
        return ""
    return " Key items: " + " | ".join(dict.fromkeys(items))


def parse_with_docling(file_path: str, original_filename: str) -> List[Document]:
    """
    Parse a document with Docling and return LangChain Documents.
    Each Document has:
      page_content  = summary (embedded into ChromaDB)
      metadata.raw  = full original content (passed to LLM at generation)
    """
    try:
        from docling.document_converter import DocumentConverter
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr              = True   # OCR fallback for scanned regions
        pipeline_opts.do_table_structure  = True   # structural table parsing
        pipeline_opts.table_structure_options.do_cell_matching = True

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
            }
        )

        logger.info(f"[docling] Converting: {original_filename}")
        result   = converter.convert(file_path)
        doc_obj  = result.document

        chunks: List[Document] = []

        # ── Extract tables as structured Markdown ─────────────────────────────
        for i, table in enumerate(doc_obj.tables):
            try:
                df          = table.export_to_dataframe()
                md_table    = df.to_markdown(index=False)
                page_num    = table.prov[0].page_no if table.prov else "unknown"

                # Summary for embedding — describes what the table contains
                col_names   = list(df.columns)
                row_count   = len(df)
                summary     = (
                    f"Table {i+1} from '{original_filename}' (page {page_num}). "
                    f"Contains {row_count} rows with columns: {', '.join(str(c) for c in col_names)}. "
                    f"First row sample: {df.iloc[0].to_dict() if row_count > 0 else 'empty'}"
                )
                summary += _extract_key_items(md_table)

                chunks.append(Document(
                    page_content=summary,
                    metadata={
                        "source":    original_filename,
                        "type":      "table",
                        "page":      str(page_num),
                        "table_idx": i,
                        "raw":       md_table,
                        "pipeline":  "docling",
                    }
                ))
                logger.info(f"[docling] Table {i+1}: {row_count} rows × {len(col_names)} cols")
            except Exception as e:
                logger.warning(f"[docling] Table {i+1} export failed: {e}")

        # ── Extract text sections (paragraphs, headings) ──────────────────────
        full_md = doc_obj.export_to_markdown()

        # Split on section boundaries (## headings or page breaks)
        import re
        sections = re.split(r'\n(?=#{1,3} )', full_md)

        for j, section in enumerate(sections):
            section = section.strip()
            if not section or len(section) < 30:
                continue

            # Skip sections that are pure table content (already extracted above)
            if section.count('|') > section.count(' ') * 0.3:
                continue

            # Summary = first 300 chars of the section (heading + opening)
            summary = section[:300].replace('\n', ' ').strip()
            summary += _extract_key_items(section)

            chunks.append(Document(
                    page_content=summary,
                metadata={
                    "source":      original_filename,
                    "type":        "text_section",
                    "section_idx": j,
                    "raw":         section,
                    "pipeline":    "docling",
                }
            ))

        logger.info(f"[docling] {original_filename}: {len(chunks)} chunks ({len(doc_obj.tables)} tables)")
        return chunks

    except ImportError:
        logger.error("[docling] Docling not installed — run: pip install docling")
        raise
    except Exception as e:
        logger.exception(f"[docling] Parsing failed for {original_filename}: {e}")
        raise

