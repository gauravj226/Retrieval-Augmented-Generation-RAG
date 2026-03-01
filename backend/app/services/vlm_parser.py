"""
VLM-based visual parser using Qwen2.5-VL via Ollama.
"""
from __future__ import annotations

import base64
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from langchain.schema import Document

from ..config import settings

logger = logging.getLogger(__name__)

VLM_MODEL = os.getenv("VLM_MODEL", "qwen2.5vl:3b")


def _image_to_base64(image) -> str:
    import io

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _call_vlm(b64_image: str, page_num: int, filename: str) -> str:
    import httpx

    prompt = (
        "You are a precise document analyser. Extract ALL information from this document page.\n"
        "Rules:\n"
        "- Extract all text exactly as written\n"
        "- Represent tables in Markdown format with | separators\n"
        "- Label figures/charts: [FIGURE: brief description]\n"
        "- Label signatures/stamps: [SIGNATURE] or [STAMP: text]\n"
        "- Preserve numbers, dates, and monetary values exactly\n"
        "- Output structured, clean text with no commentary\n\n"
        f"Document: {filename}, Page {page_num}"
    )

    payload = {
        "model": VLM_MODEL,
        "prompt": prompt,
        "images": [b64_image],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "num_predict": 2048,
        },
    }
    ollama_url = f"http://{settings.OLLAMA_HOST}:{settings.OLLAMA_PORT}/api/generate"

    try:
        resp = httpx.post(ollama_url, json=payload, timeout=float(settings.VLM_TIMEOUT_SEC))
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except httpx.TimeoutException:
        logger.warning("[vlm] Page %s timed out", page_num)
        return ""
    except Exception as e:
        logger.error("[vlm] Page %s failed: %s", page_num, e)
        return ""


def parse_with_vlm(file_path: str, original_filename: str) -> List[Document]:
    path = Path(file_path)
    ext = path.suffix.lower()
    chunks: List[Document] = []

    if ext == ".pdf":
        try:
            from pdf2image import convert_from_path

            dpi = int(settings.VLM_DPI)
            pages = convert_from_path(file_path, dpi=dpi)
            logger.info("[vlm] %s: %s pages at %s DPI", original_filename, len(pages), dpi)
        except Exception as e:
            logger.error("[vlm] pdf2image failed: %s", e)
            raise
    elif ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"):
        from PIL import Image

        pages = [Image.open(file_path)]
    else:
        raise ValueError(f"VLM pipeline does not support extension: {ext}")

    max_pages = int(settings.VLM_MAX_PAGES)
    original_page_count = len(pages)
    pages = pages[:max_pages]
    if original_page_count > max_pages:
        logger.warning("[vlm] Truncated from %s to %s pages", original_page_count, max_pages)

    page_inputs = [(i, _image_to_base64(page_img)) for i, page_img in enumerate(pages, start=1)]

    max_workers = max(1, int(settings.VLM_CONCURRENCY))
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(_call_vlm, b64, page_num, original_filename): page_num
            for page_num, b64 in page_inputs
        }
        for fut in as_completed(futures):
            page_num = futures[fut]
            try:
                results[page_num] = fut.result()
            except Exception as e:
                logger.warning("[vlm] Page %s worker failed: %s", page_num, e)
                results[page_num] = ""

    for page_num in sorted(results.keys()):
        raw = results[page_num]
        if not raw:
            continue

        summary = f"Page {page_num} of '{original_filename}': " + raw[:400].replace("\n", " ")
        chunks.append(
            Document(
                page_content=summary,
                metadata={
                    "source": original_filename,
                    "type": "vlm_page",
                    "page": str(page_num),
                    "raw": raw,
                    "pipeline": "vlm",
                    "model": VLM_MODEL,
                },
            )
        )

    logger.info("[vlm] %s: %s page chunks extracted", original_filename, len(chunks))
    return chunks

