"""
KB Manifest — lightweight index of documents built at ingest time.
Powers the introspect node to answer meta-queries like "what topics can you help with?".
"""
import json
import os
import re
from pathlib import Path
from typing import Dict, List

MANIFEST_DIR = Path(os.getenv("MANIFEST_DIR", "/app/data/manifests"))


def _extract_headings(text: str) -> List[str]:
    headings = re.findall(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
    if not headings:
        headings = re.findall(r"^([A-Z][A-Z\s]{4,50})$", text, re.MULTILINE)
    return [h.strip() for h in headings[:10]]


def _clean_filename(filename: str) -> str:
    name = re.sub(r"-\d{6}-\d{6}\..*$", "", filename)
    name = re.sub(r"\.(pdf|docx?|xlsx?|pptx?|csv|txt|md)$", "", name, flags=re.I)
    return name.replace("-", " ").replace("_", " ").strip()


def upsert_manifest(kb_id: int, filename: str, text: str, pipeline: str) -> None:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    path = MANIFEST_DIR / f"kb_{kb_id}.json"
    manifest: Dict = {}
    if path.exists():
        try:
            manifest = json.loads(path.read_text())
        except Exception:
            manifest = {}
    manifest[filename] = {
        "filename": filename,
        "display_name": _clean_filename(filename),
        "pipeline": pipeline,
        "headings": _extract_headings(text),
        "summary": " ".join(text.split()[:60]),
    }
    path.write_text(json.dumps(manifest, indent=2))


def load_manifest(kb_id: int) -> Dict:
    path = MANIFEST_DIR / f"kb_{kb_id}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def delete_from_manifest(kb_id: int, filename: str) -> None:
    path = MANIFEST_DIR / f"kb_{kb_id}.json"
    if not path.exists():
        return
    try:
        manifest = json.loads(path.read_text())
        manifest.pop(filename, None)
        path.write_text(json.dumps(manifest, indent=2))
    except Exception:
        pass
