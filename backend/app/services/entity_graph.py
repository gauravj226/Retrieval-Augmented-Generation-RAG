import re
from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy.orm import Session

from ..models.models import EntityRelationship, InvoiceMetadata

_CROSS_REF_PATTERNS = [
    re.compile(
        r"(?:as per|refer to|in accordance with|see also|outlined in)\s+(?:the\s+)?([A-Z][a-zA-Z\s]{3,60}(?:Policy|Procedure|Guideline|Framework))",
        flags=re.IGNORECASE,
    ),
    re.compile(
        r"([A-Z][a-zA-Z\s]{3,60}(?:Policy|Procedure))\s+(?:sets out|provides|defines|governs)",
        flags=re.IGNORECASE,
    ),
]


def _clean_entity_name(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", (value or "").strip())
    return cleaned[:120]


def _safe_source_entity(filename: str) -> str:
    stem = Path(filename or "").stem.strip()
    return stem[:120] if stem else "unknown_document"


def _insert_relationship(
    db: Session,
    *,
    kb_id: int,
    source_entity: str,
    relation_type: str,
    target_entity: str,
    source_doc: str,
    target_doc: Optional[str] = None,
    confidence: float = 1.0,
) -> bool:
    existing = (
        db.query(EntityRelationship)
        .filter(
            EntityRelationship.kb_id == int(kb_id),
            EntityRelationship.source_entity == source_entity,
            EntityRelationship.relation_type == relation_type,
            EntityRelationship.target_entity == target_entity,
            EntityRelationship.source_doc == source_doc,
            EntityRelationship.target_doc == target_doc,
        )
        .first()
    )
    if existing:
        return False

    db.add(
        EntityRelationship(
            kb_id=int(kb_id),
            source_entity=source_entity,
            relation_type=relation_type,
            target_entity=target_entity,
            source_doc=source_doc,
            target_doc=target_doc,
            confidence=float(confidence),
        )
    )
    return True


def extract_cross_references(
    *,
    filename: str,
    content: str,
    kb_id: int,
    db: Session,
) -> int:
    source_doc = str(filename or "")
    source_entity = _safe_source_entity(source_doc)
    text = str(content or "")
    if not text.strip():
        return 0

    created = 0
    for pattern in _CROSS_REF_PATTERNS:
        for match in pattern.finditer(text):
            referenced = _clean_entity_name(match.group(1))
            if not referenced:
                continue
            if _insert_relationship(
                db,
                kb_id=int(kb_id),
                source_entity=source_entity,
                relation_type="references",
                target_entity=referenced,
                source_doc=source_doc,
                confidence=0.9,
            ):
                created += 1
    if created:
        db.commit()
    return created


def extract_invoice_relationships(
    *,
    kb_id: int,
    source_doc: str,
    invoice: InvoiceMetadata,
    db: Session,
) -> int:
    created = 0
    source_entity = _safe_source_entity(source_doc)
    if invoice.vendor_name:
        if _insert_relationship(
            db,
            kb_id=int(kb_id),
            source_entity=source_entity,
            relation_type="has_vendor",
            target_entity=_clean_entity_name(str(invoice.vendor_name)),
            source_doc=source_doc,
            confidence=1.0,
        ):
            created += 1

    if invoice.invoice_number:
        if _insert_relationship(
            db,
            kb_id=int(kb_id),
            source_entity=source_entity,
            relation_type="has_invoice_number",
            target_entity=_clean_entity_name(str(invoice.invoice_number)),
            source_doc=source_doc,
            confidence=1.0,
        ):
            created += 1

    if invoice.currency:
        if _insert_relationship(
            db,
            kb_id=int(kb_id),
            source_entity=source_entity,
            relation_type="uses_currency",
            target_entity=_clean_entity_name(str(invoice.currency)),
            source_doc=source_doc,
            confidence=0.8,
        ):
            created += 1

    if created:
        db.commit()
    return created


def match_entities_for_query(question: str) -> Iterable[str]:
    src = str(question or "")
    out = set()

    for policy in re.findall(
        r"\b[A-Z][A-Za-z0-9/&\-\s]{3,60}(?:Policy|Procedure|Guideline|Framework)\b",
        src,
    ):
        out.add(_clean_entity_name(policy))

    for token in re.findall(r"[a-zA-Z0-9_\-]{4,}", src.lower()):
        out.add(token)

    for drive in re.findall(r"\b([a-z])\s*:?\s*drive\b", src.lower()):
        out.add(f"{drive} drive")

    return out
