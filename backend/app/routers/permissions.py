"""
Reusable permission dependency helpers for group-based access control.
"""
from typing import List, Optional
from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..models.models import GroupKBPermission, UserGroupMapping

PERM_LEVELS = {"read": 0, "manage": 1}


def _get_user_group_ids(user_id: int, db: Session) -> List[int]:
    return [
        m.group_id
        for m in db.query(UserGroupMapping)
        .filter(UserGroupMapping.user_id == user_id)
        .all()
    ]


def get_kb_permission(user, kb_id: int, db: Session) -> Optional[str]:
    """Returns 'manage', 'read', or None for a user on a specific KB."""
    if user.is_admin:
        return "manage"

    group_ids = _get_user_group_ids(user.id, db)
    if not group_ids:
        return None

    perms = (
        db.query(GroupKBPermission)
        .filter(
            GroupKBPermission.group_id.in_(group_ids),
            GroupKBPermission.kb_id == kb_id,
        )
        .all()
    )

    if not perms:
        return None
    if any(p.permission == "manage" for p in perms):
        return "manage"
    return "read"


def require_kb_access(user, kb_id: int, min_perm: str, db: Session):
    """Raises 403 if the user doesn't meet the required permission level."""
    perm = get_kb_permission(user, kb_id, db)
    if perm is None:
        raise HTTPException(status_code=403, detail="No access to this knowledge base")
    if PERM_LEVELS.get(perm, -1) < PERM_LEVELS.get(min_perm, 99):
        raise HTTPException(status_code=403, detail=f"Requires '{min_perm}' permission")


def get_accessible_kb_ids(user, db: Session) -> Optional[List[int]]:
    """Returns KB IDs accessible to the user, or None (meaning all) for admins."""
    if user.is_admin:
        return None

    group_ids = _get_user_group_ids(user.id, db)
    if not group_ids:
        return []

    rows = (
        db.query(GroupKBPermission.kb_id)
        .filter(GroupKBPermission.group_id.in_(group_ids))
        .distinct()
        .all()
    )
    return [r.kb_id for r in rows]
