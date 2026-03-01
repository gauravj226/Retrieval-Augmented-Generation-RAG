from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.models import Group, GroupKBPermission, KnowledgeBase, User, UserGroupMapping
from ..routers.auth import get_admin_user
from ..schemas.schemas import GroupCreate, GroupKBPermissionSet, GroupResponse, GroupUpdate
from ..services.audit import audit_event

router = APIRouter(prefix="/admin/groups", tags=["Groups"])


def _build_response(group: Group, db: Session) -> GroupResponse:
    kb_perms = (
        db.query(GroupKBPermission)
        .filter(GroupKBPermission.group_id == group.id)
        .all()
    )
    return GroupResponse(
        id=group.id,
        name=group.name,
        description=group.description,
        created_at=group.created_at,
        member_count=len(group.members),
        kb_permissions=[
            {"kb_id": p.kb_id, "permission": p.permission} for p in kb_perms
        ],
    )


@router.get("", response_model=List[GroupResponse])
async def list_groups(
    db: Session = Depends(get_db), admin: User = Depends(get_admin_user)
):
    groups = db.query(Group).all()
    return [_build_response(g, db) for g in groups]


@router.post("", response_model=GroupResponse)
async def create_group(
    data: GroupCreate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    if db.query(Group).filter(Group.name == data.name).first():
        raise HTTPException(status_code=400, detail="Group name already exists")
    group = Group(name=data.name, description=data.description)
    db.add(group)
    db.commit()
    db.refresh(group)
    audit_event(
        "group.created",
        actor=admin,
        target_type="group",
        target_id=group.id,
        details={"name": group.name},
    )
    return _build_response(group, db)


@router.put("/{group_id}", response_model=GroupResponse)
async def update_group(
    group_id: int,
    data: GroupUpdate,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Not found")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(group, field, value)
    db.commit()
    db.refresh(group)
    audit_event(
        "group.updated",
        actor=admin,
        target_type="group",
        target_id=group.id,
        details={"name": group.name},
    )
    return _build_response(group, db)


@router.delete("/{group_id}")
async def delete_group(
    group_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Not found")
    group_name = group.name
    db.delete(group)
    db.commit()
    audit_event(
        "group.deleted",
        actor=admin,
        target_type="group",
        target_id=group_id,
        details={"name": group_name},
    )
    return {"message": "Deleted"}


# ── Members ───────────────────────────────────────────────────────────────────

@router.put("/{group_id}/members")
async def set_group_members(
    group_id: int,
    user_ids: List[int],
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    group = db.query(Group).filter(Group.id == group_id).first()
    if not group:
        raise HTTPException(status_code=404, detail="Not found")

    # Remove existing mappings
    db.query(UserGroupMapping).filter(UserGroupMapping.group_id == group_id).delete()

    # Add new mappings
    for uid in user_ids:
        if db.query(User).filter(User.id == uid).first():
            db.add(UserGroupMapping(user_id=uid, group_id=group_id))

    db.commit()
    audit_event(
        "group.members_updated",
        actor=admin,
        target_type="group",
        target_id=group_id,
        details={"member_count": len(user_ids)},
    )
    return {"message": "Members updated", "count": len(user_ids)}


@router.get("/{group_id}/members")
async def get_group_members(
    group_id: int,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    mappings = (
        db.query(UserGroupMapping)
        .filter(UserGroupMapping.group_id == group_id)
        .all()
    )
    return [m.user_id for m in mappings]


# ── KB Permissions ────────────────────────────────────────────────────────────

@router.put("/{group_id}/permissions")
async def set_group_kb_permissions(
    group_id: int,
    permissions: List[GroupKBPermissionSet],
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
):
    if not db.query(Group).filter(Group.id == group_id).first():
        raise HTTPException(status_code=404, detail="Group not found")

    db.query(GroupKBPermission).filter(GroupKBPermission.group_id == group_id).delete()

    for perm in permissions:
        if not db.query(KnowledgeBase).filter(KnowledgeBase.id == perm.kb_id).first():
            continue
        db.add(GroupKBPermission(
            group_id=group_id, kb_id=perm.kb_id, permission=perm.permission
        ))

    db.commit()
    audit_event(
        "group.permissions_updated",
        actor=admin,
        target_type="group",
        target_id=group_id,
        details={"permissions_count": len(permissions)},
    )
    return {"message": "Permissions updated"}
