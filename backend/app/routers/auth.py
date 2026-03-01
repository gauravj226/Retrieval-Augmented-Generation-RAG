from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from ..config import settings
from ..database import get_db
from ..models.models import User
from ..schemas.schemas import Token, UserRegister, UserResponse
from ..services.audit import audit_event

router = APIRouter(prefix="/auth", tags=["Auth"])

pwd_context    = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme  = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_access_token(data: dict) -> str:
    expire  = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {**data, "exp": expire}
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        raw_sub = payload.get("sub")
        if raw_sub is None:
            raise credentials_exc
        user_id = int(raw_sub)
    except (JWTError, ValueError, TypeError):
        raise credentials_exc

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise credentials_exc
    return user



async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


async def get_admin_user(
    current_user: User = Depends(get_current_active_user),
) -> User:
    if not current_user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


@router.post("/register", response_model=UserResponse)
async def register(data: UserRegister, db: Session = Depends(get_db)):
    if db.query(User).filter(
        (User.email == data.email) | (User.username == data.username)
    ).first():
        raise HTTPException(status_code=400, detail="Email or username already registered")

    is_first_user = db.query(User.id).first() is None

    user = User(
        email=data.email,
        username=data.username,
        full_name=data.full_name,
        department=data.department,
        hashed_password=get_password_hash(data.password),
        is_admin=is_first_user,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    audit_event(
        "user.created",
        actor=user,
        target_type="user",
        target_id=user.id,
        details={"email": user.email, "is_admin": user.is_admin},
    )
    return {**user.__dict__, "group_ids": []}


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        audit_event(
            "auth.login_failed",
            actor_username=form_data.username,
            status="failed",
            details={"reason": "invalid_credentials"},
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
        )
    if not user.is_active:
        audit_event(
            "auth.login_failed",
            actor=user,
            status="failed",
            details={"reason": "inactive_account"},
        )
        raise HTTPException(status_code=400, detail="Account is disabled")

    token = create_access_token({"sub": str(user.id)})
    audit_event("auth.login_success", actor=user, target_type="user", target_id=user.id)
    group_ids = [m.group_id for m in user.group_memberships]
    return {
        "access_token": token,
        "token_type":   "bearer",
        "user": {
            "id":         user.id,
            "username":   user.username,
            "email":      user.email,
            "full_name":  user.full_name,
            "department": user.department,
            "is_admin":   user.is_admin,
            "is_active":  user.is_active,
            "group_ids":  group_ids,
        },
    }


@router.get("/me", response_model=UserResponse)
async def get_me(current_user: User = Depends(get_current_active_user), db: Session = Depends(get_db)):
    group_ids = [m.group_id for m in current_user.group_memberships]
    return {**current_user.__dict__, "group_ids": group_ids}

