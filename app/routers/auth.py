from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
from sqlmodel import Session, select

from app.models import User
from app.services.auth import (
    SESSION_COOKIE_NAME,
    create_session_cookie,
    get_current_user,
    hash_password,
    verify_password,
)
from app.services.db import get_session

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


class Credentials(BaseModel):
    username: str
    password: str


def _set_session_cookie(response: Response, user_id: str) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=create_session_cookie(user_id),
        httponly=True,
        samesite="lax",
    )


@router.post("/register")
async def register(payload: Credentials, response: Response, session: Session = Depends(get_session)):
    username = payload.username.strip()
    if not username or not payload.password:
        raise HTTPException(400, "Username and password are required")

    existing = session.exec(select(User).where(User.username == username)).first()
    if existing is not None:
        raise HTTPException(400, "Username already taken")

    user = User(username=username, password_hash=hash_password(payload.password))
    session.add(user)
    session.commit()
    session.refresh(user)

    _set_session_cookie(response, user.id)
    return {"username": user.username}


@router.post("/login")
async def login(payload: Credentials, response: Response, session: Session = Depends(get_session)):
    user = session.exec(select(User).where(User.username == payload.username.strip())).first()
    if user is None or not verify_password(payload.password, user.password_hash):
        raise HTTPException(401, "Invalid username or password")

    _set_session_cookie(response, user.id)
    return {"username": user.username}


@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}


@router.get("/me")
async def me(current_user: User = Depends(get_current_user)):
    return {"username": current_user.username}
