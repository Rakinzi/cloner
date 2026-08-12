from fastapi import Depends, HTTPException, Request
from passlib.context import CryptContext
from itsdangerous import BadSignature, URLSafeTimedSerializer
from sqlmodel import Session

from app.config import settings
from app.models import User
from app.services.db import get_session

SESSION_COOKIE_NAME = "session"
_SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 30  # 30 days

_pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_serializer = URLSafeTimedSerializer(settings.session_secret, salt="cloner-session")


def hash_password(password: str) -> str:
    return _pwd_context.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    return _pwd_context.verify(password, password_hash)


def create_session_cookie(user_id: str) -> str:
    return _serializer.dumps({"user_id": user_id})


def read_session_cookie(token: str) -> str | None:
    try:
        data = _serializer.loads(token, max_age=_SESSION_MAX_AGE_SECONDS)
    except BadSignature:
        return None
    return data.get("user_id")


async def get_current_user(request: Request, session: Session = Depends(get_session)) -> User:
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        raise HTTPException(401, "Not authenticated")
    user_id = read_session_cookie(token)
    if user_id is None:
        raise HTTPException(401, "Not authenticated")
    user = session.get(User, user_id)
    if user is None:
        raise HTTPException(401, "Not authenticated")
    return user
