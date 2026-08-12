# Portal: Auth + Multi-page Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add username/password auth with per-user data scoping, and
replace the single `index.html` page with four real pages (Login,
Generate, Voices, History) sharing a common nav and JS helpers.

**Architecture:** A new `User` table plus a required `user_id` FK on
`Voice` and `Generation`. Sessions are a signed cookie (`itsdangerous`)
carrying the user id — no server-side session table. A
`get_current_user` FastAPI dependency gates every voice/generation route
and scopes every query. Four static HTML pages replace `index.html`,
sharing `common.js` for auth-check, nav rendering, and fetch-with-cookie
helpers.

**Tech Stack:** FastAPI, `sqlmodel` (existing), `passlib[bcrypt]` (new),
`itsdangerous` (new), plain HTML/JS (no framework, no build step,
matching the existing `app/static/index.html` style).

## Global Constraints

- No OAuth, no password reset/email flows, no admin roles, no login rate
  limiting — spec: "Explicitly out of scope."
- No data migration — existing rows in `storage/db.sqlite3` are dropped;
  `user_id` is a required (non-nullable) FK from the start, not backfilled.
- No new frontend framework, no build step — static HTML + shared vanilla
  JS only.
- Session cookie is `httponly`, `samesite=lax`, NOT `secure` by default
  (plain HTTP on a LAN/GPU box) — spec explicitly calls this out as a
  one-line change if ever put behind TLS, not an oversight to fix now.
- Cross-user access to another user's voice/generation returns 404, never
  403 — spec: "avoids confirming the resource exists under a different
  account."
- Login failure (bad username or bad password) returns 401 with a message
  that does not reveal which field was wrong.
- `index.html` is deleted, not kept as an alternate view — fully
  superseded by the four new pages.
- No voice rename in this iteration — delete + re-upload only.

---

## File Structure

- Modify `app/models.py` — add `User` table; add `user_id` to `Voice` and
  `Generation`.
- Create `app/services/auth.py` — password hashing, session cookie
  sign/verify, `get_current_user` dependency.
- Create `app/routers/auth.py` — `/api/v1/auth/register`, `/login`,
  `/logout`, `/me`.
- Modify `app/routers/cloning.py` — every route depends on
  `get_current_user` and scopes its query/writes by `user_id`.
- Modify `app/config.py` — add `session_secret`.
- Modify `app/main.py` — include the auth router, serve the four new
  pages, redirect `/` to `/generate`, remove the `index.html` route.
- Delete `app/static/index.html`.
- Create `app/static/common.js` — `esc()`, `requireAuth()`,
  `renderNav()`, `apiFetch()`.
- Create `app/static/login.html`.
- Create `app/static/generate.html`.
- Create `app/static/voices.html`.
- Create `app/static/history.html`.
- Modify `pyproject.toml` / `requirements.txt` — add `passlib[bcrypt]`,
  `itsdangerous`.
- Modify `tests/conftest.py` — `client` fixture stays as-is; add a
  `logged_in_client` fixture (or a helper) that registers+logs in a user
  and returns a client with the session cookie set, for tests that need
  an authenticated caller.
- Create `tests/test_auth.py`.
- Modify `tests/test_voices.py`, `tests/test_generations.py` — update to
  register/log in first (unauthenticated calls now 401 instead of
  succeeding).

---

### Task 1: User model, session_secret config, and dependencies

**Files:**
- Modify: `app/models.py`
- Modify: `app/config.py`
- Modify: `pyproject.toml`, `requirements.txt`
- Test: `tests/test_models.py` (extend)

**Interfaces:**
- Produces: `User(SQLModel, table=True)` with fields `id: str` (pk),
  `username: str` (unique), `password_hash: str`, `created_at: datetime`.
- Produces: `Voice.user_id: str` (FK `user.id`, required — added as a new
  field, no default, since no migration/backfill is in scope).
- Produces: `Generation.user_id: str` (FK `user.id`, required).
- Produces: `settings.session_secret: str` on the `Settings` class in
  `app/config.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py — add these to the existing file
from app.models import User


def test_user_defaults():
    user = User(id="u1", username="tariro", password_hash="hashed")
    assert user.username == "tariro"
    assert isinstance(user.created_at, datetime)


def test_voice_requires_user_id():
    voice = Voice(id="v1", label="My Voice", filename="raw.wav", wav_path="a.wav", user_id="u1")
    assert voice.user_id == "u1"


def test_generation_requires_user_id():
    gen = Generation(id="g1", voice_id="v1", text="Mhoro", language="en", user_id="u1")
    assert gen.user_id == "u1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ImportError: cannot import name 'User' from 'app.models'`

- [ ] **Step 3: Add dependencies**

In `pyproject.toml`, add to `dependencies` (alongside `sqlmodel`):
`"passlib[bcrypt]>=1.7.4"`, `"itsdangerous>=2.2.0"`. Add the same two
lines to `requirements.txt`.

Run: `uv sync`
Expected: installs cleanly.

- [ ] **Step 4: Write minimal implementation**

In `app/models.py`, add after the existing imports and before
`GenerationStatus`:

```python
class User(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    username: str = Field(unique=True, index=True)
    password_hash: str
    created_at: datetime = Field(default_factory=_now)
```

Add `user_id: str = Field(foreign_key="user.id")` to both `Voice` and
`Generation` (add it right after `id`, before `label`/`voice_id`
respectively).

In `app/config.py`, add to `Settings` (after `db_path`):

```python
    session_secret: str = Field(default_factory=lambda: __import__("secrets").token_hex(32))
```

This needs `from sqlmodel import Field` already used elsewhere in the
codebase for `SQLModel` classes — but `Settings` is a `pydantic_settings.BaseSettings`,
which uses `pydantic.Field`, not `sqlmodel.Field`. Use this instead:

```python
import secrets
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_cache_dir: Path = Path("./storage/models")
    voices_dir: Path = Path("./storage/voices")
    db_path: Path = Path("./storage/db.sqlite3")
    sample_rate: int = 22050
    device: str = "cuda"
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    finetuned_model_path: str = ""
    session_secret: str = Field(default_factory=lambda: secrets.token_hex(32))

    model_config = {"env_prefix": "CLONER_", "env_file": ".env"}
```

(This replaces the full top of `app/config.py` — the rest of the file,
the `settings = Settings()` instantiation and `mkdir` calls, is unchanged.)

Also add, right after `settings = Settings()`:

```python
if "CLONER_SESSION_SECRET" not in __import__("os").environ:
    import logging
    logging.getLogger(__name__).warning(
        "CLONER_SESSION_SECRET not set — using a random secret generated at startup. "
        "Sessions will not survive a process restart. Set CLONER_SESSION_SECRET to persist logins."
    )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (6 tests — 3 existing + 3 new)

- [ ] **Step 6: Commit**

```bash
git add app/models.py app/config.py pyproject.toml requirements.txt uv.lock tests/test_models.py
git commit -m "feat: add User model, user_id FKs, and session secret config"
```

---

### Task 2: Auth service — password hashing and session cookie

**Files:**
- Create: `app/services/auth.py`
- Test: `tests/test_auth_service.py`

**Interfaces:**
- Consumes: `User` from `app/models.py` (Task 1). `settings.session_secret`
  from `app/config.py`.
- Produces: `hash_password(password: str) -> str`.
- Produces: `verify_password(password: str, password_hash: str) -> bool`.
- Produces: `create_session_cookie(user_id: str) -> str` — a signed token.
- Produces: `read_session_cookie(token: str) -> str | None` — returns the
  user id if the signature is valid, `None` otherwise (invalid/expired/
  tampered).
- Produces: `SESSION_COOKIE_NAME = "session"` — module constant, reused by
  the router (Task 3) when setting/clearing the cookie.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth_service.py
from app.services.auth import (
    SESSION_COOKIE_NAME,
    create_session_cookie,
    hash_password,
    read_session_cookie,
    verify_password,
)


def test_hash_and_verify_password_roundtrip():
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong password", h)


def test_hash_password_produces_different_hashes_for_same_input():
    assert hash_password("same") != hash_password("same")


def test_session_cookie_roundtrip():
    token = create_session_cookie("user-123")
    assert read_session_cookie(token) == "user-123"


def test_session_cookie_rejects_tampered_token():
    token = create_session_cookie("user-123")
    tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
    assert read_session_cookie(tampered) is None


def test_session_cookie_name_is_stable_constant():
    assert SESSION_COOKIE_NAME == "session"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_auth_service.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.auth'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/auth.py
from passlib.context import CryptContext
from itsdangerous import BadSignature, URLSafeTimedSerializer

from app.config import settings

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_auth_service.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/auth.py tests/test_auth_service.py
git commit -m "feat: add password hashing and signed session cookie helpers"
```

---

### Task 3: Auth router — register/login/logout/me + get_current_user dependency

**Files:**
- Create: `app/routers/auth.py`
- Modify: `app/services/auth.py` (add the FastAPI dependency)
- Test: `tests/test_auth.py`
- Modify: `tests/conftest.py` (add a registration helper usable by other
  test files)

**Interfaces:**
- Consumes: `hash_password`, `verify_password`, `create_session_cookie`,
  `read_session_cookie`, `SESSION_COOKIE_NAME` from `app/services/auth.py`
  (Task 2). `User` from `app/models.py`. `get_session` from
  `app/services/db.py`.
- Produces: `POST /api/v1/auth/register` — body `{username, password}`.
  400 if username taken or either field blank. Sets the session cookie.
  Returns `{"username": ...}`.
- Produces: `POST /api/v1/auth/login` — body `{username, password}`. 401
  on bad credentials (either field wrong — same message either way). Sets
  the session cookie. Returns `{"username": ...}`.
- Produces: `POST /api/v1/auth/logout` — clears the cookie. Returns
  `{"ok": true}`.
- Produces: `GET /api/v1/auth/me` — 401 if not logged in, else
  `{"username": ...}`.
- Produces (in `app/services/auth.py`):
  `async def get_current_user(request: Request, session: Session = Depends(get_session)) -> User`
  — reads the cookie, verifies it, loads the `User` row; raises
  `HTTPException(401, "Not authenticated")` if the cookie is
  missing/invalid or the user no longer exists. This is what Task 4 wires
  into every voice/generation route.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_auth.py
def test_register_creates_user_and_sets_cookie(client):
    res = client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    assert res.status_code == 200
    assert res.json()["username"] == "tariro"
    assert "session" in res.cookies


def test_register_rejects_duplicate_username(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.post("/api/v1/auth/register", json={"username": "tariro", "password": "different"})
    assert res.status_code == 400


def test_register_rejects_blank_fields(client):
    res = client.post("/api/v1/auth/register", json={"username": "", "password": "x"})
    assert res.status_code == 400
    res = client.post("/api/v1/auth/register", json={"username": "x", "password": ""})
    assert res.status_code == 400


def test_login_succeeds_with_correct_credentials(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.post("/api/v1/auth/login", json={"username": "tariro", "password": "hunter2pass"})
    assert res.status_code == 200
    assert res.json()["username"] == "tariro"


def test_login_rejects_wrong_password(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.post("/api/v1/auth/login", json={"username": "tariro", "password": "wrong"})
    assert res.status_code == 401


def test_login_rejects_unknown_username(client):
    res = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "x"})
    assert res.status_code == 401


def test_me_requires_authentication(client):
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 401


def test_me_returns_username_when_logged_in(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 200
    assert res.json()["username"] == "tariro"


def test_logout_clears_session(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.post("/api/v1/auth/logout")
    assert res.status_code == 200
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 401
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_auth.py -v`
Expected: FAIL — `/api/v1/auth/register` doesn't exist (404).

- [ ] **Step 3: Add `get_current_user` to the auth service**

Append to `app/services/auth.py`:

```python
from fastapi import Depends, HTTPException, Request
from sqlmodel import Session

from app.models import User
from app.services.db import get_session


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
```

- [ ] **Step 4: Write the router**

```python
# app/routers/auth.py
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
```

- [ ] **Step 5: Wire the router into `app/main.py`**

```python
from app.routers import auth, cloning
```

(replaces the existing `from app.routers import cloning`), and add
alongside the existing `app.include_router(cloning.router)`:

```python
app.include_router(auth.router)
```

- [ ] **Step 6: Add a registration helper to `tests/conftest.py`**

Append to `tests/conftest.py`:

```python
def register_and_login(client, username="testuser", password="testpass123"):
    res = client.post("/api/v1/auth/register", json={"username": username, "password": password})
    assert res.status_code == 200, res.text
    return client
```

- [ ] **Step 7: Run test to verify it passes**

Run: `uv run pytest tests/test_auth.py -v`
Expected: PASS (9 tests)

- [ ] **Step 8: Commit**

```bash
git add app/services/auth.py app/routers/auth.py app/main.py tests/test_auth.py tests/conftest.py
git commit -m "feat: add register/login/logout/me endpoints and get_current_user dependency"
```

---

### Task 4: Scope voices and generations to the logged-in user

**Files:**
- Modify: `app/routers/cloning.py`
- Modify: `tests/test_voices.py`, `tests/test_generations.py` (update
  existing tests to log in first; add cross-user isolation tests)

**Interfaces:**
- Consumes: `get_current_user` from `app/services/auth.py` (Task 3).
- Changes every existing route from the prior spec to require
  `current_user: User = Depends(get_current_user)` and filter/stamp by
  `current_user.id`. No new routes, no signature changes to the JSON
  bodies/response shapes beyond scoping.

- [ ] **Step 1: Update existing tests to authenticate first**

In `tests/test_voices.py`, add `from tests.conftest import register_and_login`
at the top, and prepend `register_and_login(client)` as the first line of
every test function body (before the existing `client.post(...)` calls).

In `tests/test_generations.py`, do the same — add the import and prepend
`register_and_login(client)` to every test.

- [ ] **Step 2: Write the new failing cross-user isolation tests**

```python
# tests/test_voices.py — add to the end of the file
from tests.conftest import register_and_login


def test_voice_list_is_scoped_to_current_user(client, tiny_wav_bytes):
    register_and_login(client, "alice", "alicepass123")
    client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Alice's Voice"},
    )
    client.post("/api/v1/auth/logout")

    register_and_login(client, "bob", "bobpass123")
    res = client.get("/api/v1/voices")
    assert res.json() == []


def test_cannot_delete_another_users_voice(client, tiny_wav_bytes):
    register_and_login(client, "alice", "alicepass123")
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Alice's Voice"},
    )
    voice_id = res.json()["voice_id"]
    client.post("/api/v1/auth/logout")

    register_and_login(client, "bob", "bobpass123")
    res = client.delete(f"/api/v1/voices/{voice_id}")
    assert res.status_code == 404


def test_upload_requires_authentication(client, tiny_wav_bytes):
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "No Auth"},
    )
    assert res.status_code == 401
```

```python
# tests/test_generations.py — add to the end of the file
from tests.conftest import register_and_login


def test_cannot_generate_against_another_users_voice(client, tiny_wav_bytes):
    register_and_login(client, "alice", "alicepass123")
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Alice's Voice"},
    )
    voice_id = res.json()["voice_id"]
    client.post("/api/v1/auth/logout")

    register_and_login(client, "bob", "bobpass123")
    res = client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "hi", "language": "en"})
    assert res.status_code == 404


def test_generation_list_is_scoped_to_current_user(client, tiny_wav_bytes):
    register_and_login(client, "alice", "alicepass123")
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Alice's Voice"},
    )
    voice_id = res.json()["voice_id"]
    client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "hi", "language": "en"})
    client.post("/api/v1/auth/logout")

    register_and_login(client, "bob", "bobpass123")
    res = client.get("/api/v1/generations")
    assert res.json() == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_voices.py tests/test_generations.py -v`
Expected: the new cross-user tests FAIL (routes aren't scoped yet); the
updated existing tests still PASS (since routes don't require auth yet,
`register_and_login` is a harmless no-op prefix at this point).

- [ ] **Step 4: Rewrite `app/routers/cloning.py` with scoping**

```python
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import settings
from app.models import Generation, GenerationStatus, User, Voice
from app.services.audio_processor import convert_to_wav
from app.services.auth import get_current_user
from app.services.db import get_session
from app.services.queue import enqueue

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["voices"])


@router.post("/voices/upload")
async def upload_voice(
    file: UploadFile,
    label: str = Form(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not file.filename:
        raise HTTPException(400, "No file provided")
    if not label or not label.strip():
        raise HTTPException(400, "Label is required")

    voice = Voice(label=label.strip(), filename=file.filename, wav_path="", user_id=current_user.id)
    raw_dir = settings.voices_dir / voice.id
    raw_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix or ".wav"
    raw_path = raw_dir / f"raw{ext}"

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")

    raw_path.write_bytes(content)
    wav_path = convert_to_wav(raw_path, raw_dir / "voice", sample_rate=settings.sample_rate)
    raw_path.unlink(missing_ok=True)

    voice.wav_path = str(wav_path)
    session.add(voice)
    session.commit()
    session.refresh(voice)

    return {"voice_id": voice.id, "label": voice.label}


@router.get("/voices")
async def list_voices(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    voices = session.exec(
        select(Voice).where(Voice.user_id == current_user.id).order_by(Voice.created_at.desc())
    ).all()
    return [
        {"id": v.id, "label": v.label, "filename": v.filename, "created_at": v.created_at.isoformat()}
        for v in voices
    ]


@router.delete("/voices/{voice_id}", status_code=204)
async def delete_voice(
    voice_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    voice = session.get(Voice, voice_id)
    if voice is None or voice.user_id != current_user.id:
        raise HTTPException(404, f"Voice '{voice_id}' not found")

    voice_dir = settings.voices_dir / voice_id
    shutil.rmtree(voice_dir, ignore_errors=True)

    session.delete(voice)
    session.commit()
    return Response(status_code=204)


class GenerateRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "en"


@router.post("/generations")
async def create_generation(
    payload: GenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    voice = session.get(Voice, payload.voice_id)
    if voice is None or voice.user_id != current_user.id:
        raise HTTPException(404, f"Voice '{payload.voice_id}' not found")

    gen = Generation(
        voice_id=payload.voice_id,
        text=payload.text,
        language=payload.language,
        user_id=current_user.id,
    )
    session.add(gen)
    session.commit()
    session.refresh(gen)

    await enqueue(gen.id)
    return {"generation_id": gen.id, "status": gen.status}


@router.get("/generations/{generation_id}")
async def get_generation(
    generation_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    gen = session.get(Generation, generation_id)
    if gen is None or gen.user_id != current_user.id:
        raise HTTPException(404, f"Generation '{generation_id}' not found")
    return {
        "id": gen.id,
        "voice_id": gen.voice_id,
        "text": gen.text,
        "language": gen.language,
        "status": gen.status,
        "error": gen.error,
        "created_at": gen.created_at.isoformat(),
        "completed_at": gen.completed_at.isoformat() if gen.completed_at else None,
    }


@router.get("/generations/{generation_id}/audio")
async def get_generation_audio(
    generation_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    gen = session.get(Generation, generation_id)
    if gen is None or gen.user_id != current_user.id:
        raise HTTPException(404, f"Generation '{generation_id}' not found")
    if gen.status != GenerationStatus.DONE:
        return JSONResponse(status_code=409, content={"status": gen.status})
    return FileResponse(gen.output_path, media_type="audio/wav")


@router.get("/generations")
async def list_generations(
    voice_id: str | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    query = select(Generation).where(Generation.user_id == current_user.id).order_by(
        Generation.created_at.desc()
    )
    if voice_id:
        query = query.where(Generation.voice_id == voice_id)
    gens = session.exec(query).all()
    return [
        {
            "id": g.id,
            "voice_id": g.voice_id,
            "text": g.text,
            "status": g.status,
            "created_at": g.created_at.isoformat(),
        }
        for g in gens
    ]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_voices.py tests/test_generations.py -v`
Expected: all PASS, including the new cross-user isolation tests.

- [ ] **Step 6: Run the full suite**

Run: `uv run pytest -v`
Expected: all tests across every file PASS.

- [ ] **Step 7: Commit**

```bash
git add app/routers/cloning.py tests/test_voices.py tests/test_generations.py
git commit -m "feat: scope voices and generations to the authenticated user"
```

---

### Task 5: Shared frontend JS (`common.js`)

**Files:**
- Create: `app/static/common.js`

**Interfaces:**
- Consumes: `/api/v1/auth/me`, `/api/v1/auth/logout` (Task 3).
- Produces: `esc(s)` — HTML-escaping helper (same implementation as the
  one currently inline in `index.html`).
- Produces: `async function requireAuth()` — calls `GET /api/v1/auth/me`;
  on 401, redirects to `/login`; on success, returns
  `{username}`. Pages call this before rendering their own content.
- Produces: `function renderNav(activePage)` — injects a full-height left
  sidebar (Generate / Voices / History links, "Log out" pinned at the
  bottom) into `document.getElementById('nav')`; `activePage` is one of
  `"generate" | "voices" | "history"` and controls which link is marked
  active. This is a sidebar, not a top nav bar — the page layout (Task 7's
  `portal.css`) wraps `#nav` and `<main>` in a flex row so the sidebar
  runs the full height of the viewport with content to its right. The
  "Log out" link calls `/api/v1/auth/logout` then redirects to `/login`.
- Produces: `async function apiFetch(url, opts)` — wraps `fetch(url, {
  ...opts, credentials: 'same-origin' })`; if the response status is 401,
  redirects to `/login` and never resolves (navigation interrupts
  execution) — callers don't need their own 401 handling.

No automated test for this file — it's DOM/browser-only glue with no unit
test harness in this codebase (consistent with how `index.html`'s
original inline script was never tested). Verified manually in Task 9.

- [ ] **Step 1: Write the file**

```javascript
// app/static/common.js

function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function apiFetch(url, opts = {}) {
  const res = await fetch(url, { ...opts, credentials: 'same-origin' });
  if (res.status === 401) {
    window.location.href = '/login';
    return new Promise(() => {}); // navigation is taking over; never resolve
  }
  return res;
}

async function requireAuth() {
  const res = await apiFetch('/api/v1/auth/me');
  return res.json();
}

function renderNav(activePage) {
  const nav = document.getElementById('nav');
  if (!nav) return;
  const links = [
    { id: 'generate', href: '/generate', label: 'Generate' },
    { id: 'voices', href: '/voices', label: 'Voices' },
    { id: 'history', href: '/history', label: 'History' },
  ];
  nav.innerHTML =
    '<div class="sidebar-brand">Shona Voice Cloner</div>' +
    '<div class="sidebar-links">' +
    links.map(l => `<a href="${l.href}" class="${l.id === activePage ? 'active' : ''}">${l.label}</a>`).join('') +
    '</div>' +
    '<a href="#" id="logoutLink" class="sidebar-logout">Log out</a>';

  document.getElementById('logoutLink').addEventListener('click', async (e) => {
    e.preventDefault();
    await fetch('/api/v1/auth/logout', { method: 'POST', credentials: 'same-origin' });
    window.location.href = '/login';
  });
}
```

- [ ] **Step 2: Commit**

```bash
git add app/static/common.js
git commit -m "feat: add shared frontend auth/nav/fetch helpers"
```

---

### Task 6: Login page

**Files:**
- Create: `app/static/login.html`
- Modify: `app/main.py` (serve the page, redirect `/`)

**Interfaces:**
- Consumes: `POST /api/v1/auth/register`, `POST /api/v1/auth/login`
  (Task 3). `esc` from `common.js` (Task 5, though login.html has no
  server-sourced strings to escape — included for consistency, unused is
  fine here).
- Produces: `GET /login` route serving this file (added to `app/main.py`).

- [ ] **Step 1: Write the page**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Shona Voice Cloner — Log In</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, sans-serif;
      background: #F8FAFC;
      color: #1E293B;
      min-height: 100dvh;
      display: flex;
      align-items: center;
      justify-content: center;
    }
    .card {
      background: #FFFFFF;
      border: 1px solid #E2E8F0;
      border-radius: 10px;
      padding: 32px;
      width: 100%;
      max-width: 380px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    h1 { font-size: 18px; margin-bottom: 20px; }
    label { display: block; font-size: 13px; font-weight: 500; margin-bottom: 6px; }
    input {
      width: 100%;
      padding: 9px 12px;
      border: 1px solid #E2E8F0;
      border-radius: 8px;
      font-size: 14px;
      margin-bottom: 16px;
    }
    button {
      width: 100%;
      padding: 10px;
      border: none;
      border-radius: 8px;
      background: #2563EB;
      color: white;
      font-size: 14px;
      font-weight: 600;
      cursor: pointer;
    }
    button:hover { background: #1D4ED8; }
    .toggle { text-align: center; margin-top: 14px; font-size: 13px; }
    .toggle a { color: #2563EB; cursor: pointer; }
    .error { color: #DC2626; font-size: 13px; margin-bottom: 12px; display: none; }
  </style>
</head>
<body>
  <div class="card">
    <h1 id="formTitle">Log in</h1>
    <div class="error" id="errorMsg"></div>
    <form id="authForm">
      <label for="username">Username</label>
      <input type="text" id="username" autocomplete="username" required />
      <label for="password">Password</label>
      <input type="password" id="password" autocomplete="current-password" required />
      <button type="submit" id="submitBtn">Log in</button>
    </form>
    <div class="toggle">
      <span id="toggleText">Don't have an account?</span>
      <a id="toggleLink">Register</a>
    </div>
  </div>

  <script>
    let mode = 'login'; // or 'register'

    const form = document.getElementById('authForm');
    const errorMsg = document.getElementById('errorMsg');
    const toggleLink = document.getElementById('toggleLink');

    toggleLink.addEventListener('click', () => {
      mode = mode === 'login' ? 'register' : 'login';
      document.getElementById('formTitle').textContent = mode === 'login' ? 'Log in' : 'Register';
      document.getElementById('submitBtn').textContent = mode === 'login' ? 'Log in' : 'Register';
      document.getElementById('toggleText').textContent = mode === 'login' ? "Don't have an account?" : 'Already have an account?';
      toggleLink.textContent = mode === 'login' ? 'Register' : 'Log in';
      errorMsg.style.display = 'none';
    });

    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      errorMsg.style.display = 'none';
      const username = document.getElementById('username').value.trim();
      const password = document.getElementById('password').value;

      try {
        const res = await fetch(`/api/v1/auth/${mode}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          credentials: 'same-origin',
          body: JSON.stringify({ username, password }),
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Something went wrong.');
        window.location.href = '/generate';
      } catch (err) {
        errorMsg.textContent = err.message;
        errorMsg.style.display = 'block';
      }
    });
  </script>
</body>
</html>
```

- [ ] **Step 2: Wire the route in `app/main.py`**

Replace the existing `ui()` route and its imports:

```python
from fastapi.responses import FileResponse, RedirectResponse
```

(replaces `from fastapi.responses import FileResponse`), and replace the
`@app.get("/")` handler:

```python
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/generate")


@app.get("/login", include_in_schema=False)
async def login_page():
    return FileResponse(str(_static / "login.html"))
```

(Task 7-9 add the `/generate`, `/voices`, `/history` routes the same way;
this step only adds `/login` and the `/` redirect so this task's page is
independently reachable and testable.)

- [ ] **Step 3: Manual verification**

Run: `uv run uvicorn app.main:app --reload`, visit `http://localhost:8000/login`.
- Register a new user → redirects to `/generate` (will 404 until Task 7;
  that's expected at this point — confirm the POST succeeded via Network
  tab / a 200 response before the redirect).
- Toggle to "Log in" mode, log out via `curl -X POST localhost:8000/api/v1/auth/logout -b cookies.txt`, revisit `/login`, log back in with the same credentials → succeeds.
- Try registering the same username twice → error message shown inline.

- [ ] **Step 4: Commit**

```bash
git add app/static/login.html app/main.py
git commit -m "feat: add login/register page"
```

---

### Task 7: Generate page

**Files:**
- Create: `app/static/generate.html`
- Modify: `app/main.py` (serve the page)
- Delete: `app/static/index.html`

**Interfaces:**
- Consumes: `common.js` (`esc`, `requireAuth`, `renderNav`, `apiFetch`)
  from Task 5. `GET /api/v1/voices`, `POST /api/v1/generations`,
  `GET /api/v1/generations/{id}`, `GET /api/v1/generations/{id}/audio`
  from Task 4.
- Produces: `GET /generate` route serving this file.

This carries over the generate-flow markup/logic from the old
`index.html` (Task 6 of the prior plan), minus the inline upload form
(moved to `voices.html`, Task 8) and minus the history list (moved to
`history.html`, Task 9) — voice selection here is a picker only, refreshed
from `GET /api/v1/voices` on load.

- [ ] **Step 1: Write the page**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Shona Voice Cloner — Generate</title>
  <link rel="stylesheet" href="/static/portal.css" />
</head>
<body>
<div class="portal-shell">
  <nav id="nav" class="sidebar"></nav>
  <main>
  <div class="card">
    <div class="card-title">Generate Speech</div>

    <div class="form-group" style="margin-bottom:20px;">
      <label for="voicePicker">Voice</label>
      <select id="voicePicker" aria-label="Saved voices">
        <option value="">— none selected —</option>
      </select>
      <p class="helper">No voices yet? <a href="/voices">Upload one</a>.</p>
    </div>

    <label for="textInput">Text to synthesize *</label>
    <textarea id="textInput" placeholder="Type or paste Shona text here..." oninput="updateCharCount()"></textarea>
    <p class="helper"><span id="charCount">0</span> characters</p>

    <div class="form-group" style="margin-top:16px;">
      <label for="languageSelect">Language mode</label>
      <select id="languageSelect">
        <option value="en">English (en) — recommended for Shona</option>
        <option value="fr">French (fr)</option>
        <option value="de">German (de)</option>
        <option value="es">Spanish (es)</option>
        <option value="pt">Portuguese (pt)</option>
        <option value="ar">Arabic (ar)</option>
        <option value="hi">Hindi (hi)</option>
        <option value="zh-cn">Chinese (zh-cn)</option>
        <option value="ja">Japanese (ja)</option>
      </select>
      <p class="helper">This model was finetuned on Shona speech using the <strong>en</strong> language token.</p>
    </div>

    <div class="generate-actions">
      <span id="voiceStatus" class="helper">No voice selected</span>
      <button class="btn btn-cta" id="generateBtn" onclick="generate()" disabled>Generate Audio</button>
    </div>

    <div class="status-bar" id="generateStatus"></div>

    <div class="audio-result" id="audioResult">
      <audio id="audioPlayer" controls></audio>
      <button class="btn btn-ghost" id="downloadBtn">Download WAV</button>
    </div>
  </div>
  </main>
</div>

<script src="/static/common.js"></script>
<script>
  let voiceId = null;
  let pollTimer = null;
  const voicePicker = document.getElementById('voicePicker');

  async function init() {
    await requireAuth();
    renderNav('generate');
    await loadVoices();
  }

  async function loadVoices() {
    const res = await apiFetch('/api/v1/voices');
    const voices = await res.json();
    voicePicker.innerHTML = '<option value="">— none selected —</option>' +
      voices.map(v => `<option value="${esc(v.id)}">${esc(v.label)}</option>`).join('');
  }

  voicePicker.addEventListener('change', () => selectVoice(voicePicker.value));

  function selectVoice(id) {
    voiceId = id || null;
    const status = document.getElementById('voiceStatus');
    const btn = document.getElementById('generateBtn');
    status.textContent = voiceId ? 'Voice ready' : 'No voice selected';
    btn.disabled = !voiceId;
  }

  async function generate() {
    const text = document.getElementById('textInput').value.trim();
    if (!text) { setStatus('generateStatus', 'error', 'Please enter text to synthesize.'); return; }
    if (!voiceId) { setStatus('generateStatus', 'error', 'Please select a voice first.'); return; }

    const language = document.getElementById('languageSelect').value;
    const btn = document.getElementById('generateBtn');
    btn.disabled = true;
    document.getElementById('audioResult').classList.remove('visible');
    setStatus('generateStatus', 'loading', 'Queued for generation...');

    try {
      const res = await apiFetch('/api/v1/generations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_id: voiceId, text, language }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `Server error ${res.status}`);
      const { generation_id } = await res.json();
      pollGeneration(generation_id);
    } catch (err) {
      setStatus('generateStatus', 'error', err.message);
      btn.disabled = false;
    }
  }

  function pollGeneration(generationId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      const res = await apiFetch(`/api/v1/generations/${generationId}`);
      const gen = await res.json();

      if (gen.status === 'pending' || gen.status === 'running') {
        setStatus('generateStatus', 'loading', `Status: ${gen.status}...`);
        return;
      }

      clearInterval(pollTimer);
      document.getElementById('generateBtn').disabled = false;

      if (gen.status === 'done') {
        clearStatus('generateStatus');
        const audioRes = await apiFetch(`/api/v1/generations/${generationId}/audio`);
        const blob = await audioRes.blob();
        const player = document.getElementById('audioPlayer');
        player.src = URL.createObjectURL(blob);
        document.getElementById('audioResult').classList.add('visible');
        player.play();
        document.getElementById('downloadBtn').onclick = () => {
          const a = document.createElement('a');
          a.href = URL.createObjectURL(blob);
          a.download = 'shona_voice_output.wav';
          a.click();
        };
      } else {
        setStatus('generateStatus', 'error', gen.error || 'Generation failed.');
      }
    }, 1500);
  }

  function updateCharCount() {
    document.getElementById('charCount').textContent = document.getElementById('textInput').value.length;
  }

  function setStatus(id, type, msg) {
    const el = document.getElementById(id);
    el.className = `status-bar visible ${type}`;
    el.textContent = msg;
  }

  function clearStatus(id) {
    const el = document.getElementById(id);
    el.className = 'status-bar';
    el.textContent = '';
  }

  init();
</script>
</body>
</html>
```

- [ ] **Step 2: Extract shared CSS with a left sidebar layout**

Create `app/static/portal.css` containing the `.card`, `.btn`,
`.status-bar`, `.helper`, `textarea`, `select`, and `main` content rules —
copy the relevant style rules from the old `index.html`'s `<style>` block
(`.card`, `.card-title`, `.btn*`, `.status-bar*`, `.audio-result*`,
`textarea`, `select`, `.generate-actions`, `.helper`, body/`:root`
variables), and replace the old top-`<header>` rules with a full-height
sidebar layout:

```css
body { margin: 0; }
.portal-shell { display: flex; min-height: 100dvh; }

.sidebar {
  width: 220px;
  flex-shrink: 0;
  background: var(--surface, #FFFFFF);
  border-right: 1px solid var(--border, #E2E8F0);
  display: flex;
  flex-direction: column;
  padding: 20px 0;
}
.sidebar-brand { font-size: 15px; font-weight: 600; padding: 0 20px 20px; }
.sidebar-links { display: flex; flex-direction: column; flex: 1; }
.sidebar-links a, .sidebar-logout {
  padding: 10px 20px;
  color: var(--text-secondary, #64748B);
  text-decoration: none;
  font-size: 14px;
  font-weight: 500;
}
.sidebar-links a.active { color: var(--primary, #2563EB); background: #EFF6FF; }
.sidebar-links a:hover, .sidebar-logout:hover { background: var(--bg, #F8FAFC); }
.sidebar-logout { border-top: 1px solid var(--border, #E2E8F0); margin-top: auto; cursor: pointer; }

main {
  flex: 1;
  max-width: 820px;
  margin: 0 auto;
  padding: 40px 24px 80px;
  display: flex;
  flex-direction: column;
  gap: 24px;
}
```

This file is shared by `generate.html`, `voices.html`, and `history.html`
(Tasks 7-9 all link it via `<link rel="stylesheet" href="/static/portal.css" />`).
Every page uses the same body structure:
`<div class="portal-shell"><nav id="nav" class="sidebar"></nav><main>...page content...</main></div>` —
`renderNav()` (Task 5) fills in the `#nav` sidebar's contents; the page
itself only needs the empty `<nav id="nav" class="sidebar"></nav>` element
and its own `<main>` content. Tasks 8 and 9 must use this exact wrapper
structure, not the old `<header><h1>...</h1><nav id="nav"></nav></header>`
pattern.

- [ ] **Step 3: Wire the route and delete `index.html`**

In `app/main.py`, add:

```python
@app.get("/generate", include_in_schema=False)
async def generate_page():
    return FileResponse(str(_static / "generate.html"))
```

Delete `app/static/index.html` (`rm app/static/index.html`).

- [ ] **Step 4: Manual verification**

Run: `uv run uvicorn app.main:app --reload`, visit `http://localhost:8000/generate`
while logged out → confirm client-side redirect to `/login` (via
`requireAuth()`). Log in, revisit `/generate` → page loads, left sidebar
shows Generate/Voices/History/Log out with Generate marked active.

- [ ] **Step 5: Commit**

```bash
git add app/static/generate.html app/static/portal.css app/main.py
git rm app/static/index.html
git commit -m "feat: add generate page, shared portal styles; remove old single-page UI"
```

---

### Task 8: Voices page

**Files:**
- Create: `app/static/voices.html`
- Modify: `app/main.py` (serve the page)

**Interfaces:**
- Consumes: `common.js` (Task 5). `POST /api/v1/voices/upload`,
  `GET /api/v1/voices`, `DELETE /api/v1/voices/{id}` (Task 4).
- Produces: `GET /voices` route serving this file.

- [ ] **Step 1: Write the page**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Shona Voice Cloner — Voices</title>
  <link rel="stylesheet" href="/static/portal.css" />
</head>
<body>
<div class="portal-shell">
  <nav id="nav" class="sidebar"></nav>
  <main>
  <div class="card">
    <div class="card-title">Upload Voice Sample</div>
    <div class="form-group" style="margin-bottom:16px;">
      <label for="labelInput">Voice label *</label>
      <input type="text" id="labelInput" placeholder="e.g. Tariro — female, warm" />
    </div>
    <div class="upload-zone" id="uploadZone">
      <input type="file" id="fileInput" accept=".mp3,.wav,.ogg,.m4a,.flac" aria-label="Upload audio file" />
      <p><strong>Click to upload</strong> or drag and drop</p>
      <p class="file-types">MP3, WAV, OGG, M4A, FLAC — up to 50MB</p>
    </div>
    <div class="status-bar" id="uploadStatus"></div>
  </div>

  <div class="card">
    <div class="card-title">Your Voices</div>
    <div id="voiceList"><p class="helper">No voices yet.</p></div>
  </div>
  </main>
</div>

<script src="/static/common.js"></script>
<script>
  const fileInput = document.getElementById('fileInput');
  const uploadZone = document.getElementById('uploadZone');

  async function init() {
    await requireAuth();
    renderNav('voices');
    await loadVoices();
  }

  uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
  uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    const file = e.dataTransfer.files[0];
    if (file) uploadFile(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) uploadFile(fileInput.files[0]);
  });

  async function uploadFile(file) {
    const label = document.getElementById('labelInput').value.trim();
    if (!label) { setStatus('uploadStatus', 'error', 'Enter a label for this voice first.'); return; }
    setStatus('uploadStatus', 'loading', `Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', label);

    try {
      const res = await apiFetch('/api/v1/voices/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Upload failed');
      clearStatus('uploadStatus');
      document.getElementById('labelInput').value = '';
      fileInput.value = '';
      await loadVoices();
    } catch (err) {
      setStatus('uploadStatus', 'error', err.message);
    }
  }

  async function loadVoices() {
    const res = await apiFetch('/api/v1/voices');
    const voices = await res.json();
    const container = document.getElementById('voiceList');
    if (voices.length === 0) {
      container.innerHTML = '<p class="helper">No voices yet.</p>';
      return;
    }
    container.innerHTML = voices.map(v => `
      <div class="voice-row" data-id="${esc(v.id)}">
        <span>${esc(v.label)}</span>
        <button class="btn btn-ghost delete-btn" data-id="${esc(v.id)}">Delete</button>
      </div>
    `).join('');
    container.querySelectorAll('.delete-btn').forEach(btn => {
      btn.addEventListener('click', () => deleteVoice(btn.dataset.id));
    });
  }

  async function deleteVoice(id) {
    await apiFetch(`/api/v1/voices/${id}`, { method: 'DELETE' });
    await loadVoices();
  }

  function setStatus(id, type, msg) {
    const el = document.getElementById(id);
    el.className = `status-bar visible ${type}`;
    el.textContent = msg;
  }

  function clearStatus(id) {
    const el = document.getElementById(id);
    el.className = 'status-bar';
    el.textContent = '';
  }

  init();
</script>
</body>
</html>
```

- [ ] **Step 2: Add `.voice-row` styling to `portal.css`**

```css
.voice-row { display:flex; justify-content:space-between; align-items:center; padding:12px; border:1px solid var(--border, #E2E8F0); border-radius:8px; margin-bottom:8px; }
```

- [ ] **Step 3: Wire the route**

In `app/main.py`, add:

```python
@app.get("/voices", include_in_schema=False)
async def voices_page():
    return FileResponse(str(_static / "voices.html"))
```

- [ ] **Step 4: Manual verification**

Visit `/voices` while logged in. Upload a voice with a label → appears in
the list below. Click Delete → disappears, confirmed gone via
`GET /api/v1/voices`. Navigate to `/generate` → the newly uploaded voice
appears in the picker there too (proves both pages read the same backend
state).

- [ ] **Step 5: Commit**

```bash
git add app/static/voices.html app/static/portal.css app/main.py
git commit -m "feat: add voices library page"
```

---

### Task 9: History page

**Files:**
- Create: `app/static/history.html`
- Modify: `app/main.py` (serve the page)

**Interfaces:**
- Consumes: `common.js` (Task 5). `GET /api/v1/generations`,
  `GET /api/v1/voices` (for the filter dropdown), from Task 4.
- Produces: `GET /history` route serving this file.

- [ ] **Step 1: Write the page**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Shona Voice Cloner — History</title>
  <link rel="stylesheet" href="/static/portal.css" />
</head>
<body>
<div class="portal-shell">
  <nav id="nav" class="sidebar"></nav>
  <main>
  <div class="card">
    <div class="card-title">Generation History</div>
    <div class="form-group" style="margin-bottom:16px;">
      <label for="voiceFilter">Filter by voice</label>
      <select id="voiceFilter">
        <option value="">All voices</option>
      </select>
    </div>
    <div id="historyList"><p class="helper">No generations yet.</p></div>
    <button class="btn btn-ghost" id="loadMoreBtn" style="display:none; margin-top:12px;">Load more</button>
  </div>
  </main>
</div>

<script src="/static/common.js"></script>
<script>
  const PAGE_SIZE = 50;
  let offset = 0;
  let currentVoiceFilter = '';

  async function init() {
    await requireAuth();
    renderNav('history');
    await loadVoiceFilterOptions();
    await loadHistory(true);
  }

  async function loadVoiceFilterOptions() {
    const res = await apiFetch('/api/v1/voices');
    const voices = await res.json();
    const select = document.getElementById('voiceFilter');
    select.innerHTML = '<option value="">All voices</option>' +
      voices.map(v => `<option value="${esc(v.id)}">${esc(v.label)}</option>`).join('');
    select.addEventListener('change', () => {
      currentVoiceFilter = select.value;
      loadHistory(true);
    });
  }

  async function loadHistory(reset) {
    if (reset) offset = 0;
    const url = currentVoiceFilter
      ? `/api/v1/generations?voice_id=${encodeURIComponent(currentVoiceFilter)}`
      : '/api/v1/generations';
    const res = await apiFetch(url);
    const gens = await res.json();
    const container = document.getElementById('historyList');
    const page = gens.slice(offset, offset + PAGE_SIZE);

    if (gens.length === 0) {
      container.innerHTML = '<p class="helper">No generations yet.</p>';
      document.getElementById('loadMoreBtn').style.display = 'none';
      return;
    }

    const rows = page.map(g => `
      <div class="history-row">
        <span class="history-text">${esc(g.text)}</span>
        <span class="history-status status-${esc(g.status)}">
          ${esc(g.status)}${g.status === 'done' ? ` — <a href="/api/v1/generations/${esc(g.id)}/audio" target="_blank">play</a>` : ''}
        </span>
      </div>
    `).join('');

    container.innerHTML = reset ? rows : container.innerHTML + rows;
    offset += page.length;

    document.getElementById('loadMoreBtn').style.display = offset < gens.length ? 'inline-flex' : 'none';
  }

  document.getElementById('loadMoreBtn').addEventListener('click', () => loadHistory(false));

  init();
</script>
</body>
</html>
```

- [ ] **Step 2: Add history-row styling to `portal.css`**

```css
.history-row { display:flex; justify-content:space-between; align-items:center; padding:10px 12px; border:1px solid var(--border, #E2E8F0); border-radius:8px; margin-bottom:8px; font-size:13px; }
.status-done { color: #16A34A; font-weight:600; }
.status-failed { color: #DC2626; font-weight:600; }
.status-pending, .status-running { color: #64748B; font-weight:600; }
```

- [ ] **Step 3: Wire the route**

In `app/main.py`, add:

```python
@app.get("/history", include_in_schema=False)
async def history_page():
    return FileResponse(str(_static / "history.html"))
```

- [ ] **Step 4: Manual verification**

Visit `/history` while logged in. Confirm past generations (from Task 4's
manual testing) appear, filtering by voice narrows the list, and a
completed generation's "play" link serves working audio.

- [ ] **Step 5: Commit**

```bash
git add app/static/history.html app/static/portal.css app/main.py
git commit -m "feat: add generation history page with voice filter"
```

---

## Self-Review Notes

- **Spec coverage:** `User` model + FKs (Task 1), password hashing +
  session cookie (Task 2), register/login/logout/me + `get_current_user`
  (Task 3), full endpoint scoping with 404-not-403 cross-user policy
  (Task 4), shared JS helpers (Task 5), and all four pages (Tasks 6-9) —
  every spec section maps to a task.
- **Type/interface consistency:** `get_current_user` is defined once in
  `app/services/auth.py` (Task 3) and consumed identically by every route
  in Task 4 — same import path, same dependency signature. `SESSION_COOKIE_NAME`
  is defined once (Task 2) and reused by both the router (Task 3, to set/
  clear it) and nowhere else needs to know its literal value. `esc()`,
  `requireAuth()`, `renderNav()`, `apiFetch()` are defined once in
  `common.js` (Task 5) and consumed identically (same signatures) across
  Tasks 7-9 — no page redefines them.
- **No placeholders:** every step has complete code, not descriptions of
  code. The `session_secret` default-generation warning is explicit code,
  not a "handle this later" note.
- **Scope discipline:** no rename endpoint, no OAuth, no password reset —
  none of these appear anywhere in the plan, matching the spec's explicit
  exclusions.
