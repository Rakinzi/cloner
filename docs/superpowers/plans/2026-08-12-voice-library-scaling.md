# Voice Library + Generation Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace anonymous UUID voice folders with a SQLite-backed, labeled
voice library, and serialize all speech generation through a single async
worker queue so concurrent requests never race on the GPU, while every
generation is tracked as browsable history.

**Architecture:** `sqlmodel` models (`Voice`, `Generation`) backed by
`storage/db.sqlite3`. A module-level `asyncio.Queue` plus one background
worker task (started in the FastAPI `lifespan`) is the sole caller of the
existing, unchanged `TTSModelManager.generate()` — this makes "only one
generation runs at a time" true by construction, no locks needed. Routers
become thin: validate, write/read DB rows, enqueue. The static frontend
moves from a single in-memory `voiceId` to fetching a voice list and polling
generation status.

**Tech Stack:** FastAPI, `sqlmodel` (new dependency), SQLite, stdlib
`asyncio`, `pytest` + `httpx.TestClient` (new dev dependency for testing).

## Global Constraints

- SQLite only, no Postgres, no external job broker (Celery/Redis) — spec:
  "Explicitly out of scope: auth/multi-user, Postgres, an external job
  broker, rate limiting."
- No migrations tooling (alembic) — `SQLModel.metadata.create_all(engine)`
  on startup is sufficient for this schema.
- `TTSModelManager.generate()` in `app/services/tts_model.py` is unchanged —
  it stays fully synchronous/blocking; the queue worker calls it via
  `asyncio.to_thread`.
- Exactly one worker task processes the queue — this is the entire
  concurrency guarantee for GPU access, per spec.
- Voice upload requires a non-empty `label` — 400 if missing/empty.
- Unknown `voice_id` on generate → 404 before enqueueing (fail fast).
- Fetching audio for a non-`done` generation → 409 with current status in
  the body (not 404 — the row exists, it's just not ready).
- A failed generation must not wedge the queue — worker catches exceptions
  per-job and continues.

---

## File Structure

- Create `app/models.py` — `Voice` and `Generation` sqlmodel table classes.
- Create `app/services/db.py` — engine, `create_db_and_tables()`, `get_session()`.
- Create `app/services/queue.py` — `asyncio.Queue`, `enqueue()`, `worker()`.
- Rewrite `app/routers/cloning.py` — voice CRUD + generation endpoints.
- Modify `app/main.py` — call `create_db_and_tables()` and start the worker
  task in `lifespan`.
- Modify `app/config.py` — add `db_path`.
- Modify `app/static/index.html` — voice picker + label input + polling.
- Modify `pyproject.toml` / `requirements.txt` — add `sqlmodel`; add
  `pytest` to the dev group.
- Create `tests/conftest.py` — shared fixtures (temp DB, `TestClient`).
- Create `tests/test_voices.py` — voice upload/list/delete tests.
- Create `tests/test_generations.py` — generation queue/status/audio tests.

---

### Task 1: Data models

**Files:**
- Create: `app/models.py`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `Voice(SQLModel, table=True)` with fields `id: str` (pk),
  `label: str`, `filename: str`, `wav_path: str`,
  `created_at: datetime`.
- Produces: `Generation(SQLModel, table=True)` with fields `id: str` (pk),
  `voice_id: str`, `text: str`, `language: str`, `status: str`,
  `output_path: str | None`, `error: str | None`, `created_at: datetime`,
  `completed_at: datetime | None`.
- Produces: `GenerationStatus` — a small `str` enum-like constant set:
  `PENDING = "pending"`, `RUNNING = "running"`, `DONE = "done"`,
  `FAILED = "failed"`, used by later tasks instead of raw string literals.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py
from datetime import datetime, timezone

from app.models import Generation, GenerationStatus, Voice


def test_voice_defaults():
    voice = Voice(id="v1", label="My Voice", filename="raw.wav", wav_path="storage/voices/v1/voice.wav")
    assert voice.id == "v1"
    assert voice.label == "My Voice"
    assert isinstance(voice.created_at, datetime)


def test_generation_defaults():
    gen = Generation(id="g1", voice_id="v1", text="Mhoro", language="en")
    assert gen.status == GenerationStatus.PENDING
    assert gen.output_path is None
    assert gen.error is None
    assert gen.completed_at is None


def test_generation_status_values():
    assert GenerationStatus.PENDING == "pending"
    assert GenerationStatus.RUNNING == "running"
    assert GenerationStatus.DONE == "done"
    assert GenerationStatus.FAILED == "failed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.models'`

- [ ] **Step 3: Add `sqlmodel` dependency**

In `pyproject.toml`, add `"sqlmodel>=0.0.22"` to the `dependencies` list
(alongside `fastapi`, near the other core deps). Add the same line to
`requirements.txt`. Also add a `[dependency-groups] dev` entry for
`"pytest>=8.0"` (keep the existing `httpx>=0.28.1` entry).

Run: `uv sync`
Expected: installs `sqlmodel` and `pytest` without errors.

- [ ] **Step 4: Write minimal implementation**

```python
# app/models.py
import uuid
from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return str(uuid.uuid4())


def _now() -> datetime:
    return datetime.now(timezone.utc)


class GenerationStatus:
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Voice(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    label: str
    filename: str
    wav_path: str
    created_at: datetime = Field(default_factory=_now)


class Generation(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    voice_id: str = Field(foreign_key="voice.id")
    text: str
    language: str
    status: str = Field(default=GenerationStatus.PENDING)
    output_path: str | None = Field(default=None)
    error: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=_now)
    completed_at: datetime | None = Field(default=None)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add app/models.py tests/test_models.py pyproject.toml requirements.txt uv.lock
git commit -m "feat: add Voice and Generation sqlmodel tables"
```

---

### Task 2: DB engine and session helper

**Files:**
- Create: `app/services/db.py`
- Modify: `app/config.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `Voice`, `Generation` from `app/models.py` (Task 1).
- Produces: `create_db_and_tables() -> None` — creates the sqlite file and
  tables if missing.
- Produces: `get_session() -> Generator[Session, None, None]` — FastAPI
  dependency yielding a `sqlmodel.Session`.
- Produces: `engine` — module-level `sqlmodel.create_engine` instance,
  importable for direct use in the queue worker (which runs outside
  FastAPI's request/dependency cycle).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_db.py
import os

from sqlmodel import Session, select

from app.models import Voice


def test_create_db_and_tables_creates_voice_table(tmp_path, monkeypatch):
    db_file = tmp_path / "test.sqlite3"
    monkeypatch.setenv("CLONER_DB_PATH", str(db_file))

    # Re-import with the new env var in effect
    import importlib

    import app.config
    import app.services.db as db_module

    importlib.reload(app.config)
    importlib.reload(db_module)

    db_module.create_db_and_tables()
    assert db_file.exists()

    with Session(db_module.engine) as session:
        session.add(Voice(id="v1", label="Test", filename="a.wav", wav_path="a.wav"))
        session.commit()
        result = session.exec(select(Voice)).all()
        assert len(result) == 1
        assert result[0].label == "Test"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.db'`

- [ ] **Step 3: Add `db_path` to config**

In `app/config.py`, add to the `Settings` class (after `voices_dir`):

```python
    db_path: Path = Path("./storage/db.sqlite3")
```

- [ ] **Step 4: Write minimal implementation**

```python
# app/services/db.py
from typing import Generator

from sqlmodel import Session, SQLModel, create_engine

from app.config import settings

settings.db_path.parent.mkdir(parents=True, exist_ok=True)
engine = create_engine(f"sqlite:///{settings.db_path}", connect_args={"check_same_thread": False})


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, None, None]:
    with Session(engine) as session:
        yield session
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add app/services/db.py app/config.py tests/test_db.py
git commit -m "feat: add sqlite engine and session helper"
```

---

### Task 3: Generation queue worker

**Files:**
- Create: `app/services/queue.py`
- Test: `tests/test_queue.py`

**Interfaces:**
- Consumes: `engine` from `app/services/db.py` (Task 2). `Generation`,
  `GenerationStatus` from `app/models.py` (Task 1).
- Consumes: `TTSModelManager` from `app/services/tts_model.py` — existing
  `generate(text: str, voice_path: str, language: str = "en") -> bytes`.
- Produces: `generation_queue: asyncio.Queue[str]` — module-level queue of
  generation ids.
- Produces: `async def enqueue(generation_id: str) -> None` — puts an id on
  the queue.
- Produces: `async def worker() -> None` — infinite loop: pull one id,
  process it, repeat. Intended to run as a single background task.
- Produces: `async def process_one(generation_id: str) -> None` — processes
  exactly one generation by id (marks running, calls the model via
  `asyncio.to_thread`, writes output, marks done/failed). Exposed
  separately from `worker()` so tests can call it directly without needing
  a running queue loop.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_queue.py
from unittest.mock import patch

import pytest
from sqlmodel import Session

from app.models import Generation, GenerationStatus, Voice
from app.services import queue as queue_module
from app.services.db import engine, create_db_and_tables


@pytest.fixture(autouse=True)
def _fresh_db():
    create_db_and_tables()
    yield


def _make_voice_and_generation(text="Mhoro"):
    with Session(engine) as session:
        voice = Voice(label="Test Voice", filename="a.wav", wav_path="storage/voices/v1/voice.wav")
        session.add(voice)
        session.commit()
        session.refresh(voice)

        gen = Generation(voice_id=voice.id, text=text, language="en")
        session.add(gen)
        session.commit()
        session.refresh(gen)
        return voice.id, gen.id


@pytest.mark.asyncio
async def test_process_one_marks_done_on_success(tmp_path):
    voice_id, gen_id = _make_voice_and_generation()

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch("app.services.tts_model.TTSModelManager.generate", return_value=b"FAKEWAV"):
            await queue_module.process_one(gen_id)

    with Session(engine) as session:
        gen = session.get(Generation, gen_id)
        assert gen.status == GenerationStatus.DONE
        assert gen.output_path is not None
        assert (tmp_path / f"{gen_id}.wav").exists()
        assert gen.completed_at is not None


@pytest.mark.asyncio
async def test_process_one_marks_failed_on_exception(tmp_path):
    voice_id, gen_id = _make_voice_and_generation()

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch("app.services.tts_model.TTSModelManager.generate", side_effect=RuntimeError("boom")):
            await queue_module.process_one(gen_id)

    with Session(engine) as session:
        gen = session.get(Generation, gen_id)
        assert gen.status == GenerationStatus.FAILED
        assert gen.error == "boom"
        assert gen.output_path is None
```

Add `pytest-asyncio` to the dev dependency group in `pyproject.toml`
(`"pytest-asyncio>=0.24"`) and to `requirements.txt`, then `uv sync`. Add a
`tests/conftest.py` (if not already created by a later task) with:

```python
# tests/conftest.py
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
```

And in `pyproject.toml` add:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_queue.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'app.services.queue'`

- [ ] **Step 3: Write minimal implementation**

```python
# app/services/queue.py
import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session

from app.models import Generation, GenerationStatus
from app.services.db import engine
from app.services.tts_model import TTSModelManager

logger = logging.getLogger(__name__)

GENERATIONS_DIR = Path("./storage/generations")
GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)

generation_queue: asyncio.Queue[str] = asyncio.Queue()


async def enqueue(generation_id: str) -> None:
    await generation_queue.put(generation_id)


async def process_one(generation_id: str) -> None:
    with Session(engine) as session:
        gen = session.get(Generation, generation_id)
        if gen is None:
            logger.warning("Generation %s vanished before processing", generation_id)
            return
        gen.status = GenerationStatus.RUNNING
        session.add(gen)
        session.commit()
        voice_id, text, language = gen.voice_id, gen.text, gen.language

    from app.models import Voice

    with Session(engine) as session:
        voice = session.get(Voice, voice_id)
        voice_path = voice.wav_path

    manager = TTSModelManager()
    try:
        wav_bytes = await asyncio.to_thread(manager.generate, text, voice_path, language)
        GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = GENERATIONS_DIR / f"{generation_id}.wav"
        output_path.write_bytes(wav_bytes)

        with Session(engine) as session:
            gen = session.get(Generation, generation_id)
            gen.status = GenerationStatus.DONE
            gen.output_path = str(output_path)
            gen.completed_at = datetime.now(timezone.utc)
            session.add(gen)
            session.commit()
    except Exception as exc:
        logger.exception("Generation %s failed", generation_id)
        with Session(engine) as session:
            gen = session.get(Generation, generation_id)
            gen.status = GenerationStatus.FAILED
            gen.error = str(exc)
            gen.completed_at = datetime.now(timezone.utc)
            session.add(gen)
            session.commit()


async def worker() -> None:
    while True:
        generation_id = await generation_queue.get()
        try:
            await process_one(generation_id)
        finally:
            generation_queue.task_done()
```

Note: `manager.generate` currently takes `(self, text, voice_path,
language)` as keyword-friendly positional args — check the exact signature
in `app/services/tts_model.py:133` before wiring the `asyncio.to_thread`
call; pass them positionally as shown (matches the existing signature).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_queue.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add app/services/queue.py tests/test_queue.py tests/conftest.py pyproject.toml requirements.txt uv.lock
git commit -m "feat: add single-worker generation queue"
```

---

### Task 4: Voice router (upload, list, delete)

**Files:**
- Rewrite: `app/routers/cloning.py` (voice endpoints only — generation
  endpoints come in Task 5)
- Test: `tests/test_voices.py`
- Create: `tests/conftest.py` additions (shared `client` fixture) — extend
  the file created in Task 3.

**Interfaces:**
- Consumes: `Voice` from `app/models.py`. `get_session` from
  `app/services/db.py`. `convert_to_wav` from
  `app/services/audio_processor.py` (unchanged, existing).
- Produces: `POST /api/v1/voices/upload` — form fields `file`, `label`.
  Returns `{voice_id, label}`. 400 if `label` missing/blank.
- Produces: `GET /api/v1/voices` — returns
  `[{id, label, filename, created_at}, ...]`, newest first.
- Produces: `DELETE /api/v1/voices/{voice_id}` — 204 on success, 404 if not
  found.

- [ ] **Step 1: Write the failing test**

```python
# tests/conftest.py  (add to the file from Task 3)
import io

import pytest
from fastapi.testclient import TestClient

from app.services.db import create_db_and_tables


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CLONER_DB_PATH", str(tmp_path / "test.sqlite3"))
    monkeypatch.setenv("CLONER_VOICES_DIR", str(tmp_path / "voices"))

    import importlib

    import app.config as config_module
    importlib.reload(config_module)
    import app.services.db as db_module
    importlib.reload(db_module)

    create_db_and_tables()

    from app.main import app as fastapi_app
    return TestClient(fastapi_app)


@pytest.fixture()
def tiny_wav_bytes():
    # Minimal valid WAV header + silence, enough for ffmpeg to accept.
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    buf.seek(0)
    return buf.read()
```

```python
# tests/test_voices.py
def test_upload_requires_label(client, tiny_wav_bytes):
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": ""},
    )
    assert res.status_code == 400


def test_upload_and_list_voice(client, tiny_wav_bytes):
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "My Voice"},
    )
    assert res.status_code == 200
    voice_id = res.json()["voice_id"]
    assert res.json()["label"] == "My Voice"

    res = client.get("/api/v1/voices")
    assert res.status_code == 200
    voices = res.json()
    assert len(voices) == 1
    assert voices[0]["id"] == voice_id
    assert voices[0]["label"] == "My Voice"


def test_delete_voice(client, tiny_wav_bytes):
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Temp"},
    )
    voice_id = res.json()["voice_id"]

    res = client.delete(f"/api/v1/voices/{voice_id}")
    assert res.status_code == 204

    res = client.get("/api/v1/voices")
    assert res.json() == []


def test_delete_unknown_voice_404(client):
    res = client.delete("/api/v1/voices/does-not-exist")
    assert res.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_voices.py -v`
Expected: FAIL — current router has no `label` field, no `GET`/`DELETE`.

- [ ] **Step 3: Add `voices_dir` env override support**

Confirm `app/config.py`'s `Settings` already exposes `voices_dir: Path`
with `env_prefix = "CLONER_"` — it does (existing code), so
`CLONER_VOICES_DIR` in the test fixture already works with no changes.

- [ ] **Step 4: Write minimal implementation**

```python
# app/routers/cloning.py
import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import settings
from app.models import Generation, GenerationStatus, Voice
from app.services.audio_processor import convert_to_wav
from app.services.db import get_session
from app.services.queue import GENERATIONS_DIR, enqueue
from app.services.tts_model import TTSModelManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["voices"])


@router.post("/voices/upload")
async def upload_voice(file: UploadFile, label: str = Form(...), session: Session = Depends(get_session)):
    if not file.filename:
        raise HTTPException(400, "No file provided")
    if not label or not label.strip():
        raise HTTPException(400, "Label is required")

    voice = Voice(label=label.strip(), filename=file.filename, wav_path="")
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
async def list_voices(session: Session = Depends(get_session)):
    voices = session.exec(select(Voice).order_by(Voice.created_at.desc())).all()
    return [
        {"id": v.id, "label": v.label, "filename": v.filename, "created_at": v.created_at.isoformat()}
        for v in voices
    ]


@router.delete("/voices/{voice_id}", status_code=204)
async def delete_voice(voice_id: str, session: Session = Depends(get_session)):
    voice = session.get(Voice, voice_id)
    if voice is None:
        raise HTTPException(404, f"Voice '{voice_id}' not found")

    voice_dir = settings.voices_dir / voice_id
    shutil.rmtree(voice_dir, ignore_errors=True)

    session.delete(voice)
    session.commit()
    return Response(status_code=204)
```

Note: `Generation`, `GenerationStatus`, `GENERATIONS_DIR`, `enqueue`,
`TTSModelManager` imports above are unused until Task 5 adds the
generation endpoints to this same file — leave the imports in place since
Task 5 modifies this file further, or omit them now and add in Task 5
(prefer omitting now to keep this task's diff self-contained; add them
back in Task 5's edit). For this task, drop the last three imports
(`Generation`, `GenerationStatus`, `GENERATIONS_DIR`, `enqueue`,
`TTSModelManager`) since they're not yet used.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_voices.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add app/routers/cloning.py tests/test_voices.py tests/conftest.py
git commit -m "feat: labeled voice upload with list/delete endpoints"
```

---

### Task 5: Generation endpoints

**Files:**
- Modify: `app/routers/cloning.py` (add generation endpoints)
- Modify: `app/main.py` (start DB + worker in `lifespan`)
- Test: `tests/test_generations.py`

**Interfaces:**
- Consumes: `enqueue`, `worker`, `process_one` from `app/services/queue.py`
  (Task 3). `Generation`, `GenerationStatus` from `app/models.py`.
- Produces: `POST /api/v1/generations` — body `{voice_id, text, language}`.
  404 if `voice_id` unknown. Returns `{generation_id, status: "pending"}`.
- Produces: `GET /api/v1/generations/{id}` — returns
  `{id, voice_id, text, language, status, error, created_at, completed_at}`.
  404 if not found.
- Produces: `GET /api/v1/generations/{id}/audio` — 200 + wav bytes if
  `status == "done"`. 409 with `{"status": <status>}` body otherwise. 404
  if generation not found.
- Produces: `GET /api/v1/generations?voice_id=...` — list, newest first,
  optionally filtered by `voice_id`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generations.py
from unittest.mock import patch


def _upload_voice(client, tiny_wav_bytes, label="Voice A"):
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": label},
    )
    return res.json()["voice_id"]


def test_generate_unknown_voice_404(client):
    res = client.post("/api/v1/generations", json={"voice_id": "nope", "text": "Mhoro", "language": "en"})
    assert res.status_code == 404


def test_generate_returns_pending_and_processes(client, tiny_wav_bytes):
    voice_id = _upload_voice(client, tiny_wav_bytes)

    with patch("app.services.tts_model.TTSModelManager.generate", return_value=b"FAKEWAV"):
        res = client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "Mhoro", "language": "en"})
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "pending"
        gen_id = body["generation_id"]

        # Drain the queue synchronously for the test (worker isn't running in TestClient).
        import asyncio
        from app.services.queue import process_one
        asyncio.get_event_loop().run_until_complete(process_one(gen_id))

    res = client.get(f"/api/v1/generations/{gen_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "done"

    res = client.get(f"/api/v1/generations/{gen_id}/audio")
    assert res.status_code == 200
    assert res.content == b"FAKEWAV"


def test_audio_not_ready_returns_409(client, tiny_wav_bytes):
    voice_id = _upload_voice(client, tiny_wav_bytes)
    res = client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "Mhoro", "language": "en"})
    gen_id = res.json()["generation_id"]

    res = client.get(f"/api/v1/generations/{gen_id}/audio")
    assert res.status_code == 409
    assert res.json()["status"] == "pending"


def test_list_generations_filtered_by_voice(client, tiny_wav_bytes):
    voice_a = _upload_voice(client, tiny_wav_bytes, label="A")
    voice_b = _upload_voice(client, tiny_wav_bytes, label="B")
    client.post("/api/v1/generations", json={"voice_id": voice_a, "text": "one", "language": "en"})
    client.post("/api/v1/generations", json={"voice_id": voice_b, "text": "two", "language": "en"})

    res = client.get(f"/api/v1/generations?voice_id={voice_a}")
    gens = res.json()
    assert len(gens) == 1
    assert gens[0]["voice_id"] == voice_a
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generations.py -v`
Expected: FAIL — `/api/v1/generations` doesn't exist yet (404 for wrong
reason / no route).

- [ ] **Step 3: Add generation endpoints to the router**

Append to `app/routers/cloning.py` (restore the imports dropped at the end
of Task 4 — `Generation`, `GenerationStatus`, `GENERATIONS_DIR`, `enqueue`
— they're used below):

```python
from fastapi.responses import FileResponse, JSONResponse

from app.models import Generation, GenerationStatus
from app.services.queue import enqueue


class GenerateRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "en"


@router.post("/generations")
async def create_generation(payload: GenerateRequest, session: Session = Depends(get_session)):
    voice = session.get(Voice, payload.voice_id)
    if voice is None:
        raise HTTPException(404, f"Voice '{payload.voice_id}' not found")

    gen = Generation(voice_id=payload.voice_id, text=payload.text, language=payload.language)
    session.add(gen)
    session.commit()
    session.refresh(gen)

    await enqueue(gen.id)
    return {"generation_id": gen.id, "status": gen.status}


@router.get("/generations/{generation_id}")
async def get_generation(generation_id: str, session: Session = Depends(get_session)):
    gen = session.get(Generation, generation_id)
    if gen is None:
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
async def get_generation_audio(generation_id: str, session: Session = Depends(get_session)):
    gen = session.get(Generation, generation_id)
    if gen is None:
        raise HTTPException(404, f"Generation '{generation_id}' not found")
    if gen.status != GenerationStatus.DONE:
        return JSONResponse(status_code=409, content={"status": gen.status})
    return FileResponse(gen.output_path, media_type="audio/wav")


@router.get("/generations")
async def list_generations(voice_id: str | None = None, session: Session = Depends(get_session)):
    query = select(Generation).order_by(Generation.created_at.desc())
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

Remove the old `/voices/generate` endpoint entirely (superseded by
`POST /generations`).

- [ ] **Step 4: Wire DB creation and worker startup into `lifespan`**

In `app/main.py`, modify the `lifespan` function:

```python
import asyncio

from app.services.db import create_db_and_tables
from app.services.queue import worker as queue_worker


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db_and_tables()
    threading.Thread(target=_warm_model_in_background, daemon=True).start()
    worker_task = asyncio.create_task(queue_worker())
    logger.info("Application ready at http://0.0.0.0:8000")
    yield
    worker_task.cancel()
    logger.info("Shutting down.")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_generations.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the full test suite**

Run: `uv run pytest -v`
Expected: all tests from Tasks 1-5 PASS.

- [ ] **Step 7: Commit**

```bash
git add app/routers/cloning.py app/main.py tests/test_generations.py
git commit -m "feat: async generation queue endpoints with status polling"
```

---

### Task 6: Frontend — voice picker, labels, polling, history

**Files:**
- Modify: `app/static/index.html`

**Interfaces:**
- Consumes: `GET /api/v1/voices`, `POST /api/v1/voices/upload` (now needs
  `label`), `POST /api/v1/generations`, `GET /api/v1/generations/{id}`,
  `GET /api/v1/generations/{id}/audio`, `GET /api/v1/generations` — all
  from Tasks 4-5.

No automated test for this task (static HTML/JS, no test harness in this
codebase for frontend) — verified manually per Step 3 below, consistent
with how `app/static/index.html` has been maintained so far.

- [ ] **Step 1: Add label input and voice picker markup**

In the "Step 1: Upload Voice Sample" card in `app/static/index.html`,
before the `.upload-zone` div, add:

```html
<div class="form-group" style="margin-bottom: 16px;">
  <label for="labelInput">Voice label <span style="color:var(--error)">*</span></label>
  <input type="text" id="labelInput" placeholder="e.g. Tariro — female, warm" style="width:100%; padding:9px 12px; border:1px solid var(--border); border-radius:8px; font-family:inherit; font-size:14px;" />
</div>
```

After the `.voice-id-box` div, add a saved-voices picker:

```html
<div class="form-group" style="margin-top:20px;">
  <label for="voicePicker">Or select a saved voice</label>
  <select id="voicePicker" aria-label="Saved voices">
    <option value="">— none selected —</option>
  </select>
</div>
```

Add a history card after the "Step 2: Generate Speech" card:

```html
<div class="card">
  <div class="card-title"><div class="step-badge">3</div> Generation History</div>
  <div id="historyList" style="display:flex; flex-direction:column; gap:10px;">
    <p class="helper">No generations yet.</p>
  </div>
</div>
```

- [ ] **Step 2: Replace the upload/generate JS with label + picker + polling + history**

Replace the `<script>` block's `uploadFile`, `generate`, and add new
functions. Full replacement of the script body between `let voiceId =
null;` and the closing `</script>`:

```html
<script>
  const ECOCASH_TEXT = "Chadzoka zvine mutsindo! Chakachaya/Ziyawa kuEcoCash Promotion. Wana mukana wekuhwina mibairo inosanganisira motokari gumi, holiday trip kuVictoria Falls, mombe masolar system, masmartphones, magrocery vouchers, EcoCash, madeep freezer, magas stoves, matelevision sets, masmartphones nemimwe mibairo mizhinji. Chako kushandisa EcoCash kuita Cash In, kutumira mari kuvadikani, kutumirwa mari kubva diaspora, kutenga airtime kana mabundles, kubhadhara kana kuita bank to wallet transfer. Ukawana 30 points zvichikwira pavhiki roga roga, watopinda mumakwikwi! Batika panohwina vamwe. Pinda muChaka-Chaya, Ziyawa kuEcoCash Promotion. Shandisa EcoCash Super App kana kuchaya *151# nhasi. EcoCash, ndiwo mararamiro edu!";

  let voiceId = null;
  let pollTimer = null;

  const fileInput = document.getElementById('fileInput');
  const uploadZone = document.getElementById('uploadZone');
  const voicePicker = document.getElementById('voicePicker');

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

  async function loadVoices(selectId) {
    const res = await fetch('/api/v1/voices');
    const voices = await res.json();
    voicePicker.innerHTML = '<option value="">— none selected —</option>' +
      voices.map(v => `<option value="${v.id}">${v.label}</option>`).join('');
    if (selectId) {
      voicePicker.value = selectId;
      selectVoice(selectId);
    }
  }

  voicePicker.addEventListener('change', () => selectVoice(voicePicker.value));

  function selectVoice(id) {
    voiceId = id || null;
    const status = document.getElementById('voiceStatus');
    const btn = document.getElementById('generateBtn');
    if (voiceId) {
      status.textContent = 'Voice ready';
      status.style.color = 'var(--success)';
      btn.disabled = false;
    } else {
      status.textContent = 'No voice selected';
      status.style.color = 'var(--text-secondary)';
      btn.disabled = true;
    }
  }

  async function uploadFile(file) {
    const label = document.getElementById('labelInput').value.trim();
    if (!label) { setStatus('uploadStatus', 'error', 'Enter a label for this voice first.'); return; }

    setStatus('uploadStatus', 'loading', `Uploading ${file.name}...`);

    const formData = new FormData();
    formData.append('file', file);
    formData.append('label', label);

    try {
      const res = await fetch('/api/v1/voices/upload', { method: 'POST', body: formData });
      const data = await res.json();

      if (!res.ok) throw new Error(data.detail || 'Upload failed');

      document.getElementById('voiceIdDisplay').textContent = data.voice_id;
      document.getElementById('voiceIdBox').classList.add('visible');
      clearStatus('uploadStatus');
      await loadVoices(data.voice_id);
    } catch (err) {
      setStatus('uploadStatus', 'error', err.message);
    }
  }

  async function generate() {
    const text = document.getElementById('textInput').value.trim();
    if (!text) { setStatus('generateStatus', 'error', 'Please enter text to synthesize.'); return; }
    if (!voiceId) { setStatus('generateStatus', 'error', 'Please select a voice first.'); return; }

    const language = document.getElementById('languageSelect').value;
    const btn = document.getElementById('generateBtn');
    btn.disabled = true;
    btn.innerHTML = '<div class="spinner"></div> Queuing...';
    document.getElementById('audioResult').classList.remove('visible');
    setStatus('generateStatus', 'loading', 'Queued for generation...');

    try {
      const res = await fetch('/api/v1/generations', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ voice_id: voiceId, text, language }),
      });
      if (!res.ok) throw new Error((await res.json()).detail || `Server error ${res.status}`);

      const { generation_id } = await res.json();
      pollGeneration(generation_id);
    } catch (err) {
      setStatus('generateStatus', 'error', err.message);
      resetGenerateBtn();
    }
  }

  function pollGeneration(generationId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      const res = await fetch(`/api/v1/generations/${generationId}`);
      const gen = await res.json();

      if (gen.status === 'pending' || gen.status === 'running') {
        setStatus('generateStatus', 'loading', `Status: ${gen.status}...`);
        return;
      }

      clearInterval(pollTimer);
      resetGenerateBtn();

      if (gen.status === 'done') {
        clearStatus('generateStatus');
        const audioRes = await fetch(`/api/v1/generations/${generationId}/audio`);
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

      loadHistory();
    }, 1500);
  }

  function resetGenerateBtn() {
    const btn = document.getElementById('generateBtn');
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"/></svg> Generate Audio`;
  }

  async function loadHistory() {
    const res = await fetch('/api/v1/generations');
    const gens = await res.json();
    const container = document.getElementById('historyList');
    if (gens.length === 0) {
      container.innerHTML = '<p class="helper">No generations yet.</p>';
      return;
    }
    container.innerHTML = gens.slice(0, 20).map(g => `
      <div style="display:flex; justify-content:space-between; align-items:center; padding:10px 12px; border:1px solid var(--border); border-radius:8px; font-size:13px;">
        <span style="overflow:hidden; text-overflow:ellipsis; white-space:nowrap; max-width:420px;">${g.text}</span>
        <span style="color:${g.status === 'done' ? 'var(--success)' : g.status === 'failed' ? 'var(--error)' : 'var(--text-secondary)'}; font-weight:600;">
          ${g.status}${g.status === 'done' ? ` — <a href="/api/v1/generations/${g.id}/audio" target="_blank">play</a>` : ''}
        </span>
      </div>
    `).join('');
  }

  function fillScript() {
    document.getElementById('textInput').value = ECOCASH_TEXT;
    updateCharCount();
  }

  function updateCharCount() {
    document.getElementById('charCount').textContent = document.getElementById('textInput').value.length;
  }

  function setStatus(id, type, msg) {
    const el = document.getElementById(id);
    el.className = `status-bar visible ${type}`;
    el.innerHTML = type === 'loading'
      ? `<div class="status-spinner"></div> ${msg}`
      : `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg> ${msg}`;
  }

  function clearStatus(id) {
    const el = document.getElementById(id);
    el.className = 'status-bar';
    el.innerHTML = '';
  }

  document.getElementById('textInput').addEventListener('keydown', e => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') generate();
  });

  // Initial load
  selectVoice('');
  loadVoices();
  loadHistory();
</script>
```

- [ ] **Step 3: Manual verification**

Run: `uv run uvicorn app.main:app --reload` then open `http://localhost:8000`.
- Upload a wav without a label → inline error, no request sent past the
  label check (or 400 from server if bypassed).
- Upload a wav with a label → appears in the "saved voice" picker,
  auto-selected.
- Reload the page → the voice is still listed (persisted in SQLite, unlike
  today's in-memory-only state).
- Enter text, click Generate → status shows `pending`/`running` while
  polling, then plays audio on `done`.
- Open two browser tabs, submit a generation in each within a few seconds
  of each other → confirm the second one's status stays `pending` until
  the first reaches `done`/`failed` (proves serialized GPU access).
- Generation history card shows past generations with clickable `play`
  links for completed ones.

- [ ] **Step 4: Commit**

```bash
git add app/static/index.html
git commit -m "feat: voice picker, labels, generation polling, and history in UI"
```

---

### Task 7: Storage directories in Docker image

**Files:**
- Modify: `Dockerfile`

**Interfaces:**
- Consumes: `storage/generations`, `storage/db.sqlite3` parent dir — paths
  introduced in Tasks 2-3.

- [ ] **Step 1: Update the `mkdir` line**

In `Dockerfile`, change:

```dockerfile
RUN mkdir -p /app/storage/voices /app/storage/models
```

to:

```dockerfile
RUN mkdir -p /app/storage/voices /app/storage/models /app/storage/generations
```

(The sqlite file itself is created at runtime by `create_db_and_tables()`,
same as `storage/matplotlib` already is by `tts_model.py`.)

- [ ] **Step 2: Verify the build**

Run: `docker build -t shona-voice-cloner-test .`
Expected: image builds successfully (this only needs to succeed, not run,
since GPU/model download aren't available in this build check).

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "chore: create storage/generations dir in Docker image"
```

---

## Self-Review Notes

- **Spec coverage:** Data model (Task 1), generation queue serializing GPU
  access (Task 3), voice label requirement + list/delete (Task 4),
  async generation endpoints + 409-on-not-ready (Task 5), frontend picker +
  polling + history (Task 6), Docker storage dirs (Task 7) — every spec
  section has a task.
- **Type consistency:** `GenerationStatus` constants (`PENDING`, `RUNNING`,
  `DONE`, `FAILED`) defined once in Task 1, reused verbatim in Tasks 3 and
  5 — no ad-hoc string literals elsewhere. `enqueue(generation_id: str)`
  signature matches its Task 5 call site. `process_one(generation_id: str)`
  matches its Task 3 definition and Task 5/6 test usage.
- **Import cleanup flagged:** Task 4's implementation step explicitly notes
  which imports to drop (since they're unused until Task 5) rather than
  leaving dead imports — avoided as a placeholder risk.
