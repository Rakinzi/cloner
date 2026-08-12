# Voice library + generation queue — design spec

Written 2026-08-12. Scope: turn the working Shona XTTS finetuned-inference
FastAPI app into a DB-backed voice library with labeled voices, a
generation queue that serializes GPU access, and generation history.

## Context

`app/main.py`, `app/routers/cloning.py`, and `app/services/tts_model.py`
already work end-to-end: upload a reference wav, call `TTSModelManager.generate()`
against the finetuned checkpoint (`CLONER_FINETUNED_MODEL_PATH`), get audio
back. Voices are anonymous UUID folders under `storage/voices/`, tracked only
in the browser's in-memory JS state — nothing persists across page loads,
there's no way to list or label saved voices, and nothing stops two
concurrent `/generate` calls from hitting the GPU (6 GB VRAM on the target
WSL box) at the same time.

This spec adds:
1. A SQLite-backed voice library — voices get a required user-provided label.
2. An async generation queue — one background worker serializes all calls
   into `TTSModelManager.generate()`, so concurrent requests queue instead
   of racing on the GPU.
3. Generation history — every synth request is a row, so past outputs are
   listable and re-downloadable.

Explicitly out of scope: auth/multi-user, Postgres, an external job broker
(Celery/Redis), rate limiting. Single-process, single-GPU, SQLite is
sufficient — inference time dominates, not DB access.

## Data model

New `sqlmodel` models in `app/models.py`, backed by `storage/db.sqlite3`.

```python
class Voice(SQLModel, table=True):
    id: str  # uuid, primary key
    label: str  # required, user-provided
    filename: str  # original upload filename
    wav_path: str  # storage/voices/{id}/voice.wav
    created_at: datetime

class Generation(SQLModel, table=True):
    id: str  # uuid, primary key
    voice_id: str  # FK -> Voice.id
    text: str
    language: str
    status: str  # "pending" | "running" | "done" | "failed"
    output_path: str | None  # storage/generations/{id}.wav, set when done
    error: str | None
    created_at: datetime
    completed_at: datetime | None
```

No migrations tooling (alembic) — the schema is small and stable enough
that `SQLModel.metadata.create_all(engine)` on startup is sufficient. If the
schema needs to change later, that's a good point to add alembic.

## Generation queue

- `app/services/queue.py`: a module-level `asyncio.Queue[str]` (generation
  ids) plus a single `worker()` coroutine.
- Started as an `asyncio.create_task` in the FastAPI `lifespan`, alongside
  the existing model warm-up thread.
- `POST /api/v1/generations`:
  1. Validates the voice exists.
  2. Inserts a `Generation` row with `status="pending"`.
  3. Puts the id on the queue.
  4. Returns `{generation_id, status: "pending"}` immediately (no waiting).
- The worker loop: pull one id, mark `running`, call
  `TTSModelManager.generate()` (unchanged — still fully synchronous/blocking,
  run via `asyncio.to_thread` so it doesn't block the event loop), write
  the wav to `storage/generations/{id}.wav`, mark `done` with `output_path`
  set, or `failed` with `error` set on exception. Because there is exactly
  one worker task, only one generation ever runs at a time — this is the
  entire concurrency guarantee, no locking needed elsewhere.
- `GET /api/v1/generations/{id}` — returns the row (status, text, error, etc).
- `GET /api/v1/generations/{id}/audio` — 404 unless `status == "done"`,
  otherwise streams `output_path`.
- `GET /api/v1/generations?voice_id=...` — list, newest first, for history.

## Voice endpoints (`app/routers/cloning.py`, rewritten)

- `POST /api/v1/voices/upload` — now takes `label` (form field, required,
  non-empty) alongside `file`. Creates a `Voice` row. Same 50 MB limit and
  `convert_to_wav` handling as today.
- `GET /api/v1/voices` — list all voices (id, label, filename, created_at),
  newest first.
- `DELETE /api/v1/voices/{id}` — deletes the row and `storage/voices/{id}/`.
  404 if not found.
- `POST /api/v1/generations` replaces the old `POST /api/v1/voices/generate`
  (moved under its own resource path since it's no longer "part of" a
  single voice operation — body is unchanged: `{voice_id, text, language}`).

## Frontend (`app/static/index.html`)

Current flow holds one `voiceId` in a JS variable and re-uploads every
session. New flow:

- Upload form gains a required "Label" text input; disabled submit until
  filled.
- A voice picker (dropdown or list) populated from `GET /api/v1/voices` on
  page load — select an existing voice instead of re-uploading every time.
  Uploading a new voice adds it to the list and selects it.
- Generate button calls `POST /api/v1/generations`, then polls
  `GET /api/v1/generations/{id}` (e.g. every 1.5s) showing
  pending/running/done/failed state, and on `done` plays/links
  `GET /api/v1/generations/{id}/audio`.
- A simple history panel: `GET /api/v1/generations` (optionally filtered to
  the selected voice), newest first, each row links to its audio if done.

## Config (`app/config.py`)

Add `db_path: Path = Path("./storage/db.sqlite3")`.

## Dependencies

Add `sqlmodel` to `pyproject.toml` and `requirements.txt`. No other new
dependencies — `asyncio.Queue` and `asyncio.to_thread` are stdlib.

## Error handling

- Upload: missing/empty label → 400. Oversized file → 413 (existing).
- Generate: unknown `voice_id` → 404 before enqueueing (fail fast, don't
  queue a job that can't succeed).
- Worker: any exception from `TTSModelManager.generate()` is caught, row
  set to `failed` with `str(exception)` in `error`, worker continues to the
  next queued item (one bad job never wedges the queue).
- Audio fetch on a non-`done` generation → 409 with the current status in
  the body (not 404 — the row exists, it's just not ready).

## Testing

- Manual: upload a labeled voice, generate against it, confirm the row
  transitions pending → running → done and audio plays. Submit two
  generations back-to-back, confirm the second visibly waits (status stays
  `pending` until the first finishes) rather than both running at once.
- Restart the app, confirm previously uploaded voices and past generations
  are still listed (DB persists across restarts, unlike today's
  browser-memory-only state).
