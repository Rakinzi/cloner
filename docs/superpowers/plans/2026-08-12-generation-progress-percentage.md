# Generation Progress Percentage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the spinner-only "Status: running..." generation UI with
a real, incrementally-updating progress percentage, driven by XTTS's
streaming inference API on the finetuned-checkpoint code path.

**Architecture:** `Generation` gains a `progress: float` column. A new
`TTSModelManager.generate_streaming()` method wraps `Xtts.inference_stream`
(a generator yielding audio chunks as they decode) and invokes a per-chunk
callback with cumulative audio-seconds produced. The worker
(`process_one`) uses this on the finetuned path to write `progress` to the
DB after every chunk, estimating the total from
`len(text) * SECONDS_PER_OUTPUT_AUDIO_SECOND_PER_CHAR`-style constants.
The existing blocking `generate()` method is untouched and still used
verbatim on the non-finetuned (base model) path, where progress stays at
today's spinner-only behavior. The frontend polls the same endpoint as
today (no interval change) and renders a `<progress>` bar when
`progress > 0`, falling back to the existing spinner/text otherwise.

**Tech Stack:** FastAPI, sqlmodel/SQLite (existing), coqui-tts's
`Xtts.inference_stream` (existing dependency, new API surface used), pytest
(existing).

## Global Constraints

- `TTSModelManager.generate()` (the existing blocking method) must remain
  byte-for-byte unchanged — existing tests patch it directly
  (`tests/test_queue.py`, `tests/test_generations.py`), and it stays the
  only code path for the non-finetuned (base) model.
- `generate_streaming` is only invoked when `settings.finetuned_model_path`
  is set. When unset, `process_one` calls the existing `generate()` exactly
  as today — no behavior change for that path, including `progress`
  staying `0.0` throughout.
- Progress must never report `>= 1.0` while `status` is still `running` —
  clamp to `0.99` until the generation actually completes, per spec.
- Every chunk yielded by `inference_stream` writes `Generation.progress`
  to the DB — no throttling, per the approved design (single-worker,
  single-generation-at-a-time, so no write contention to economize for).
- The final output wav file must be byte-identical in format/encoding to
  today's (concatenated int16 PCM via `scipy.io.wavfile`, 24000 Hz) — the
  `/audio` endpoint and download behavior are unaffected.
- No change to `voices.html`, `history.html`, or the `GET /generations`
  (list) endpoint — progress is single-generation, generate-page-only.
- No self-tuning of the timing constants from historical data — log the
  `(text_len, wall_clock_elapsed, audio_seconds_produced)` tuple on
  completion so the constants can be retuned by hand later.

---

## File Structure

- Modify `app/models.py` — add `Generation.progress`.
- Modify `app/services/tts_model.py` — add `generate_streaming()` and the
  timing constants; `generate()` untouched.
- Modify `app/services/queue.py` — branch `process_one` on
  `settings.finetuned_model_path` to call `generate_streaming` (with a
  progress-writing callback) or the existing `generate()`.
- Modify `app/routers/cloning.py` — add `progress` to the
  `GET /generations/{id}` response.
- Modify `app/static/generate.html` — render a progress bar when
  `progress > 0`, keep spinner/text fallback otherwise.
- Modify `app/static/portal.css` — style the new progress bar with
  existing tokens.
- Modify `tests/test_models.py`, `tests/test_queue.py`,
  `tests/test_generations.py` — extend for the new field/behavior.

---

### Task 1: `Generation.progress` field

**Files:**
- Modify: `app/models.py`
- Test: `tests/test_models.py` (extend)

**Interfaces:**
- Produces: `Generation.progress: float` (default `0.0`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_models.py — add to the existing file
def test_generation_defaults_progress_to_zero():
    gen = Generation(id="g1", voice_id="v1", text="Mhoro", language="en", user_id="u1")
    assert gen.progress == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `AttributeError: 'Generation' object has no attribute 'progress'`

- [ ] **Step 3: Write minimal implementation**

In `app/models.py`, add to the `Generation` class (after `status`, before
`output_path`):

```python
    progress: float = Field(default=0.0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_models.py -v`
Expected: PASS (all tests in the file, including the new one)

- [ ] **Step 5: Delete the stale dev DB so the new column applies**

The project's `storage/db.sqlite3` (if present) predates this column and
`create_db_and_tables()` never alters existing tables (this exact failure
mode already happened once for `user_id` — see project history). Confirm
before deleting:

Run: `sqlite3 storage/db.sqlite3 "SELECT COUNT(*) FROM generation;" 2>&1 || echo "no db file, nothing to do"`

If the file exists and the count is a real number the user cares about,
STOP and ask before deleting. Otherwise:

Run: `rm -f storage/db.sqlite3`

- [ ] **Step 6: Commit**

```bash
git add app/models.py tests/test_models.py
git commit -m "feat: add progress field to Generation model"
```

---

### Task 2: Streaming generation in TTSModelManager

**Files:**
- Modify: `app/services/tts_model.py`
- Test: `tests/test_tts_model_streaming.py`

**Interfaces:**
- Consumes: `Xtts.inference_stream(text, language, gpt_cond_latent,
  speaker_embedding, ...)` — a generator yielding `torch.Tensor` audio
  chunks (confirmed via source inspection: coqui-tts 0.27.5,
  `TTS/tts/models/xtts.py`).
- Produces: `TTSModelManager.generate_streaming(text: str, voice_path: str,
  language: str, on_chunk: Callable[[int], None]) -> bytes` — a new
  method. `on_chunk` receives the **cumulative audio seconds produced so
  far** (a `float`) after each chunk; return value is the final
  concatenated wav bytes, encoded identically to `generate()` (int16 PCM
  via `scipy.io.wavfile.write`, rate=24000).
- Produces: module-level constant
  `OUTPUT_SECONDS_PER_CHAR = 0.22` in `app/services/tts_model.py` — the
  estimated ratio of output *audio* duration to input character count
  (derived from two real verified generations: "Mangwanani" → 2.18s/10
  chars = 0.218, and a 5-char sample → 1.29s/5 chars = 0.258; 0.22 is a
  rounded midpoint). This is a distinct constant from any wall-clock
  timing estimate — it estimates **output audio length**, which is what
  chunks are measured against, not generation wall-clock time.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_tts_model_streaming.py
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from app.services.tts_model import OUTPUT_SECONDS_PER_CHAR, TTSModelManager


def _fake_chunks(n=3, samples_per_chunk=4800):
    # 4800 samples @ 24000Hz = 0.2s per chunk
    return [torch.zeros(samples_per_chunk) for _ in range(n)]


def test_generate_streaming_calls_on_chunk_per_yielded_chunk():
    manager = TTSModelManager()
    fake_model = MagicMock()
    fake_model.get_conditioning_latents.return_value = ("latent", "embedding")
    fake_model.inference_stream.return_value = iter(_fake_chunks(3))

    seen_progress = []

    with patch.object(TTSModelManager, "model", new=fake_model):
        with patch("app.services.tts_model.settings") as mock_settings:
            mock_settings.finetuned_model_path = "fake.pth"
            result = manager.generate_streaming(
                text="Mhoro",
                voice_path="fake.wav",
                language="en",
                on_chunk=lambda seconds_so_far: seen_progress.append(seconds_so_far),
            )

    assert len(seen_progress) == 3
    # Cumulative, strictly increasing
    assert seen_progress[0] < seen_progress[1] < seen_progress[2]
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_output_seconds_per_char_is_positive_constant():
    assert OUTPUT_SECONDS_PER_CHAR > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_tts_model_streaming.py -v`
Expected: FAIL with `ImportError: cannot import name 'generate_streaming'`
or `'OUTPUT_SECONDS_PER_CHAR'` (whichever import fails first).

- [ ] **Step 3: Write minimal implementation**

In `app/services/tts_model.py`, add near the top (after the existing
imports, before the `TTSModelManager` class):

```python
# Estimated output-audio-seconds per input character, derived from two
# verified real generations ("Mangwanani" -> 2.18s/10 chars = 0.218;
# a 5-char sample -> 1.29s/5 chars = 0.258). Used only to estimate a
# generation's total expected duration for progress reporting — retune
# by hand from real (text_len, audio_seconds_produced) log lines as more
# data accumulates. Not a wall-clock timing estimate.
OUTPUT_SECONDS_PER_CHAR = 0.22
```

Add the new method to `TTSModelManager` (after the existing `generate`
method):

```python
    def generate_streaming(self, text: str, voice_path: str, language: str, on_chunk) -> bytes:
        """Stream-generate audio via Xtts.inference_stream, invoking on_chunk(seconds_so_far)
        after every decoded chunk. Only valid on the finetuned-checkpoint code path — the
        stock TTS.api.TTS object has no streaming equivalent in this codebase."""
        import io

        import torch
        from scipy.io import wavfile

        model = self.model  # triggers load_model() if needed
        sample_rate = 24000

        with torch.inference_mode():
            gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[voice_path])

            chunks = []
            seconds_so_far = 0.0
            for chunk in model.inference_stream(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
            ):
                chunks.append(chunk)
                seconds_so_far += len(chunk) / sample_rate
                on_chunk(seconds_so_far)

            full_wav = torch.cat(chunks, dim=-1) if chunks else torch.zeros(0)
            wav_tensor = full_wav.to("cpu")
            wav_int = (wav_tensor * 32767).to(torch.int16).numpy()

            buf = io.BytesIO()
            wavfile.write(buf, rate=sample_rate, data=wav_int)
            buf.seek(0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return buf.getvalue()
```

Note: the test patches `TTSModelManager.model` as a property and
`app.services.tts_model.settings` as a module-level mock — this method
itself does not reference `settings` directly (the finetuned-vs-base
branch lives in the worker, Task 3), so the `mock_settings` patch in the
test is there only because `TTSModelManager.model` (the real property)
would otherwise trigger `load_model()`, which reads `settings`. Since the
test patches `model` directly via `patch.object`, this guards against any
accidental fallthrough — keep it in the test as written.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_tts_model_streaming.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Run the full existing test suite to confirm no regression**

Run: `uv run pytest tests/ -v`
Expected: all previously-passing tests still PASS — `generate()` itself
was not modified, so `tests/test_queue.py` and `tests/test_generations.py`
(which patch `TTSModelManager.generate`) must be unaffected.

- [ ] **Step 6: Commit**

```bash
git add app/services/tts_model.py tests/test_tts_model_streaming.py
git commit -m "feat: add streaming generation with per-chunk progress callback"
```

---

### Task 3: Wire progress into the worker

**Files:**
- Modify: `app/services/queue.py`
- Test: `tests/test_queue.py` (extend)

**Interfaces:**
- Consumes: `TTSModelManager.generate_streaming` (Task 2),
  `OUTPUT_SECONDS_PER_CHAR` (Task 2), `Generation.progress` (Task 1),
  `settings.finetuned_model_path` (existing, from `app.config`).
- Changes `process_one`'s generation-call branch: if
  `settings.finetuned_model_path` is set, calls `generate_streaming` with
  a callback that writes `Generation.progress` to the DB on every chunk,
  clamped to `min(0.99, seconds_so_far / estimated_total_seconds)`. If
  not set, calls the existing `generate()` unchanged (progress stays at
  its default `0.0`).
- On completion, sets `progress = 1.0` alongside the existing
  `status = DONE` write (both paths).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_queue.py — add to the existing file
from unittest.mock import patch

from app.config import settings


@pytest.mark.asyncio
async def test_process_one_updates_progress_on_streaming_path(tmp_path, _fresh_db):
    voice_id, gen_id = _make_voice_and_generation(_fresh_db, text="Mhoro zvenyu")

    def fake_generate_streaming(text, voice_path, language, on_chunk):
        on_chunk(0.5)
        on_chunk(1.0)
        on_chunk(1.5)
        return b"FAKEWAV_STREAMED"

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch.object(settings, "finetuned_model_path", "fake.pth"):
            with patch(
                "app.services.tts_model.TTSModelManager.generate_streaming",
                side_effect=fake_generate_streaming,
            ):
                await queue_module.process_one(gen_id)

    with Session(_fresh_db) as session:
        gen = session.get(Generation, gen_id)
        assert gen.status == GenerationStatus.DONE
        assert gen.progress == 1.0
        assert (tmp_path / f"{gen_id}.wav").read_bytes() == b"FAKEWAV_STREAMED"


@pytest.mark.asyncio
async def test_process_one_progress_never_reaches_one_while_running(tmp_path, _fresh_db):
    voice_id, gen_id = _make_voice_and_generation(_fresh_db, text="a")  # 1 char, tiny estimate

    captured_progress = []

    def fake_generate_streaming(text, voice_path, language, on_chunk):
        # Deliberately overshoot the estimated total to prove the clamp holds.
        on_chunk(10.0)
        with Session(_fresh_db) as session:
            captured_progress.append(session.get(Generation, gen_id).progress)
        return b"FAKEWAV"

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch.object(settings, "finetuned_model_path", "fake.pth"):
            with patch(
                "app.services.tts_model.TTSModelManager.generate_streaming",
                side_effect=fake_generate_streaming,
            ):
                await queue_module.process_one(gen_id)

    assert captured_progress[0] <= 0.99


@pytest.mark.asyncio
async def test_process_one_base_model_path_unaffected(tmp_path, _fresh_db):
    voice_id, gen_id = _make_voice_and_generation(_fresh_db)

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch.object(settings, "finetuned_model_path", ""):
            with patch("app.services.tts_model.TTSModelManager.generate", return_value=b"FAKEWAV"):
                await queue_module.process_one(gen_id)

    with Session(_fresh_db) as session:
        gen = session.get(Generation, gen_id)
        assert gen.status == GenerationStatus.DONE
        assert gen.progress == 1.0  # set on completion regardless of path
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_queue.py -v`
Expected: the three new tests FAIL — `process_one` doesn't yet branch on
`finetuned_model_path` or call `generate_streaming`. Existing tests in
this file still PASS (unmodified behavior for the code they cover).

- [ ] **Step 3: Write minimal implementation**

Replace `app/services/queue.py`'s `process_one` function body (the
`try:` block that currently calls
`await asyncio.to_thread(manager.generate, text, voice_path, language)`)
with:

```python
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

    with Session(engine) as session:
        voice = session.get(Voice, voice_id)
        voice_path = voice.wav_path

    manager = TTSModelManager()

    def _write_progress(progress: float) -> None:
        with Session(engine) as session:
            gen = session.get(Generation, generation_id)
            if gen is not None:
                gen.progress = progress
                session.add(gen)
                session.commit()

    try:
        if settings.finetuned_model_path:
            estimated_total_seconds = max(len(text) * OUTPUT_SECONDS_PER_CHAR, 0.01)

            def on_chunk(seconds_so_far: float) -> None:
                progress = min(0.99, seconds_so_far / estimated_total_seconds)
                _write_progress(progress)

            wav_bytes = await asyncio.to_thread(
                manager.generate_streaming, text, voice_path, language, on_chunk
            )
        else:
            wav_bytes = await asyncio.to_thread(manager.generate, text, voice_path, language)

        GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = GENERATIONS_DIR / f"{generation_id}.wav"
        output_path.write_bytes(wav_bytes)

        with Session(engine) as session:
            gen = session.get(Generation, generation_id)
            gen.status = GenerationStatus.DONE
            gen.progress = 1.0
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
```

Add the required imports at the top of `app/services/queue.py`:

```python
from app.config import settings
from app.services.tts_model import OUTPUT_SECONDS_PER_CHAR, TTSModelManager
```

(`TTSModelManager` is likely already imported — check before duplicating;
`settings` and `OUTPUT_SECONDS_PER_CHAR` are new.)

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_queue.py -v`
Expected: PASS (all tests in the file, old and new)

- [ ] **Step 5: Run the full suite**

Run: `uv run pytest tests/ -v`
Expected: all tests PASS, no regressions in `tests/test_generations.py`
(base-model path via mocked `generate` still works identically).

- [ ] **Step 6: Commit**

```bash
git add app/services/queue.py tests/test_queue.py
git commit -m "feat: write incremental progress during streaming generation"
```

---

### Task 4: Expose progress via the API

**Files:**
- Modify: `app/routers/cloning.py`
- Test: `tests/test_generations.py` (extend)

**Interfaces:**
- Consumes: `Generation.progress` (Task 1).
- Changes: `GET /generations/{generation_id}` response dict gains
  `"progress": gen.progress`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generations.py — add to the existing file
def test_generation_response_includes_progress(client, tiny_wav_bytes):
    register_and_login(client)
    voice_id = _upload_voice(client, tiny_wav_bytes)
    res = client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "Mhoro", "language": "en"})
    gen_id = res.json()["generation_id"]

    res = client.get(f"/api/v1/generations/{gen_id}")
    assert res.status_code == 200
    assert "progress" in res.json()
    assert res.json()["progress"] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_generations.py -v`
Expected: FAIL — `test_generation_response_includes_progress` fails with
a `KeyError`/`assert "progress" in {...}` failure; other tests in the
file still PASS.

- [ ] **Step 3: Write minimal implementation**

In `app/routers/cloning.py`, in `get_generation`'s return dict, add one
line:

```python
    return {
        "id": gen.id,
        "voice_id": gen.voice_id,
        "text": gen.text,
        "language": gen.language,
        "status": gen.status,
        "progress": gen.progress,
        "error": gen.error,
        "created_at": gen.created_at.isoformat(),
        "completed_at": gen.completed_at.isoformat() if gen.completed_at else None,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_generations.py -v`
Expected: PASS (all tests in the file)

- [ ] **Step 5: Commit**

```bash
git add app/routers/cloning.py tests/test_generations.py
git commit -m "feat: expose generation progress via the status endpoint"
```

---

### Task 5: Progress bar in the Generate page

**Files:**
- Modify: `app/static/generate.html`
- Modify: `app/static/portal.css`

**Interfaces:**
- Consumes: `gen.progress` from `GET /api/v1/generations/{id}` (Task 4).

No automated test — this is DOM/browser-only glue with no test harness in
this codebase, consistent with how the rest of `generate.html`'s
polling/rendering logic has been handled. Verified manually in Step 4.

- [ ] **Step 1: Add the progress bar markup**

In `app/static/generate.html`, replace the single status-bar div:

```html
    <div class="status-bar" id="generateStatus" role="status" aria-live="polite"></div>
```

with:

```html
    <div class="status-bar" id="generateStatus" role="status" aria-live="polite"></div>
    <div class="progress-container" id="progressContainer" hidden>
      <progress id="generateProgress" max="100" value="0"></progress>
      <span class="helper" id="progressLabel">0%</span>
    </div>
```

- [ ] **Step 2: Add progress bar styling to `portal.css`**

Append:

```css
.progress-container {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 12px;
}
.progress-container progress {
  flex: 1;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  appearance: none;
}
.progress-container progress::-webkit-progress-bar { background: var(--border); border-radius: 4px; }
.progress-container progress::-webkit-progress-value { background: var(--accent); border-radius: 4px; transition: width 300ms ease; }
.progress-container progress::-moz-progress-bar { background: var(--accent); border-radius: 4px; }
.progress-container #progressLabel { font-variant-numeric: tabular-nums; min-width: 36px; text-align: right; }
```

- [ ] **Step 3: Update `pollGeneration` to drive the bar**

Replace the `pollGeneration` function in `app/static/generate.html`:

```javascript
  function pollGeneration(generationId) {
    if (pollTimer) clearInterval(pollTimer);
    pollTimer = setInterval(async () => {
      const res = await apiFetch(`/api/v1/generations/${generationId}`);
      const gen = await res.json();

      if (gen.status === 'pending' || gen.status === 'running') {
        if (gen.progress > 0) {
          document.getElementById('generateProgress').value = Math.round(gen.progress * 100);
          document.getElementById('progressLabel').textContent = `${Math.round(gen.progress * 100)}%`;
          document.getElementById('progressContainer').hidden = false;
          clearStatus('generateStatus');
        } else {
          document.getElementById('progressContainer').hidden = true;
          setStatus('generateStatus', 'loading', `Status: ${gen.status}...`);
        }
        return;
      }

      clearInterval(pollTimer);
      document.getElementById('generateBtn').disabled = false;
      document.getElementById('progressContainer').hidden = true;

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
```

Also reset the progress bar at the start of `generate()` — in the
existing `generate()` function, right after
`document.getElementById('audioResult').classList.remove('visible');`,
add:

```javascript
    document.getElementById('progressContainer').hidden = true;
    document.getElementById('generateProgress').value = 0;
```

- [ ] **Step 4: Manual verification**

Run: `uv run uvicorn app.main:app --reload` (with a real
`CLONER_FINETUNED_MODEL_PATH` set, matching the project's `.env`).
- Log in, go to `/generate`, select a voice, enter a few words of Shona
  text, click Generate.
- Confirm: initially shows the spinner/"Status: pending..." (progress is
  0 before the worker picks it up), then within a couple of poll ticks
  switches to the progress bar advancing from a low percentage upward as
  the worker streams chunks, then completes and plays audio exactly as
  before.
- Confirm dark mode: toggle the sidebar theme switch, repeat the above,
  confirm the progress bar's fill color and track are both legible against
  the dark background (uses `var(--accent)` / `var(--border)`, already
  theme-aware).

- [ ] **Step 5: Commit**

```bash
git add app/static/generate.html app/static/portal.css
git commit -m "feat: show real-time progress bar during generation"
```

---

## Self-Review Notes

- **Spec coverage:** `Generation.progress` field (Task 1), streaming
  generation with per-chunk callback (Task 2), worker branching + clamp +
  every-chunk DB writes (Task 3), API exposure (Task 4), frontend bar with
  base-model fallback preserved (Task 5) — every spec section maps to a
  task.
- **Type/interface consistency:** `generate_streaming(text, voice_path,
  language, on_chunk)` signature defined in Task 2 is called identically
  in Task 3's `process_one`. `OUTPUT_SECONDS_PER_CHAR` is defined once in
  Task 2 and consumed only in Task 3 — no duplicate constant. `on_chunk`'s
  contract (receives cumulative seconds, not a raw chunk) is stated
  identically in both tasks' Interfaces sections.
- **No placeholders:** every step has complete, runnable code — the
  clamp logic, the DB write helper, and the frontend polling changes are
  all fully written out, not described.
- **Global Constraints honored in each task:** Task 2 leaves `generate()`
  untouched (verified by Task 2 Step 5 running the full existing suite
  before Task 3 even starts touching the worker). Task 3's base-model
  branch is a literal no-change call to the existing method. Task 3's
  clamp test (`test_process_one_progress_never_reaches_one_while_running`)
  directly enforces the "never >= 1.0 while running" constraint. Task 3
  writes progress on every single `on_chunk` call, no batching.
