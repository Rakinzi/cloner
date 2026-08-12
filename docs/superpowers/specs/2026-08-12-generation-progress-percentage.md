# Generation progress percentage — design spec

Written 2026-08-12. Scope: replace the spinner/status-text-only generation UI
with a real, incrementally-updating progress percentage, backed by XTTS's
streaming inference API.

## Context

Today, `POST /api/v1/generations` enqueues a job; the worker
(`app/services/queue.py:process_one`) calls `TTSModelManager.generate()`,
which makes a single blocking call to `model.inference(...)` and only
returns once the entire wav is decoded. The frontend polls
`GET /api/v1/generations/{id}` every 1.5s and shows `Status: running...`
with a spinner — no indication of how much longer it will take, which
matters a lot on this project's CPU-only hardware (a single generation can
take minutes; the longest verified real case, video-log'd during
debugging, was ~15 minutes for a 320-character passage).

Investigation found that `Xtts.inference_stream(...)` (defined in
coqui-tts, used by the finetuned-checkpoint code path this app already
runs through) is a **generator** that yields decoded audio chunks
incrementally (chunks of `stream_chunk_size=20` GPT tokens by default,
each roughly 1-2 seconds of audio) rather than blocking until the whole
utterance is done. This gives a genuine, non-fabricated signal to report
progress from — a "chunks generated so far" counter — even though XTTS
itself never exposes a hard percentage or total-duration figure.

## Approach: text-length-based percentage estimate

Since neither `inference` nor `inference_stream` know the total output
duration in advance, progress is *estimated*, not exact:

1. On enqueue, estimate total generation wall-clock time as
   `len(text) * SECONDS_PER_CHAR`, a tunable module-level constant in
   `app/services/tts_model.py`.
2. As chunks stream in, track cumulative **audio-seconds produced so far**
   (each chunk's sample count / sample rate).
3. Progress = `min(1.0, audio_seconds_so_far / estimated_total_audio_seconds)`,
   where `estimated_total_audio_seconds` is derived from the same
   `SECONDS_PER_CHAR`-style heuristic applied to expected *output* duration
   (not wall-clock) — see Task-level detail in the implementation plan for
   the exact formula, since output-duration-per-char and
   wall-clock-time-per-char are different constants and must not be
   conflated.
4. This is explicitly an estimate: it can overshoot 100% before the
   generation is actually marked `done` (clamped to 99% while still
   `running`, jumping to 100%/`done` only on actual completion) and can
   undershoot for text with unusual pacing. It is still strictly better
   than no signal at all, and the constant is easy to retune later against
   real generation logs.

**Calibration note:** one real data point exists from prior debugging — a
320-character passage took ~907s wall-clock on this CPU and produced
~25s of audio (before the 250-char XTTS truncation warning kicked in,
so treat this as directional, not authoritative). The implementation
should start from a conservative constant and log actual
`(text_len, wall_clock_elapsed, audio_seconds_produced)` tuples at
completion so the constant can be retuned from real usage without
guessing again.

## Data model (`app/models.py`)

Add to `Generation`:

```python
    progress: float = Field(default=0.0)
```

Semantics: `0.0` while `pending`; updates incrementally while `running`;
implicitly `1.0` once `status == "done"` (not required to be literally
written as 1.0, but the worker will set it to 1.0 for consistency so
clients never need special-case `done` to know the bar should be full).
Meaningless (stays whatever it last was, likely 0.0) for `failed`.

## Streaming generation (`app/services/tts_model.py`)

Add `TTSModelManager.generate_streaming(text, voice_path, language, on_chunk)`:

- Only available on the finetuned-checkpoint code path (`Xtts` model with
  `inference_stream`). The stock `TTS.api.TTS` object used when
  `settings.finetuned_model_path` is empty has no streaming equivalent in
  this codebase's dependency version — that path keeps calling the
  existing blocking `generate()`, and progress simply stays at the
  pre-existing spinner/status-text behavior (no percentage) for it. This
  is an intentional scope boundary, not a gap to fix now — the project
  only actually runs the finetuned path in practice (`.env` sets
  `CLONER_FINETUNED_MODEL_PATH`).
- Computes conditioning latents once (same as today), then iterates
  `model.inference_stream(text=..., language=..., gpt_cond_latent=...,
  speaker_embedding=...)`, calling `on_chunk(chunk_wav_tensor)` for each
  yielded chunk and collecting chunks into a list for final concatenation.
- Returns the final concatenated wav bytes (same int16 conversion /
  `scipy.io.wavfile` encoding as today's `generate()`), so the output file
  format and the `/audio` endpoint are byte-for-byte unaffected.
- `on_chunk` is synchronous (called from the worker thread via
  `asyncio.to_thread`, same threading model as today) — it is the
  worker's job to marshal the callback back into an async DB write, not
  this method's.

## Worker (`app/services/queue.py`)

`process_one` changes its generation call:

- If `settings.finetuned_model_path` is set: call `generate_streaming`
  with an `on_chunk` callback that computes
  `audio_seconds_so_far += len(chunk) / sample_rate`, estimates
  `progress = min(0.99, audio_seconds_so_far / estimated_output_seconds)`,
  and writes `Generation.progress` to the DB — **every chunk**, per the
  approved design (no throttling; this box runs one generation at a time
  by construction, so there's no write contention to economize for).
- If not set (base model, no finetuned checkpoint): unchanged — calls the
  existing blocking `generate()`, `progress` stays `0.0` throughout
  `running` (frontend falls back to spinner+text for this case, per the
  Frontend section below).
- On completion (either path): set `status = DONE`, `progress = 1.0`,
  write the output file — unchanged from today otherwise.
- On failure: unchanged (`status = FAILED`, `error` set); `progress` is
  left at whatever it last reached (not reset to 0), since a failed
  generation's progress value is never displayed once failed (frontend
  only reads `progress` while `status == "running"`).

## API (`app/routers/cloning.py`)

`GET /generations/{id}` response gains one field:

```python
"progress": gen.progress,
```

No new endpoint. `POST /generations` and `GET /generations` (the list)
are unchanged — progress is only meaningful for a single in-flight
generation, not the history list.

## Frontend (`app/static/generate.html`)

- Replace the current `setStatus('generateStatus', 'loading', 'Status:
  running...')` text-only line with a real progress bar element
  (`<progress>` HTML element, styled via `portal.css` to match the
  existing token system) shown only while `status === 'running'` **and**
  `progress > 0` (i.e., the finetuned/streaming path). While
  `status === 'pending'` or while `progress` stays `0` during `running`
  (base-model fallback case), keep today's spinner+text — no visual
  regression for that path.
- Percentage text alongside the bar (e.g. "Generating... 42%"), updating
  each poll tick same as today (poll interval unchanged at 1.5s — the DB
  is updated more frequently than that by the worker, but the UI doesn't
  need to poll faster than it already does; the bar will simply show the
  latest value each tick, which reads as smooth given how many chunks
  land per 1.5s window).
- No change to `voices.html` or `history.html` — progress is
  generate-page-only, matching where the live polling loop already lives.

## Testing

- Unit: `generate_streaming` with a mocked `inference_stream` generator
  yielding fixed-size fake chunks, asserting `on_chunk` is called once per
  yielded chunk and the concatenated output matches the mocked chunks
  joined together.
- Unit: the worker's chunk-to-progress calculation (`audio_seconds_so_far
  / estimated_total`), including the `min(0.99, ...)` clamp — assert a
  generation that "overshoots" its estimate never reports `progress >=
  1.0` while still `running`.
- Integration: extend the existing `tests/test_generations.py` /
  `tests/test_queue.py` patterns — patch
  `TTSModelManager.generate_streaming` to yield a few fake chunks, drive
  `process_one`, assert `Generation.progress` was written multiple times
  (via reading it between chunks, or asserting the final DB row has
  `progress == 1.0` at minimum) and the final `output_path` wav is the
  concatenation of the mocked chunks.
- Manual: run a real generation with a moderately long sentence, confirm
  the progress bar visibly advances across multiple poll ticks rather
  than jumping straight from 0 to 100%.

## Explicitly out of scope

- Early/partial audio playback before generation finishes (streaming the
  audio itself to the browser) — a separate, larger project per the
  brainstorming discussion.
- Self-tuning the `SECONDS_PER_CHAR` constant from historical generation
  data — the spec calls for logging the data needed to retune it by hand
  later, not an automatic learning system now.
- Progress display on `voices.html` / `history.html`.
- Progress for the base (non-finetuned) model code path beyond today's
  existing spinner/status-text behavior.
