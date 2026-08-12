# CLAUDE.md — Shona XTTS v2 (handoff notes)

Written 2026-08-05, updated 2026-08-12. Session memory does not travel between
machines — this file is the source of truth when working on this repo elsewhere
(e.g., the home Windows/WSL GPU box). Keep it updated when facts change.

## Status: TRAINING IS DONE ✅ (2026-08-06, Colab T4)

- 10 epochs, ~65,420 steps, eval loss fell at the end of **every** epoch
  (mel_ce 3.157 → 3.126 over the last five) — generalizing, no overfit.
- **The model** (both files required for inference), on the owner's Google Drive:
  `MyDrive/shona_xtts/checkpoints/shona_xtts/shona_xtts_v2-August-06-2026_06+47AM-9f83158/`
  → `best_model_65420.pth` (~5 GB) + `config.json`.
- Listening verdict: deep, imposing voice (~82 Hz median F0 — owner calls it
  "Madara"). Current work is **delivery/prosody tuning at inference time**, not
  retraining.

## Working rules for Claude sessions (owner's standing instructions)

- Prosody/inference work lives on branch **`prosody-controls`** (base commit
  65c8a71), NOT `main`.
- `main`'s working tree carries **uncommitted portal/auth work** (`app/models.py`,
  `app/routers/auth.py`, `app/services/{auth,db,queue}.py`, `app/static/*`,
  modified `Dockerfile`/`pyproject.toml`/...). **Never stage or commit those.**
  Always `git add` by exact file path; never `git add -A`/`.`.
- Don't push unless the owner says so. One task per session-ask; name files;
  state the acceptance test.
- README.md is the owner-facing usage guide; this file is the Claude-facing one.

## Inference & prosody controls (`scripts/infer_xtts.py`)

```bash
python scripts/infer_xtts.py \
  --checkpoint .../best_model_65420.pth --run-config .../config.json \
  --speaker-wav <clean 3-12s dataset clip> --temperature 0.75 --seed 42 \
  --sentence "Ndinotenda zvikuru | uye ndinovimba | tichashanda pamwe zvakare."
```

- `|` marks a dramatic pause: chunks are synthesized separately with the SAME
  conditioning latents, RMS edge-trimmed, stitched with `--pause-ms` silence
  (default 900 ms — measured median of a real Madara speech clip).
- `--auto-pause` inserts markers before conjunctions uye/asi/kana/nekuti/nokuti/saka.
- `--temperature/--top-p/--repetition-penalty/--speed` forwarded only when set;
  `--seed` for fair A/B. Targets from acoustic analysis: pauses 800–1400 ms,
  phrases 1–2.5 s, pitch spread ~0.6 octaves (untuned output was 0.31).
- **Phase B (pending, gated on ear tests)**: wire tuned defaults into the portal —
  `app/services/tts_model.py` `generate()`/`generate_streaming()`, cloning router,
  `generate.html`. Plan: `~/.claude/plans/silly-crunching-puffin.md`. NB:
  `tts_model.py` is dirty on main from portal work — flag before committing it.

## Non-negotiable facts (verified 2026-08-05 against coqui-tts 0.27.5 / coqui-tts-trainer 0.3.3 source — still apply to any retraining)

1. **XTTS training silently skips** clips > ~11.6 s (`max_wav_length=255995`
   @ 22050 Hz) and text > 200 chars. 97% of the raw clips are longer — always
   train on the output of `scripts/segment_dataset.py` (≤10 s, ≤190 chars).
2. The trainer synthesizes `test_sentences` at the **end of every epoch** and only
   catches `NotImplementedError`. `finetune_xtts.py` sets `test_sentences=[]` on
   purpose. Never add entries without a real `speaker_wav` file path.
3. `--resume`: `TrainerArgs.continue_path` requires the **run folder**, not a
   checkpoint file. Run folders are named `<run_name>-<Month-DD-YYYY...>-<githash>`
   (e.g. `shona_xtts_v2-August-06-2026_06+47AM-9f83158`), NOT `run-*`. The script
   accepts a `.pth` and normalizes to its parent, then requires `config.json` there.
4. FP16 is **opt-in** via `--half` and has never been exercised. On ≥16 GB do not
   use it; on the 6 GB home GPU expect dtype crash or NaN — watch first losses.
5. `save_step` counts **dataloader batches**, not optimizer steps.
6. Tokenizer: if the finetune log says `Adding N missing characters`, **stop** and
   verify `VoiceBpeTokenizer` loads the extended vocab (notebook Step 6.5 does).
   `Tokenizer already covers all Shona characters` = safe. A recovered extended
   vocab (`vocab_shona.json`) sits next to the checkpoints on Drive.
7. The `en` language token is intentional: XTTS has no `sn` token; Shona rides on
   `--xtts-language en`. Dataset text stays Shona.
8. Trainer writes `best_model_<step>.pth` (no bare `best_model.pth`); occasional
   single-step NaN losses poison the log's running-average display but NOT the
   weights — check epoch-end eval loss, not the `(nan)` averages.

## Data corruption history — never train from these

- 46% of `data/sna_xtts_ft_filtered/wavs` and the LaCie copies
  (`/Volumes/LaCie/WaxalNLP/sna_asr`, `sna_xtts_ft`) are **all-zero silent WAVs**
  (stale HF arrow-cache decode bug, root-caused 2026-07-22). Still on disk.
- Clean sources: HF `manassehzw/sna-dataset-annotated`; LaCie `sna_xtts_ft_v2`
  (audited, 0 silent); `sna_xtts_ft_v2.zip` (2.4 GB, on Drive); segmented set
  `MyDrive/shona_xtts/sna_xtts_seg10s` (what training actually used).
- Silence guards are built into `build_hf_xtts_dataset.py` and
  `segment_dataset.py` — keep them.

## Machine notes

- Mac (Intel): CPU-only, ~2 min/sentence inference. `uv sync` fails (torchcodec
  has no x86_64 macOS wheel); use the miniconda `python` (torch/torchaudio 2.2.2,
  numpy 1.26, datasets). coqui-tts itself is NOT installed locally.
- Home GPU box (Windows/WSL): RTX 4050 (6 GB VRAM), 32 GB RAM. `RUN_LOCAL.md` is
  the runbook. Copy data from LaCie into the WSL ext4 FS (`~/data`, not
  `/mnt/c/...`); never train/read through drvfs/USB. Keep ~15 GB free under `~`.

## Tooling gotchas (for Claude)

- Notebook cells lack stable ids: `NotebookEdit` `cell-N` ids are positional and
  shift after insertions — re-read between structural edits or edit the JSON.
- 2026-08-12: a Colab cell pasted at module level of `finetune_xtts.py` broke
  `import finetune_xtts` (and thus `infer_xtts.py`) with IndentationError.
  Removed. If inference ever fails at import, check for pasted notebook code first.

## Pending (as of 2026-08-12 — check `git log`/`git status` for current truth)

- `prosody-controls` branch is local-only (not pushed). `README.md` and this
  file's update are uncommitted.
- Phase B portal wiring (see above) awaits the owner's listening tests
  (A/B: pause 700/900/1200 ms × temperature default/0.75/0.85).
- Optional cleanup never authorized: deleting the corrupt copies
  (`data/sna_xtts_ft_filtered/`, LaCie `sna_asr/`, `sna_xtts_ft/`) — ask first.
