# CLAUDE.md — Shona XTTS v2 finetune (handoff notes)

Written 2026-08-05 by a Claude session on the owner's Mac. Session memory does not
travel between machines — this file is the source of truth when working on this
repo elsewhere (e.g., the home Windows/WSL GPU box). Keep it updated when facts change.

## Project

- Goal: finetune Coqui XTTS v2 to speak **Shona**. High-stakes, hard deadline.
- Two run paths:
  - **Colab** ($10 = 100 compute units, T4): `output/jupyter-notebook/shona-xtts-colab.ipynb`
  - **Local GPU** (home box, ~6 GB VRAM, Windows/WSL): `RUN_LOCAL.md`
- Dataset: HF `manassehzw/sna-dataset-annotated` (healthy at source). Filtered set:
  3,250 clips / 14.6 h / 41 speakers. It MUST be segmented before training (see below).

## Non-negotiable facts (verified 2026-08-05 against coqui-tts 0.27.5 / coqui-tts-trainer 0.3.3 source, not assumed)

1. **XTTS training silently skips** clips > ~11.6 s (`max_wav_length=255995` @ 22050 Hz)
   and text > 200 chars. 97% of the raw clips are longer — training on the unsegmented
   set means the trainer sees ~90 clips and "succeeds" while learning nothing.
   Always train on the output of `scripts/segment_dataset.py` (≤10 s, ≤190 chars).
2. The trainer synthesizes `test_sentences` at the **end of every epoch** and only
   catches `NotImplementedError` — any real exception kills the run.
   `finetune_xtts.py` sets `test_sentences=[]` on purpose. Never add entries without
   a real `speaker_wav` file path.
3. `--resume`: the underlying `TrainerArgs.continue_path` requires the **run folder**
   (`run-*/`), not a checkpoint file. The script accepts a `.pth` and normalizes to
   its parent, then requires `config.json` there.
4. FP16 is **opt-in** via `--half` and has never been exercised on a GPU.
   On ≥16 GB (Colab T4) do not use it. On the 6 GB home GPU it may be needed to fit
   memory — expect either a dtype crash (fp32 cond_mels into halved weights) or the
   risk of optimizer underflow at lr 5e-6; watch the first losses for NaN or no movement.
5. `save_step` counts **dataloader batches**, not optimizer steps. At batch size 1,
   `--save-step 3000` ≈ one ~5 GB checkpoint per 3000 clips (~2 per epoch).
6. Tokenizer: if the finetune log says `Adding N missing characters`, **stop** —
   the id assignment in `extend_tokenizer_vocab` (`max + len + 1`) can leave id gaps;
   verify `VoiceBpeTokenizer` loads the extended vocab before training (the notebook's
   Step 6.5 smoke cell does this). `Tokenizer already covers all Shona characters` = safe.
7. The `en` language token is intentional: XTTS has no `sn` token, so Shona rides on
   `--xtts-language en`. Dataset text stays Shona.

## Data corruption history — never train from these

- 46% of `data/sna_xtts_ft_filtered/wavs` and the LaCie copies
  (`/Volumes/LaCie/WaxalNLP/sna_asr`, `sna_xtts_ft`) are **all-zero silent WAVs**
  (stale HF arrow-cache decode bug, root-caused 2026-07-22). They still exist on disk.
- Clean sources: HF dataset itself; LaCie `sna_xtts_ft_v2` (audited, 0 silent);
  `sna_xtts_ft_v2.zip` (2.4 GB, made for Google Drive upload).
- Silence guards are now built into `build_hf_xtts_dataset.py` and
  `segment_dataset.py` — keep them.

## Workflow

- Colab: notebook Steps 1–6, then **Step 6.5 smoke test (mandatory, ~5 min, ~0.1 units)**,
  then Step 7. After a disconnect: Steps 1–4, then 6 (skips its work but defines paths),
  then 8 (auto-resumes the newest `run-*` folder).
- Local: `RUN_LOCAL.md`, same order: build → segment → short smoke run → full run.
- Health signals: `loss_text_ce` / `loss_mel_ce` should fall within the first epoch;
  epoch-end **eval loss** falling = generalizing. Flat train loss = stop immediately.
  `best_model.pth` is maintained continuously, so stopping early is always safe.
- Budget math (Colab): T4 ≈ 1.6–2 units/h; 10 epochs ≈ 20–35 T4-hours. Recompute from
  real it/s in the first `print_step` logs before committing to the full run.

## Machine notes

- Mac (Intel): CPU-only. `uv sync` fails (torchcodec has no x86_64 macOS wheel);
  use the miniconda `python` (torch/torchaudio 2.2.2, numpy 1.26, datasets).
- Home GPU box (Windows/WSL): 32 GB RAM, RTX 4050 (**6 GB VRAM** — the binding
  constraint), internal SSD. Keep ~15 GB free under `~` (1 rolling checkpoint +
  best_model, ~5 GB each). WSL disk grows inside
  `%LOCALAPPDATA%\Packages\...\ext4.vhdx` on `C:`.
  - The dataset lives on the LaCie external drive — treat it as cold storage only.
    Copy `WaxalNLP/sna_xtts_ft_v2` (the ONLY clean folder; its siblings are 46%
    silent) into the WSL ext4 filesystem (`~/data`, not `/mnt/c/...`) before
    segmenting or training; never train reading through drvfs/USB. Write
    checkpoints to the SSD and archive to the LaCie after the run.
  - 6 GB VRAM plan: smoke-test WITHOUT `--half` first; on OOM retry with `--half`
    (untested — watch for dtype crash / NaN / frozen loss). If `--half` fails, the
    right fix is native AMP (trainer `mixed_precision` + GradScaler, fp32 master
    weights) as a patch to `finetune_xtts.py` — write it against the actual error.

## Tooling gotcha (for Claude)

- The notebook's original cells lack stable ids: `NotebookEdit` `cell-N` ids are
  **positional** and shift after insertions. A mixed insert+replace sequence corrupted
  the cell order once (repaired 2026-08-05 by rewriting the JSON directly). Re-read the
  notebook between structural edits, or edit the JSON with a script.

## Pending (as of 2026-08-05 — check `git log`/`git status` for current truth)

- The 2026-08-05 fixes (test_sentences, `--half`, resume handling, notebook Step 6.5,
  save cadence) were uncommitted. Colab Step 3A clones `github.com/Rakinzi/cloner`,
  so nothing reaches Colab until committed AND pushed.
- Optional nice-to-have: a "Step 7.5" cell that loads the latest checkpoint and
  synthesizes a few Shona sentences for listening mid-training.
