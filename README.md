# Shona XTTS v2 — Voice Cloning & TTS

Coqui XTTS v2 finetuned to speak **Shona**, plus a FastAPI voice-cloning portal.

## The trained model

Training completed 2026-08-06 on Colab (10 epochs, eval loss falling every epoch — healthy run).
The model lives on Google Drive:

```
MyDrive/shona_xtts/checkpoints/shona_xtts/shona_xtts_v2-August-06-2026_06+47AM-9f83158/
├── best_model_65420.pth   ← the model (~5 GB)
└── config.json            ← needed alongside it
```

Always pass **both** files to inference. The dataset (clean copy) is
`LaCie/WaxalNLP/sna_xtts_ft_v2` + `sna_xtts_ft_v2.zip`; segmented training set is
`MyDrive/shona_xtts/sna_xtts_seg10s`.

## Generate speech

```bash
python scripts/infer_xtts.py \
  --checkpoint path/to/best_model_65420.pth \
  --run-config path/to/config.json \
  --speaker-wav path/to/reference.wav \
  --output-dir output/shona_samples \
  --sentence "Mhoro, ndinofara kukuona nhasi."
```

- `--speaker-wav`: a clean 3–12 s clip of the voice to clone (any wav from the
  dataset works). **This choice shapes the whole delivery** — an expressive
  reference gives expressive output.
- Runs on GPU (seconds/sentence) or CPU (~2 min/sentence). On the Mac use the
  miniconda `python`, not `python3`.
- Omit `--sentence` to get four built-in Shona test sentences.

## The "Madara cadence" controls

The finetuned voice is naturally deep (~82 Hz median pitch). To get the slow,
commanding delivery instead of one long flat sentence:

**1. Dramatic pauses — put `|` where you want silence:**

```bash
--sentence "Ndinotenda zvikuru nekundibatsira kwamakaita pamusika | uye ndinovimba | tichashanda pamwe zvakare munguva pfupi iri kuuya."
```

Each chunk is synthesized with the same voice and stitched with exactly
`--pause-ms` of silence (default **900 ms** — measured from a real Madara
speech clip). No markers to place? `--auto-pause` inserts them before the
conjunctions *uye, asi, kana, nekuti, nokuti, saka*.

**2. Expressiveness — widen the pitch movement:**

```bash
--temperature 0.75 --seed 42
```

Higher temperature = more pitch/prosody variation (target ~0.6 octaves of
movement; the untuned output had ~0.3). Keep `--seed` fixed while comparing
settings so only the knob you changed differs.

**Tuning cheat sheet** (pick by ear):

| Knob | Try | Effect |
|---|---|---|
| `--pause-ms` | 700 / **900** / 1200 | short → snappy, long → dramatic |
| `--temperature` | default / **0.75** / 0.85 | flat → expressive → (too high = unstable) |
| `--speed` | 0.9 / 1.0 | slower, heavier delivery |
| phrase length | 4–10 words per `|` chunk | Madara speaks in 1–2.5 s bursts |

## Everything else

| Task | Where |
|---|---|
| Train locally (WSL2 + RTX 4050) | `RUN_LOCAL.md` — full runbook |
| Train on Colab | `output/jupyter-notebook/shona-xtts-colab.ipynb` (Steps 1–8; 6.5 smoke test is mandatory) |
| Segment long clips before training | `scripts/segment_dataset.py` (XTTS silently skips clips > ~11.6 s — never train unsegmented) |
| Build dataset from HuggingFace | `scripts/build_hf_xtts_dataset.py` |
| Run the web portal | `app/` (FastAPI); set `CLONER_FINETUNED_MODEL_PATH` to use the finetuned model |
| Project gotchas & history | `CLAUDE.md` |

## Current branches

- `main` — portal/auth work in progress (uncommitted).
- `prosody-controls` — the pause/expressiveness controls above + the
  best-model-copy fix in `finetune_xtts.py`. Portal wiring of these controls
  (Phase B) is planned, pending listening tests.
