"""
Fine-tune XTTSv2 on a Shona dataset.

Downloads base XTTSv2 checkpoints, extends the tokenizer with any missing
Shona characters, and fine-tunes only the GPT encoder — optimised for 6 GB VRAM.

Usage:
    # 1. Prepare dataset first:
    python scripts/prepare_dataset.py --input ./raw_shona --output ./data/shona_ft

    # 2. Run fine-tuning:
    python scripts/finetune_xtts.py --dataset ./data/shona_ft --output ./checkpoints/shona_xtts

    # 3. Resume from a checkpoint:
    python scripts/finetune_xtts.py --dataset ./data/shona_ft --output ./checkpoints/shona_xtts \\
        --resume ./checkpoints/shona_xtts/run-2024-01-01_00-00-00/checkpoint_1000.pth
"""

import argparse
import csv
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import torch
from trainer import Trainer, TrainerArgs

# Coqui TTS can inherit Colab's inline backend, which is invalid in some CLI runs.
os.environ["MPLBACKEND"] = "Agg"

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig
from TTS.tts.models.xtts import XttsAudioConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("finetune_xtts")

METADATA_FILENAME = "metadata.csv"

# Official Coqui HF mirror
_HF_BASE = "https://huggingface.co/coqui/XTTS-v2/resolve/main"
CHECKPOINT_FILES = {
    "model.pth": f"{_HF_BASE}/model.pth",
    "vocab.json": f"{_HF_BASE}/vocab.json",
    "dvae.pth": f"{_HF_BASE}/dvae.pth",
    "mel_stats.pth": f"{_HF_BASE}/mel_stats.pth",
    "config.json": f"{_HF_BASE}/config.json",
    "speakers_xtts.pth": f"{_HF_BASE}/speakers_xtts.pth",
}


def download_checkpoints(cache_dir: Path) -> dict[str, Path]:
    """Download base XTTSv2 checkpoints if not already cached."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for name, url in CHECKPOINT_FILES.items():
        dest = cache_dir / name
        if dest.exists():
            logger.info("  Cached: %s", dest)
        else:
            logger.info("  Downloading %s ...", name)
            torch.hub.download_url_to_file(url, str(dest), progress=True)
        paths[name] = dest

    return paths


def extend_tokenizer_vocab(vocab_path: Path, shona_texts: list[str]) -> Path:
    """
    Add any Shona characters missing from the BPE vocab as single-char tokens.
    Only touches the 'added_tokens' section — does NOT rebuild merge rules.
    """
    with open(vocab_path, encoding="utf-8") as f:
        vocab_data = json.load(f)

    existing_vocab: dict = vocab_data.get("model", {}).get("vocab", {})
    added_tokens: list = vocab_data.get("added_tokens", [])
    added_strs = {t["content"] for t in added_tokens}

    unique_chars = {ch for text in shona_texts for ch in text if ch.strip()}
    missing = unique_chars - set(existing_vocab.keys()) - added_strs

    if not missing:
        logger.info("Tokenizer already covers all Shona characters.")
        return vocab_path

    logger.info("Adding %d missing characters to tokenizer: %s", len(missing), sorted(missing))

    next_id = max(existing_vocab.values(), default=0) + len(added_tokens) + 1
    for ch in sorted(missing):
        added_tokens.append({
            "id": next_id,
            "content": ch,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": False,
        })
        next_id += 1

    vocab_data["added_tokens"] = added_tokens
    extended_path = vocab_path.parent / "vocab_shona.json"
    with open(extended_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    logger.info("Extended vocab saved to %s", extended_path)
    return extended_path


def load_shona_texts(dataset_path: Path) -> list[str]:
    texts = []
    with open(dataset_path / METADATA_FILENAME, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            if row.get("text"):
                texts.append(row["text"])
    return texts


def validate_dataset(dataset_path: Path) -> tuple[int, int]:
    """Return (train_count, eval_count) and exit on fatal issues."""
    meta = dataset_path / METADATA_FILENAME
    if not meta.exists():
        logger.error(
            "metadata.csv not found in %s — build the dataset first with build_hf_xtts_dataset.py "
            "or filter_annotated_for_xtts.py.",
            dataset_path,
        )
        sys.exit(1)

    missing_wavs = []
    count = 0

    with open(meta, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="|")
        for row in reader:
            count += 1
            wav = dataset_path / row["audio_file"]
            if not wav.exists():
                missing_wavs.append(row["audio_file"])

    if missing_wavs:
        logger.warning("%d WAV files listed in metadata but not found:", len(missing_wavs))
        for w in missing_wavs[:10]:
            logger.warning("  %s", w)
        if len(missing_wavs) > 10:
            logger.warning("  ... and %d more", len(missing_wavs) - 10)

    if count - len(missing_wavs) < 10:
        logger.error("Need at least 10 valid samples to fine-tune (found %d).", count - len(missing_wavs))
        sys.exit(1)

    eval_size = max(1, int(count * 0.01))
    return count - eval_size, eval_size


def main():
    parser = argparse.ArgumentParser(description="Fine-tune XTTSv2 on Shona")
    parser.add_argument("--dataset", required=True, help="Path to processed dataset (from prepare_dataset.py)")
    parser.add_argument("--output", default="./checkpoints/shona_xtts", help="Output directory for checkpoints")
    parser.add_argument("--cache", default="./cache/xtts_checkpoints", help="Cache dir for base model files")
    parser.add_argument("--resume", default="", help="Path to checkpoint .pth to resume from")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per GPU (default 1 for 6 GB VRAM)")
    parser.add_argument("--grad-accum", type=int, default=84, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=5e-6, help="Peak learning rate")
    parser.add_argument("--steps", type=int, default=5000, help="Total training steps")
    parser.add_argument("--save-step", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader workers (0 = main process, safer on Mac)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 (use if you get NaN losses)")
    parser.add_argument(
        "--xtts-language",
        default="en",
        help="XTTS language token to use. XTTS does not support 'sn', so keep this on a supported language such as 'en'.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    output_path = Path(args.output)
    cache_dir = Path(args.cache)

    # --- Validate ---
    train_count, eval_count = validate_dataset(dataset_path)
    logger.info("Dataset: %d train / %d eval samples", train_count, eval_count)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning("CUDA not available — training on CPU will be very slow.")
    logger.info("Device: %s", device)

    # --- Download base checkpoints ---
    logger.info("Checking base XTTSv2 checkpoints...")
    ckpt_paths = download_checkpoints(cache_dir)

    # --- Extend tokenizer for Shona characters ---
    shona_texts = load_shona_texts(dataset_path)
    tokenizer_path = extend_tokenizer_vocab(ckpt_paths["vocab.json"], shona_texts)

    # --- Dataset config ---
    dataset_config = BaseDatasetConfig(
        formatter="coqui",
        dataset_name="shona",
        path=str(dataset_path),
        meta_file_train=METADATA_FILENAME,
        language=args.xtts_language,  # Dataset text is Shona; this is only the XTTS-supported language token.
    )

    # --- Model args ---
    model_args = GPTArgs(
        max_conditioning_length=132300,   # 6s @ 22050 Hz
        min_conditioning_length=66150,    # 3s @ 22050 Hz
        max_wav_length=255995,
        max_text_length=200,
        mel_norm_file=str(ckpt_paths["mel_stats.pth"]),
        dvae_checkpoint=str(ckpt_paths["dvae.pth"]),
        xtts_checkpoint=str(ckpt_paths["model.pth"]),
        gpt_checkpoint="",
        tokenizer_file=str(tokenizer_path),
        gpt_num_audio_tokens=1026,
        gpt_start_audio_token=1024,
        gpt_stop_audio_token=1025,
        gpt_use_masking_gt_prompt_approach=True,
        gpt_use_perceiver_resampler=True,
    )

    audio_config = XttsAudioConfig(
        sample_rate=22050,
        dvae_sample_rate=22050,
        output_sample_rate=24000,
    )

    # --- Training config (6 GB VRAM tuned) ---
    config = GPTTrainerConfig(
        output_path=str(output_path),
        model_args=model_args,
        audio=audio_config,
        run_name="shona_xtts_v2",
        batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        num_loader_workers=args.workers,
        num_eval_loader_workers=args.workers,
        print_step=50,
        plot_step=500,
        save_step=args.save_step,
        save_n_checkpoints=3,
        save_checkpoints=True,
        print_eval=False,
        run_eval=True,
        eval_split_size=0.01,
        test_sentences=[
            {
                "text": "Mhoro, ndinoda kudzidza chiShona.",
                "speaker_wav": "",
                "language": args.xtts_language,
            },
            {
                "text": "EcoCash, ndiwo mararamiro edu!",
                "speaker_wav": "",
                "language": args.xtts_language,
            },
        ],
        optimizer="AdamW",
        optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
        lr=args.lr,
        lr_scheduler="MultiStepLR",
        lr_scheduler_params={
            "milestones": [int(args.steps * 0.5), int(args.steps * 0.8)],
            "gamma": 0.5,
        },
        grad_accum_steps=args.grad_accum,
    )

    # --- Load samples ---
    train_samples, eval_samples = load_tts_samples(
        [dataset_config],
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
    )
    logger.info("Loaded %d train / %d eval samples", len(train_samples), len(eval_samples))

    if len(train_samples) == 0:
        logger.error("No training samples loaded — check your metadata.csv paths.")
        sys.exit(1)

    for s in train_samples + eval_samples:
        s["language"] = "en"

    # --- Init model ---
    model = GPTTrainer.init_from_config(config, train_samples)
    model = model.to(device)

    if device == "cuda" and not args.no_half:
        logger.info("Enabling FP16 on GPT encoder for 6 GB VRAM optimisation")
        model.xtts.gpt.half()

    # --- Train ---
    trainer = Trainer(
        TrainerArgs(
            restore_path=None,
            grad_accum_steps=config.grad_accum_steps,
            continue_path=args.resume or "",
        ),
        config,
        output_path=str(output_path),
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )

    logger.info("Starting fine-tuning for %d steps...", args.steps)
    logger.info("Checkpoints saved every %d steps to: %s", args.save_step, output_path)
    trainer.fit()

    # --- Copy best model for easy inference ---
    runs = sorted(output_path.glob("run-*/best_model.pth"))
    if runs:
        best = runs[-1]
        dest = output_path / "shona_xtts_model.pth"
        shutil.copy(best, dest)
        logger.info("Best model copied to: %s", dest)
        logger.info("To use it, set CLONER_FINETUNED_MODEL_PATH=%s", dest)
    else:
        logger.warning("No best_model.pth found — check training logs.")

    logger.info("Done!")


if __name__ == "__main__":
    main()
