"""
Run local inference with the finetuned Shona XTTSv2 GPT checkpoint.

Loads the base XTTSv2 assets (dvae, mel stats, extended Shona tokenizer) the
same way finetune_xtts.py does, swaps in the finetuned GPT weights, and
synthesizes a handful of Shona sentences against a reference speaker wav.

Usage:
    python scripts/infer_xtts.py \\
        --checkpoint checkpoints/best_model_65420.pth \\
        --speaker-wav data/speaker_ref_sample.wav \\
        --output-dir output/shona_samples
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

from finetune_xtts import CHECKPOINT_FILES, download_checkpoints, extend_tokenizer_vocab

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("infer_xtts")

DEFAULT_SENTENCES = [
    "Mhoro, ndinofara kukuona nhasi.",
    "Zimbabwe inyika ine vanhu vane hunhu hwakanaka.",
    "Ndinoda kudya sadza nemuriwo mangwanani ano.",
    "Mvura yanaya zvakanyanya gore rino.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize Shona speech with the finetuned XTTS model")
    parser.add_argument("--checkpoint", required=True, help="Path to the finetuned GPT checkpoint (.pth)")
    parser.add_argument("--speaker-wav", required=True, help="Reference wav for voice cloning (3-12s, clean audio)")
    parser.add_argument("--cache", default="./cache/xtts_checkpoints", help="Cache dir for base XTTSv2 assets")
    parser.add_argument(
        "--run-config",
        default="",
        help="Path to the actual run config.json saved next to the checkpoint (preferred over rebuilding one "
        "from the base XTTSv2 release, since it reflects the exact model_args training used).",
    )
    parser.add_argument("--output-dir", default="./output/shona_samples", help="Where to write synthesized wavs")
    parser.add_argument(
        "--xtts-language",
        default="en",
        help="XTTS language token. XTTS has no 'sn' token, so Shona rides on a supported language (default 'en').",
    )
    parser.add_argument("--sentence", action="append", dest="sentences", help="Add a custom sentence (repeatable)")
    return parser.parse_args()


def load_shona_texts_from_checkpoint_neighbors() -> list[str]:
    """Best-effort text source for tokenizer coverage; falls back to the default sentences."""
    return DEFAULT_SENTENCES


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    speaker_wav = Path(args.speaker_wav)
    cache_dir = Path(args.cache)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint_path)
        sys.exit(1)
    if not speaker_wav.exists():
        logger.error("Speaker reference wav not found: %s", speaker_wav)
        sys.exit(1)

    sentences = args.sentences if args.sentences else DEFAULT_SENTENCES
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    if device == "cpu":
        logger.warning("Running on CPU — each sentence may take a couple of minutes.")

    # --- Base XTTSv2 assets (dvae, mel stats, base vocab, base config) ---
    logger.info("Checking base XTTSv2 checkpoints...")
    ckpt_paths = download_checkpoints(cache_dir)

    config = XttsConfig()
    if args.run_config:
        # Reflects the exact model_args training used (e.g. confirms the tokenizer
        # extension was a no-op for this run — training used the plain base vocab.json).
        logger.info("Loading actual training run config from %s", args.run_config)
        config.load_json(args.run_config)
        tokenizer_path = Path(config.model_args.tokenizer_file)
        if not tokenizer_path.is_absolute() and not tokenizer_path.exists():
            tokenizer_path = ckpt_paths["vocab.json"]
        config.model_args.tokenizer_file = str(tokenizer_path)
        config.model_args.mel_norm_file = str(ckpt_paths["mel_stats.pth"])
        config.model_args.dvae_checkpoint = str(ckpt_paths["dvae.pth"])
        config.model_args.xtts_checkpoint = str(ckpt_paths["model.pth"])
    else:
        config.load_json(str(ckpt_paths["config.json"]))
        tokenizer_path = extend_tokenizer_vocab(ckpt_paths["vocab.json"], load_shona_texts_from_checkpoint_neighbors())
        config.model_args.tokenizer_file = str(tokenizer_path)
        config.model_args.mel_norm_file = str(ckpt_paths["mel_stats.pth"])
        config.model_args.dvae_checkpoint = str(ckpt_paths["dvae.pth"])

    logger.info("Building Xtts model from config...")
    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=str(checkpoint_path),
        vocab_path=str(tokenizer_path),
        eval=True,
        strict=False,
    )
    model.to(device)

    logger.info("Computing speaker latents from %s", speaker_wav)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[str(speaker_wav)])

    for idx, text in enumerate(sentences, start=1):
        logger.info("Synthesizing [%d/%d]: %s", idx, len(sentences), text)
        out = model.inference(
            text=text,
            language=args.xtts_language,
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
        wav = torch.tensor(out["wav"]).unsqueeze(0)
        out_path = output_dir / f"sample_{idx:02d}.wav"
        import torchaudio

        torchaudio.save(str(out_path), wav, config.audio.output_sample_rate)
        logger.info("Wrote %s", out_path)

    logger.info("Done. %d samples written to %s", len(sentences), output_dir)


if __name__ == "__main__":
    main()
