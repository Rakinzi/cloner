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

Prosody: put "|" in a sentence wherever you want a dramatic pause — chunks are
synthesized separately (same voice) and stitched with --pause-ms of silence
(default 900 ms). --auto-pause inserts markers before Shona conjunctions when a
sentence has none, and --temperature widens pitch/prosody variation:

    python scripts/infer_xtts.py ... --temperature 0.75 --seed 42 \\
        --sentence "Ndinotenda zvikuru nekundibatsira kwamakaita pamusika | uye ndinovimba | tichashanda pamwe zvakare."
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torchaudio

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
    parser.add_argument(
        "--pause-marker",
        default="|",
        help="Character in the text marking a dramatic pause; chunks are synthesized separately "
        "and stitched with silence (default '|')",
    )
    parser.add_argument("--pause-ms", type=int, default=900,
                        help="Silence inserted at each pause marker, in ms (default 900)")
    parser.add_argument(
        "--auto-pause",
        action="store_true",
        help="If a sentence has no pause markers, insert them before Shona phrase-boundary "
        "conjunctions (uye, asi, kana, nekuti, nokuti, saka)",
    )
    parser.add_argument("--temperature", type=float, default=None,
                        help="GPT sampling temperature; higher = more pitch/prosody variation "
                        "(library default if unset)")
    parser.add_argument("--top-p", type=float, default=None,
                        help="Nucleus sampling cutoff (library default if unset)")
    parser.add_argument("--repetition-penalty", type=float, default=None,
                        help="Penalty against repeated tokens (library default if unset)")
    parser.add_argument("--speed", type=float, default=None,
                        help="Speech speed factor, e.g. 0.9 slower / 1.1 faster (library default if unset)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Fix the torch RNG seed for reproducible A/B comparisons")
    return parser.parse_args()


def load_shona_texts_from_checkpoint_neighbors() -> list[str]:
    """Best-effort text source for tokenizer coverage; falls back to the default sentences."""
    return DEFAULT_SENTENCES


PAUSE_CONJUNCTIONS = ("uye", "asi", "kana", "nekuti", "nokuti", "saka")


def insert_auto_pauses(text: str, marker: str) -> str:
    """Insert a pause marker before Shona phrase-boundary conjunctions (no-op if markers exist)."""
    if marker in text:
        return text
    pattern = re.compile(r"\s+(?=(?:%s)\b)" % "|".join(PAUSE_CONJUNCTIONS), re.IGNORECASE)
    return pattern.sub(f" {marker} ", text)


def trim_edges(wav: np.ndarray, sr: int, threshold_ratio: float = 0.01, margin_s: float = 0.05) -> np.ndarray:
    """Trim leading/trailing silence so stitched pauses stay accurate.

    XTTS emits a variable amount of silence around each chunk; without trimming
    it would stack on top of the explicit inter-chunk pause and make gap
    lengths unpredictable.
    """
    amp = np.abs(wav)
    peak = float(amp.max()) if amp.size else 0.0
    if peak <= 0.0:
        return wav
    above = np.nonzero(amp > peak * threshold_ratio)[0]
    if above.size == 0:
        return wav
    margin = int(sr * margin_s)
    start = max(0, int(above[0]) - margin)
    end = min(len(wav), int(above[-1]) + 1 + margin)
    return wav[start:end]


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
        if not tokenizer_path.is_file():
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

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Only forward knobs the user actually set, so unset flags keep the library defaults.
    sampling_kwargs = {
        key: value
        for key, value in {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "speed": args.speed,
        }.items()
        if value is not None
    }

    sample_rate = config.audio.output_sample_rate
    pause = np.zeros(int(sample_rate * args.pause_ms / 1000), dtype=np.float32)

    for idx, text in enumerate(sentences, start=1):
        if args.auto_pause:
            text = insert_auto_pauses(text, args.pause_marker)
        chunks = [c.strip() for c in text.split(args.pause_marker) if c.strip()]

        pieces: list[np.ndarray] = []
        for n, chunk in enumerate(chunks, start=1):
            logger.info("Synthesizing [%d/%d] chunk %d/%d: %s", idx, len(sentences), n, len(chunks), chunk)
            out = model.inference(
                text=chunk,
                language=args.xtts_language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                **sampling_kwargs,
            )
            if pieces:
                pieces.append(pause)
            pieces.append(trim_edges(np.asarray(out["wav"], dtype=np.float32), sample_rate))

        wav = torch.from_numpy(np.concatenate(pieces)).unsqueeze(0)
        out_path = output_dir / f"sample_{idx:02d}.wav"
        torchaudio.save(str(out_path), wav, sample_rate)
        logger.info("Wrote %s", out_path)

    logger.info("Done. %d samples written to %s", len(sentences), output_dir)


if __name__ == "__main__":
    main()
