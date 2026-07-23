#!/usr/bin/env python
"""Segment long dataset clips into XTTS-sized pieces with aligned text.

XTTS v2 training silently skips any clip longer than ~11.6 s
(max_wav_length=255995 samples @ 22050 Hz) and any text over 200 chars.
Most of our Shona clips are longer than that, so without this step the
trainer sees almost none of the data.

This script uses torchaudio's MMS forced aligner (Meta MMS, supports
Latin-script languages such as Shona) to find word-level timestamps,
then cuts each clip into chunks that fit the XTTS limits, preferring
cuts at sentence boundaries. Output is a new dataset directory in the
same Coqui metadata.csv format as the input.

Usage:
    python scripts/segment_dataset.py \
        --dataset data/sna_xtts_ft_filtered \
        --output data/sna_xtts_ft_segmented
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import torchaudio.functional as AF

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("segment_dataset")

ALIGN_SR = 16000  # MMS aligner operates at 16 kHz
SENTENCE_END = re.compile(r"[.!?…]['\"]?$")


@dataclass
class Word:
    text: str  # original word, punctuation kept
    start: float  # seconds, in the original clip
    end: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input dataset dir with metadata.csv and wavs/")
    parser.add_argument("--output", required=True, help="Output dataset dir")
    parser.add_argument("--max-seconds", type=float, default=10.0, help="Max segment duration")
    parser.add_argument("--min-seconds", type=float, default=3.2,
                        help="Drop segments shorter than this (XTTS conditioning needs >= 3 s)")
    parser.add_argument("--max-chars", type=int, default=190, help="Max segment text length (XTTS limit is 200)")
    parser.add_argument("--keep-short-seconds", type=float, default=11.0,
                        help="Clips already shorter than this are copied through unsegmented")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def normalize_word(word: str) -> str:
    """Reduce a Shona word to the a-z/apostrophe alphabet the MMS aligner expects."""
    word = unicodedata.normalize("NFKD", word.lower())
    word = word.encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z']", "", word)


def align_words(waveform_16k: torch.Tensor, text: str, model, tokenizer, aligner, device: str) -> list[Word]:
    """Return word-level timestamps for `text` in the given 16 kHz waveform."""
    original_words = text.split()
    normalized = [normalize_word(w) for w in original_words]
    kept_indices = [i for i, w in enumerate(normalized) if w]
    if not kept_indices:
        raise ValueError("no alignable words")

    with torch.inference_mode():
        emission, _ = model(waveform_16k.to(device))
    token_spans = aligner(emission[0], tokenizer([normalized[i] for i in kept_indices]))

    seconds_per_frame = waveform_16k.size(1) / emission.size(1) / ALIGN_SR
    timed: dict[int, tuple[float, float]] = {}
    for idx, spans in zip(kept_indices, token_spans):
        timed[idx] = (spans[0].start * seconds_per_frame, spans[-1].end * seconds_per_frame)

    # Words the aligner could not use (digits, pure punctuation) get folded
    # into the nearest aligned neighbour so their text is not lost.
    words: list[Word] = []
    pending_prefix: list[str] = []
    for i, orig in enumerate(original_words):
        if i in timed:
            start, end = timed[i]
            words.append(Word(" ".join(pending_prefix + [orig]), start, end))
            pending_prefix = []
        elif words:
            words[-1].text += f" {orig}"
        else:
            pending_prefix.append(orig)
    return words


def split_into_sentences(words: list[Word]) -> list[list[Word]]:
    sentences: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        if SENTENCE_END.search(word.text):
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return sentences


def chunk_words(words: list[Word], max_seconds: float, max_chars: int) -> list[list[Word]]:
    """Pack whole sentences into chunks within the duration/char budget.

    A single sentence that itself exceeds the budget is split greedily on
    word boundaries.
    """

    def duration(group: list[Word]) -> float:
        return group[-1].end - group[0].start

    def char_len(group: list[Word]) -> int:
        return len(" ".join(w.text for w in group))

    pieces: list[list[Word]] = []
    for sentence in split_into_sentences(words):
        if duration(sentence) <= max_seconds and char_len(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        current: list[Word] = []
        for word in sentence:
            candidate = current + [word]
            if current and (duration(candidate) > max_seconds or char_len(candidate) > max_chars):
                pieces.append(current)
                current = [word]
            else:
                current = candidate
        if current:
            pieces.append(current)

    chunks: list[list[Word]] = []
    for piece in pieces:
        if chunks:
            candidate = chunks[-1] + piece
            if duration(candidate) <= max_seconds and char_len(candidate) <= max_chars:
                chunks[-1] = candidate
                continue
        chunks.append(piece)
    return chunks


def cut_boundaries(chunks: list[list[Word]], clip_seconds: float, margin: float = 0.15) -> list[tuple[float, float]]:
    """Convert word-time chunks to cut points, splitting the gap between chunks."""
    boundaries = []
    for i, chunk in enumerate(chunks):
        start = chunk[0].start - margin
        end = chunk[-1].end + margin
        if i > 0:
            midpoint = (chunks[i - 1][-1].end + chunk[0].start) / 2
            start = max(start, midpoint)
        if i < len(chunks) - 1:
            midpoint = (chunk[-1].end + chunks[i + 1][0].start) / 2
            end = min(end, midpoint)
        boundaries.append((max(0.0, start), min(clip_seconds, end)))
    return boundaries


def main() -> None:
    args = parse_args()
    input_dir = Path(args.dataset)
    output_dir = Path(args.output)
    metadata_path = input_dir / "metadata.csv"
    if not metadata_path.exists():
        logger.error("metadata.csv not found in %s", input_dir)
        sys.exit(1)

    (output_dir / "wavs").mkdir(parents=True, exist_ok=True)

    logger.info("Loading MMS forced-alignment model (first run downloads ~1.2 GB)...")
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).to(args.device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()

    with open(metadata_path, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="|"))
    logger.info("Segmenting %d clips from %s", len(rows), input_dir)

    out_rows: list[dict] = []
    stats = {"input_clips": len(rows), "copied": 0, "segmented": 0, "align_failed": 0,
             "dropped_short": 0, "dropped_silent": 0, "output_seconds": 0.0}

    for n, row in enumerate(rows, 1):
        wav_path = input_dir / row["audio_file"]
        try:
            audio, sr = sf.read(wav_path, dtype="float32", always_2d=False)
        except Exception as exc:
            logger.warning("Cannot read %s: %s", wav_path, exc)
            continue
        clip_seconds = len(audio) / sr

        # Corrupt exports have produced all-zero WAVs before; the forced
        # aligner returns garbage timestamps on silence, so drop them here.
        if float(abs(audio).max()) < 1e-4:
            stats["dropped_silent"] += 1
            logger.warning("Silent audio, dropped: %s", wav_path.name)
            continue

        def emit(segment, text: str, index: int | None) -> None:
            stem = wav_path.stem if index is None else f"{wav_path.stem}_seg{index:02d}"
            out_name = f"wavs/{stem}.wav"
            sf.write(output_dir / out_name, segment, sr, subtype="PCM_16")
            out_rows.append({"audio_file": out_name, "text": text.strip(),
                             "speaker_name": row["speaker_name"],
                             "emotion_name": row.get("emotion_name", "neutral")})
            stats["output_seconds"] += len(segment) / sr

        if clip_seconds <= args.keep_short_seconds and len(row["text"]) <= args.max_chars:
            emit(audio, row["text"], None)
            stats["copied"] += 1
            continue

        try:
            wave = torch.from_numpy(audio).unsqueeze(0)
            wave_16k = AF.resample(wave, sr, ALIGN_SR)
            words = align_words(wave_16k, row["text"], model, tokenizer, aligner, args.device)
        except Exception as exc:
            stats["align_failed"] += 1
            logger.warning("Alignment failed for %s (%s); clip dropped", wav_path.name, exc)
            continue

        chunks = chunk_words(words, args.max_seconds, args.max_chars)
        boundaries = cut_boundaries(chunks, clip_seconds)
        for i, (chunk, (t0, t1)) in enumerate(zip(chunks, boundaries)):
            if t1 - t0 < args.min_seconds:
                stats["dropped_short"] += 1
                continue
            emit(audio[int(t0 * sr):int(t1 * sr)], " ".join(w.text for w in chunk), i)
        stats["segmented"] += 1

        if n % 100 == 0:
            logger.info("Processed %d/%d clips -> %d segments so far", n, len(rows), len(out_rows))

    with open(output_dir / "metadata.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_file", "text", "speaker_name", "emotion_name"],
                                delimiter="|")
        writer.writeheader()
        writer.writerows(out_rows)

    stats["output_clips"] = len(out_rows)
    stats["output_hours"] = round(stats["output_seconds"] / 3600, 3)
    del stats["output_seconds"]
    with open(output_dir / "segment_report.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2)

    logger.info("Done: %s", json.dumps(stats, indent=2))
    logger.info("Segmented dataset written to %s", output_dir)


if __name__ == "__main__":
    main()
