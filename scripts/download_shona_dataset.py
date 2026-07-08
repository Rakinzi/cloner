"""
Download teeofftechnologies/shona-google-waxal-whisper-cleaned-16k from Hugging Face
and export it in the pipe-delimited format expected by prepare_dataset.py.

Output layout:
    ./raw_shona/
        audio/          ← raw .wav files from HF (16kHz — prepare_dataset.py resamples to 22050Hz)
        metadata.csv    ← pipe-delimited: audio_filename|transcription

Usage:
    python scripts/download_shona_dataset.py
    python scripts/download_shona_dataset.py --output ./my_raw --split train
"""

import argparse
import csv
import io
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import load_dataset


HF_DATASET = "teeofftechnologies/shona-google-waxal-whisper-cleaned-16k"
DEFAULT_OUTPUT = "./raw_shona"


def export_dataset(split: str, output_dir: Path) -> int:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {HF_DATASET} (split={split}) ...")
    # decode=False gives us raw bytes — avoids the torchcodec dependency
    ds = load_dataset(HF_DATASET, split=split, trust_remote_code=True).cast_column(
        "audio", __import__("datasets").Audio(decode=False)
    )
    print(f"  {len(ds)} examples found")

    rows = []
    skipped = 0

    for i, example in enumerate(ds):
        audio = example.get("audio") or example.get("file")
        text = (
            example.get("sentence")
            or example.get("transcription")
            or example.get("text")
            or ""
        ).strip()

        if not text:
            skipped += 1
            continue

        try:
            # audio is now {"bytes": b"...", "path": "..."} — decode with soundfile
            raw = audio.get("bytes") if isinstance(audio, dict) else audio
            array, sr = sf.read(io.BytesIO(raw), dtype="int16", always_2d=False)
        except Exception as e:
            print(f"  SKIP {i}: could not decode audio — {e}")
            skipped += 1
            continue

        fname = f"shona_{i:05d}.wav"
        sf.write(str(audio_dir / fname), array, sr, subtype="PCM_16")
        rows.append((f"audio/{fname}", text))

        if (i + 1) % 500 == 0:
            print(f"  Exported {i + 1}/{len(ds)} ...")

    meta_path = output_dir / "metadata.csv"
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        for audio_file, text in rows:
            writer.writerow([audio_file, text])

    print(f"\nDone.")
    print(f"  Exported : {len(rows)} clips")
    print(f"  Skipped  : {skipped} (empty transcription or bad audio)")
    print(f"  Output   : {output_dir}")
    print(f"\nNext step:")
    print(f"  python scripts/prepare_dataset.py --input {output_dir} --output ./data/shona_ft")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description="Download Shona HF dataset for XTTSv2 fine-tuning")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--split", default="train", help="Dataset split to download (default: train)")
    args = parser.parse_args()

    export_dataset(args.split, Path(args.output))


if __name__ == "__main__":
    main()
