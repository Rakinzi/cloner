"""
Delete short WAV files from a processed XTTS dataset and rewrite metadata.

Usage:
    python scripts/prune_short_wavs.py \
        --dataset ./data/waxal_sna_ft \
        --min-duration 10
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove WAV files shorter than a duration threshold from a processed XTTS dataset."
    )
    parser.add_argument("--dataset", required=True, help="Dataset folder containing metadata.csv and wavs/")
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="Minimum allowed duration in seconds (default: 10.0)",
    )
    return parser.parse_args()


def get_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    try:
        return float(proc.stdout.strip())
    except ValueError:
        return 0.0


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset)
    metadata_path = dataset_dir / "metadata.csv"
    rejects_path = dataset_dir / "pruned_short_wavs.csv"

    if not metadata_path.exists():
        raise SystemExit(f"metadata.csv not found in {dataset_dir}")

    kept_rows: list[dict[str, str]] = []
    removed_rows: list[dict[str, str]] = []

    with open(metadata_path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        fieldnames = reader.fieldnames or ["audio_file", "text", "speaker_name", "emotion_name"]
        rows = list(reader)

    for row in rows:
        wav_path = dataset_dir / row["audio_file"]
        duration = get_duration(wav_path) if wav_path.exists() else 0.0

        if duration < args.min_duration:
            wav_path.unlink(missing_ok=True)
            removed_rows.append(
                {
                    "audio_file": row["audio_file"],
                    "speaker_name": row.get("speaker_name", ""),
                    "duration": f"{duration:.3f}",
                    "reason": f"shorter than {args.min_duration}s",
                }
            )
            continue

        kept_rows.append(row)

    with open(metadata_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="|")
        writer.writeheader()
        writer.writerows(kept_rows)

    with open(rejects_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_file", "speaker_name", "duration", "reason"])
        writer.writeheader()
        writer.writerows(removed_rows)

    print("=" * 60)
    print(f"Dataset          : {dataset_dir}")
    print(f"Minimum duration : {args.min_duration:.2f}s")
    print(f"Kept rows        : {len(kept_rows)}")
    print(f"Deleted WAVs     : {len(removed_rows)}")
    print(f"Deletion log     : {rejects_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
