"""
Filter the annotated LaCie dataset into an XTTS-ready training subset.

This script:
1. reads metadata_full.csv from the exported LaCie dataset
2. filters on split, duration, quality, and speaker availability
3. caps clips per speaker using highest-quality clips first
4. converts selected WAVs to 22050 Hz mono 16-bit WAVs
5. writes Coqui-style metadata.csv for scripts/finetune_xtts.py
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter annotated Shona dataset for XTTS fine-tuning")
    parser.add_argument("--source", default="/Volumes/LaCie/WaxalNLP/sna_asr", help="Annotated dataset root")
    parser.add_argument("--output", default="./data/sna_xtts_ft_filtered", help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], help="Source splits to include")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Minimum clip duration")
    parser.add_argument("--max-duration", type=float, default=20.0, help="Maximum clip duration")
    parser.add_argument("--min-quality", type=float, default=70.0, help="Minimum quality_score")
    parser.add_argument(
        "--max-clips-per-speaker",
        type=int,
        default=200,
        help="Maximum accepted clips per speaker (0 = no cap)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Parallel ffmpeg workers")
    return parser.parse_args()


def load_candidates(args: argparse.Namespace) -> list[dict[str, str]]:
    meta_path = Path(args.source) / "metadata_full.csv"
    if not meta_path.exists():
        raise SystemExit(f"metadata_full.csv not found in {args.source}")

    by_speaker: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(meta_path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            speaker_id = row.get("speaker_id", "").strip()
            split = row.get("split", "").strip()
            if split not in args.splits or not speaker_id or speaker_id == "unknown":
                continue

            try:
                duration = float(row.get("duration", "0") or 0)
                quality = float(row.get("quality_score", "0") or 0)
            except ValueError:
                continue

            if duration < args.min_duration or duration > args.max_duration:
                continue
            if quality < args.min_quality:
                continue

            row["_duration"] = f"{duration}"
            row["_quality"] = f"{quality}"
            by_speaker[speaker_id].append(row)

    selected: list[dict[str, str]] = []
    for speaker_id, rows in by_speaker.items():
        ranked = sorted(
            rows,
            key=lambda row: (float(row["_quality"]), -float(row["_duration"]), row["audio_file"]),
            reverse=True,
        )
        if args.max_clips_per_speaker > 0:
            ranked = ranked[: args.max_clips_per_speaker]
        selected.extend(ranked)

    selected.sort(key=lambda row: (row["speaker_id"], row["audio_file"]))
    return selected


def convert_one(source_root: Path, wav_dir: Path, row: dict[str, str]) -> dict[str, str]:
    src = source_root / row["audio_file"]
    speaker_id = row["speaker_id"].strip()
    source_id = row["source_id"].strip()
    stem = f"speaker_{speaker_id}_{source_id}"
    dst = wav_dir / f"{stem}.wav"
    dst.parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-ar",
            "22050",
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(dst),
        ],
        check=True,
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration,size",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    values = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
    duration = float(values[0]) if values else 0.0
    size = int(values[1]) if len(values) > 1 else 0
    if duration <= 0.0 or size <= 128:
        raise RuntimeError(f"invalid converted wav for {src.name}: duration={duration}, size={size}")

    return {
        "audio_file": f"wavs/{dst.name}",
        "text": row["text"].strip(),
        "speaker_name": f"speaker_{speaker_id}",
        "emotion_name": "neutral",
        "split": row["split"],
        "source_id": source_id,
        "speaker_id": speaker_id,
        "duration": row["_duration"],
        "quality_score": row["_quality"],
    }


def flush_done(
    pending: set[Future],
    converted_rows: list[dict[str, str]],
) -> None:
    done, remaining = wait(pending, return_when=FIRST_COMPLETED)
    pending.clear()
    pending.update(remaining)
    for future in done:
        converted_rows.append(future.result())


def write_outputs(output_dir: Path, rows: list[dict[str, str]]) -> None:
    rows.sort(key=lambda row: row["audio_file"])

    with open(output_dir / "metadata.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["audio_file", "text", "speaker_name", "emotion_name"],
            delimiter="|",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "audio_file": row["audio_file"],
                    "text": row["text"],
                    "speaker_name": row["speaker_name"],
                    "emotion_name": row["emotion_name"],
                }
            )

    with open(output_dir / "metadata_full.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "audio_file",
                "text",
                "speaker_name",
                "emotion_name",
                "split",
                "source_id",
                "speaker_id",
                "duration",
                "quality_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    durations = [float(row["duration"]) for row in rows]
    speaker_counts = Counter(row["speaker_id"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    report = {
        "accepted_clips": len(rows),
        "hours_of_audio": round(sum(durations) / 3600, 3),
        "unique_speakers": len(speaker_counts),
        "accepted_by_split": dict(sorted(split_counts.items())),
        "top_speakers": dict(speaker_counts.most_common(20)),
        "avg_clip_seconds": round(sum(durations) / len(durations), 3) if durations else 0.0,
    }
    with open(output_dir / "dataset_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    output_dir = Path(args.output)
    wav_dir = output_dir / "wavs"
    if output_dir.exists():
        subprocess.run(["rm", "-rf", str(output_dir)], check=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    selected = load_candidates(args)
    if not selected:
        raise SystemExit("No clips matched the requested filters")

    converted_rows: list[dict[str, str]] = []
    pending: set[Future] = set()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        for row in selected:
            pending.add(executor.submit(convert_one, source_root, wav_dir, row))
            if len(pending) >= max(4, args.workers * 4):
                flush_done(pending, converted_rows)

        while pending:
            flush_done(pending, converted_rows)

    write_outputs(output_dir, converted_rows)

    print("=" * 60)
    print(f"Source dataset  : {source_root}")
    print(f"Output dataset  : {output_dir}")
    print(f"Accepted clips  : {len(converted_rows)}")
    print(f"Unique speakers : {len({row['speaker_id'] for row in converted_rows})}")
    print(f"Hours of audio  : {sum(float(row['duration']) for row in converted_rows) / 3600:.2f}")
    print("Files written   : metadata.csv, metadata_full.csv, dataset_report.json, wavs/")
    print("=" * 60)


if __name__ == "__main__":
    main()
