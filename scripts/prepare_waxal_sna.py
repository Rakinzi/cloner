"""
Prepare the WaxalNLP Shona ASR parquet dataset for XTTSv2 transfer learning.

This script reads parquet shards with embedded audio bytes, filters low-value
examples, converts audio into Coqui-compatible WAV files, and writes a
multi-speaker `metadata.csv` that can be consumed by `scripts/finetune_xtts.py`.

Usage:
    python scripts/prepare_waxal_sna.py \
        --input /Volumes/LaCie/WaxalNLP/sna_asr/data \
        --output ./data/waxal_sna_ft
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

SAMPLE_RATE = 22050
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 20.0
MIN_RMS_DB = -40.0
MIN_TEXT_CHARS = 8
MAX_TEXT_CHARS = 200
DEFAULT_BATCH_SIZE = 64
DEFAULT_WORKERS = 4

WHITESPACE_RE = re.compile(r"\s+")
ALLOWED_TEXT_RE = re.compile(r"[^0-9A-Za-zÀ-ÿĀ-ſƀ-ɏḀ-ỿ'’.,!?;:()/%\- ]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert WaxalNLP Shona parquet shards into an XTTS-ready dataset."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Directory containing split parquet shards (for example train-*.parquet)",
    )
    parser.add_argument("--output", required=True, help="Output directory for WAVs and metadata.csv")
    parser.add_argument(
        "--include-unlabeled",
        action="store_true",
        help="Include unlabeled shards if they somehow contain transcriptions",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation"],
        help="Shard groups to include (default: train validation)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after accepting this many clips (0 = no limit)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Number of ffmpeg worker threads (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Parquet batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=MIN_DURATION_SEC,
        help=f"Minimum clip duration in seconds (default: {MIN_DURATION_SEC})",
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=MAX_DURATION_SEC,
        help=f"Maximum clip duration in seconds (default: {MAX_DURATION_SEC})",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=MIN_TEXT_CHARS,
        help=f"Minimum normalized text length (default: {MIN_TEXT_CHARS})",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=MAX_TEXT_CHARS,
        help=f"Maximum normalized text length (default: {MAX_TEXT_CHARS})",
    )
    parser.add_argument(
        "--max-clips-per-speaker",
        type=int,
        default=0,
        help="Cap accepted clips per speaker to balance the dataset (0 = no cap)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing WAV files in the output directory when present",
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Disable silence trimming during audio conversion",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    text = text.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    text = WHITESPACE_RE.sub(" ", text.strip())
    text = ALLOWED_TEXT_RE.sub(" ", text)
    text = WHITESPACE_RE.sub(" ", text).strip(" -")
    return text


def sanitize_name(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip())
    cleaned = cleaned.strip("_")
    return cleaned or fallback


def get_audio_info(path: Path) -> dict[str, float]:
    duration_proc = subprocess.run(
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
        duration = float(duration_proc.stdout.strip())
    except ValueError:
        return {"duration": 0.0, "rms_db": -99.0}

    rms_proc = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-af",
            "volumedetect",
            "-vn",
            "-sn",
            "-dn",
            "-f",
            "null",
            "/dev/null",
        ],
        capture_output=True,
        text=True,
    )
    rms_db = -99.0
    for line in rms_proc.stderr.splitlines():
        if "mean_volume" not in line:
            continue
        try:
            rms_db = float(line.split(":")[-1].strip().replace(" dB", ""))
        except ValueError:
            pass
        break

    return {"duration": duration, "rms_db": rms_db}


def validate_audio(path: Path, min_duration: float, max_duration: float) -> tuple[bool, str, dict[str, float]]:
    info = get_audio_info(path)
    duration = info["duration"]
    rms_db = info["rms_db"]

    if duration < min_duration:
        return False, f"too short ({duration:.2f}s < {min_duration}s)", info
    if duration > max_duration:
        return False, f"too long ({duration:.2f}s > {max_duration}s)", info
    if rms_db < MIN_RMS_DB:
        return False, f"too quiet ({rms_db:.1f} dB < {MIN_RMS_DB} dB)", info
    return True, "ok", info


def convert_audio_bytes(audio_bytes: bytes, output_path: Path, sample_rate: int, trim_silence: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-i",
        "pipe:0",
    ]
    if trim_silence:
        command.extend(
            [
                "-af",
                (
                    "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB:"
                    "stop_periods=1:stop_silence=0.3:stop_threshold=-50dB"
                ),
            ]
        )
    command.extend(
        [
            "-ar",
            str(sample_rate),
            "-ac",
            "1",
            "-sample_fmt",
            "s16",
            str(output_path),
        ]
    )
    process = subprocess.run(command, input=audio_bytes, capture_output=True)
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="ignore")[-400:]
        raise RuntimeError(stderr or "ffmpeg conversion failed")


def process_row(
    row: dict[str, Any],
    split_name: str,
    wav_dir: Path,
    sample_rate: int,
    min_duration: float,
    max_duration: float,
    min_text_chars: int,
    max_text_chars: int,
    trim_silence: bool,
    skip_existing: bool,
) -> dict[str, Any]:
    clip_id = sanitize_name(str(row.get("source_id") or row.get("id") or "clip"), "clip")
    speaker_id = sanitize_name(str(row.get("speaker_id") or "speaker"), "speaker")
    language = (row.get("language") or "").strip().lower()
    text = normalize_text(str(row.get("transcription") or ""))
    audio = row.get("audio") or {}
    audio_bytes = audio.get("bytes") or b""

    if language and language != "sna":
        return {"accepted": False, "reason": f"language={language}", "id": clip_id, "speaker_id": speaker_id}
    if not text:
        return {"accepted": False, "reason": "empty transcription", "id": clip_id, "speaker_id": speaker_id}
    if len(text) < min_text_chars:
        return {
            "accepted": False,
            "reason": f"text too short ({len(text)} < {min_text_chars})",
            "id": clip_id,
            "speaker_id": speaker_id,
        }
    if len(text) > max_text_chars:
        return {
            "accepted": False,
            "reason": f"text too long ({len(text)} > {max_text_chars})",
            "id": clip_id,
            "speaker_id": speaker_id,
        }
    if not audio_bytes:
        return {"accepted": False, "reason": "missing audio bytes", "id": clip_id, "speaker_id": speaker_id}

    output_name = f"{speaker_id}_{clip_id}.wav"
    output_path = wav_dir / output_name

    if not (skip_existing and output_path.exists()):
        convert_audio_bytes(audio_bytes, output_path, sample_rate=sample_rate, trim_silence=trim_silence)

    ok, reason, info = validate_audio(output_path, min_duration=min_duration, max_duration=max_duration)
    if not ok:
        output_path.unlink(missing_ok=True)
        return {"accepted": False, "reason": reason, "id": clip_id, "speaker_id": speaker_id}

    return {
        "accepted": True,
        "id": clip_id,
        "speaker_id": speaker_id,
        "split": split_name,
        "audio_file": f"wavs/{output_name}",
        "text": text,
        "gender": str(row.get("gender") or "").strip() or "unknown",
        "duration": info["duration"],
        "rms_db": info["rms_db"],
    }


def flush_futures(
    pending: set[Future],
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, str]],
    speaker_counts: Counter[str],
    max_clips_per_speaker: int,
    limit: int,
) -> bool:
    if not pending:
        return False

    done, remaining = wait(pending, return_when=FIRST_COMPLETED)
    pending.clear()
    pending.update(remaining)

    stop = False
    for future in done:
        result = future.result()
        if not result["accepted"]:
            rejected_rows.append(
                {
                    "id": result["id"],
                    "speaker_id": result["speaker_id"],
                    "reason": result["reason"],
                }
            )
            continue

        speaker_id = result["speaker_id"]
        if max_clips_per_speaker and speaker_counts[speaker_id] >= max_clips_per_speaker:
            rejected_rows.append(
                {
                    "id": result["id"],
                    "speaker_id": speaker_id,
                    "reason": f"speaker cap reached ({max_clips_per_speaker})",
                }
            )
            continue

        accepted_rows.append(result)
        speaker_counts[speaker_id] += 1
        if limit and len(accepted_rows) >= limit:
            stop = True

    return stop


def discover_parquet_files(input_dir: Path, splits: list[str], include_unlabeled: bool) -> list[tuple[str, Path]]:
    selected_splits = list(splits)
    if include_unlabeled and "unlabeled" not in selected_splits:
        selected_splits.append("unlabeled")

    discovered: list[tuple[str, Path]] = []
    for split_name in selected_splits:
        for path in sorted(input_dir.glob(f"{split_name}-*.parquet")):
            discovered.append((split_name, path))
        for path in sorted(input_dir.glob(f"sna-{split_name}-*.parquet")):
            discovered.append((split_name, path))
    return discovered


def write_outputs(
    output_dir: Path,
    accepted_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, str]],
) -> None:
    metadata_path = output_dir / "metadata.csv"
    rejects_path = output_dir / "rejects.csv"
    report_path = output_dir / "dataset_report.json"

    accepted_rows.sort(key=lambda row: row["audio_file"])

    with open(metadata_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerow(["audio_file", "text", "speaker_name", "emotion_name"])
        for row in accepted_rows:
            writer.writerow([row["audio_file"], row["text"], row["speaker_id"], "neutral"])

    with open(rejects_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "speaker_id", "reason"])
        writer.writeheader()
        writer.writerows(rejected_rows)

    speaker_counts = Counter(row["speaker_id"] for row in accepted_rows)
    split_counts = Counter(row["split"] for row in accepted_rows)
    durations = [row["duration"] for row in accepted_rows]
    report = {
        "accepted_clips": len(accepted_rows),
        "rejected_clips": len(rejected_rows),
        "unique_speakers": len(speaker_counts),
        "accepted_by_split": dict(sorted(split_counts.items())),
        "top_speakers": dict(speaker_counts.most_common(20)),
        "hours_of_audio": round(sum(durations) / 3600, 3),
        "avg_clip_seconds": round(sum(durations) / len(durations), 3) if durations else 0.0,
    }
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)


def remove_orphan_wavs(wav_dir: Path, accepted_rows: list[dict[str, Any]]) -> int:
    expected = {wav_dir / Path(row["audio_file"]).name for row in accepted_rows}
    removed = 0
    for wav_path in wav_dir.glob("*.wav"):
        if wav_path in expected:
            continue
        wav_path.unlink(missing_ok=True)
        removed += 1
    return removed


def select_columns(parquet_file: pq.ParquetFile) -> list[str]:
    available = set(parquet_file.schema.names)
    preferred = ["source_id", "id", "speaker_id", "transcription", "language", "gender", "audio"]
    return [column for column in preferred if column in available]


def main() -> None:
    args = parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    wav_dir = output_dir / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = discover_parquet_files(input_dir, args.splits, args.include_unlabeled)
    if not parquet_files:
        raise SystemExit(f"No matching parquet files found in {input_dir}")

    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, str]] = []
    speaker_counts: Counter[str] = Counter()
    pending: set[Future] = set()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        stop = False
        for split_name, parquet_path in parquet_files:
            if stop:
                break

            parquet_file = pq.ParquetFile(parquet_path)
            columns = select_columns(parquet_file)
            for batch in parquet_file.iter_batches(
                batch_size=max(1, args.batch_size),
                columns=columns,
            ):
                for row in batch.to_pylist():
                    if args.limit and len(accepted_rows) >= args.limit:
                        stop = True
                        break

                    pending.add(
                        executor.submit(
                            process_row,
                            row=row,
                            split_name=split_name,
                            wav_dir=wav_dir,
                            sample_rate=SAMPLE_RATE,
                            min_duration=args.min_duration,
                            max_duration=args.max_duration,
                            min_text_chars=args.min_text_chars,
                            max_text_chars=args.max_text_chars,
                            trim_silence=not args.no_trim,
                            skip_existing=args.skip_existing,
                        )
                    )

                    if len(pending) >= max(4, args.workers * 4):
                        if flush_futures(
                            pending,
                            accepted_rows,
                            rejected_rows,
                            speaker_counts,
                            args.max_clips_per_speaker,
                            args.limit,
                        ):
                            stop = True
                            break
                if stop:
                    break

        while pending:
            if flush_futures(
                pending,
                accepted_rows,
                rejected_rows,
                speaker_counts,
                args.max_clips_per_speaker,
                args.limit,
            ):
                pending.clear()
                break

    removed_orphans = remove_orphan_wavs(wav_dir, accepted_rows)
    write_outputs(output_dir, accepted_rows, rejected_rows)

    hours = sum(row["duration"] for row in accepted_rows) / 3600 if accepted_rows else 0.0
    print("=" * 60)
    print(f"Accepted clips : {len(accepted_rows)}")
    print(f"Rejected clips : {len(rejected_rows)}")
    print(f"Speakers       : {len({row['speaker_id'] for row in accepted_rows})}")
    print(f"Hours of audio : {hours:.2f}")
    print(f"Output dataset : {output_dir}")
    print("Files written  : metadata.csv, rejects.csv, dataset_report.json, wavs/")
    if removed_orphans:
        print(f"Orphan WAVs removed : {removed_orphans}")
    print("=" * 60)


if __name__ == "__main__":
    main()
