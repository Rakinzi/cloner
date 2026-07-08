"""
Build an XTTS-ready Shona dataset directly from Hugging Face.

This avoids relying on previously exported WAVs. It reads the annotated dataset,
filters rows, caps dominant speakers, and writes clean PCM16 WAVs plus Coqui metadata.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

HF_DATASET = "manassehzw/sna-dataset-annotated"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build XTTS subset directly from HF dataset")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"], help="Splits to include")
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=20.0)
    parser.add_argument("--min-quality", type=float, default=70.0)
    parser.add_argument("--max-clips-per-speaker", type=int, default=200)
    return parser.parse_args()


def collect_rows(args: argparse.Namespace) -> list[dict]:
    by_speaker: dict[str, list[dict]] = defaultdict(list)
    for split in args.splits:
        print(f"[info] loading split={split} from {HF_DATASET}", flush=True)
        ds = load_dataset(HF_DATASET, split=split).cast_column("audio", Audio(decode=False))
        accepted_for_split = 0
        for example in ds:
            speaker_id = str(example.get("speaker_id") or "").strip()
            if not speaker_id or speaker_id == "unknown":
                continue
            duration = float(example.get("duration") or 0.0)
            quality = float(example.get("quality_score") or 0.0)
            if duration < args.min_duration or duration > args.max_duration:
                continue
            if quality < args.min_quality:
                continue

            audio = example.get("audio") or {}
            raw = audio.get("bytes") if isinstance(audio, dict) else None
            if not raw:
                continue

            try:
                audio_array, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            except Exception as exc:
                source_id = str(example.get("source_id") or "")
                print(f"[warn] skip split={split} source_id={source_id} decode failed: {exc}", flush=True)
                continue

            by_speaker[speaker_id].append(
                {
                    "split": split,
                    "speaker_id": speaker_id,
                    "source_id": str(example.get("source_id") or ""),
                    "text": str(example.get("transcription") or "").strip(),
                    "duration": duration,
                    "quality_score": quality,
                    "audio_array": audio_array,
                    "sample_rate": int(sample_rate),
                }
            )
            accepted_for_split += 1
            if accepted_for_split % 100 == 0:
                print(f"[progress] accepted {accepted_for_split} candidate clips from split={split}", flush=True)
        print(f"[info] accepted {accepted_for_split} candidate clips from split={split}", flush=True)

    selected: list[dict] = []
    for speaker_id, rows in by_speaker.items():
        ranked = sorted(rows, key=lambda r: (r["quality_score"], -r["duration"], r["source_id"]), reverse=True)
        if args.max_clips_per_speaker > 0:
            ranked = ranked[: args.max_clips_per_speaker]
        selected.extend(ranked)

    selected.sort(key=lambda r: (r["speaker_id"], r["source_id"]))
    return selected


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    wav_dir = output_dir / "wavs"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(args)
    print(f"[info] writing {len(rows)} filtered clips to {output_dir}", flush=True)
    coqui_rows: list[dict[str, str]] = []
    full_rows: list[dict[str, str]] = []

    for idx, row in enumerate(rows, start=1):
        filename = f"speaker_{row['speaker_id']}_{row['source_id']}.wav"
        out_path = wav_dir / filename
        sf.write(out_path, row["audio_array"], row["sample_rate"], subtype="PCM_16")
        if idx % 100 == 0 or idx == len(rows):
            print(f"[progress] wrote {idx}/{len(rows)} wavs", flush=True)
        coqui_rows.append(
            {
                "audio_file": f"wavs/{filename}",
                "text": row["text"],
                "speaker_name": f"speaker_{row['speaker_id']}",
                "emotion_name": "neutral",
            }
        )
        full_rows.append(
            {
                "audio_file": f"wavs/{filename}",
                "text": row["text"],
                "speaker_name": f"speaker_{row['speaker_id']}",
                "emotion_name": "neutral",
                "split": row["split"],
                "source_id": row["source_id"],
                "speaker_id": row["speaker_id"],
                "duration": f"{row['duration']}",
                "quality_score": f"{row['quality_score']}",
            }
        )

    with open(output_dir / "metadata.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_file", "text", "speaker_name", "emotion_name"], delimiter="|")
        writer.writeheader()
        writer.writerows(coqui_rows)

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
        writer.writerows(full_rows)

    speaker_counts = Counter(row["speaker_id"] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    report = {
        "accepted_clips": len(rows),
        "hours_of_audio": round(sum(row["duration"] for row in rows) / 3600, 3),
        "unique_speakers": len(speaker_counts),
        "accepted_by_split": dict(sorted(split_counts.items())),
        "top_speakers": dict(speaker_counts.most_common(20)),
    }
    with open(output_dir / "dataset_report.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(report)


if __name__ == "__main__":
    main()
