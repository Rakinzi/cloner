"""
Export manassehzw/sna-dataset-annotated from Hugging Face into regular WAV files.

Output layout:
    /Volumes/LaCie/WaxalNLP/sna_asr/
        audio/
        metadata.csv
        metadata_full.csv

`metadata.csv` is pipe-delimited and contains the minimal `audio|text` format.
`metadata_full.csv` keeps the speaker/split metadata from the source dataset.
"""

from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path

import soundfile as sf
from datasets import Audio, load_dataset

HF_DATASET = "manassehzw/sna-dataset-annotated"
DEFAULT_OUTPUT = "/Volumes/LaCie/WaxalNLP/sna_asr"
DEFAULT_SPLITS = ("train", "validation", "test")


def export_split(split: str, output_dir: Path, offset: int) -> tuple[int, int]:
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(HF_DATASET, split=split).cast_column("audio", Audio(decode=False))
    print(f"Loaded {HF_DATASET} split={split} with {len(dataset)} rows")

    minimal_rows: list[tuple[str, str]] = []
    full_rows: list[dict[str, str]] = []
    skipped = 0

    for index, example in enumerate(dataset):
        text = str(example.get("transcription") or "").strip()
        audio = example.get("audio") or {}
        raw = audio.get("bytes") if isinstance(audio, dict) else None
        source_id = str(example.get("source_id") or f"{split}_{index:06d}")
        speaker_id = str(example.get("speaker_id") or "unknown")

        if not text or not raw:
            skipped += 1
            continue

        try:
            array, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
        except Exception as exc:
            print(f"SKIP {split}:{index} decode failed: {exc}")
            skipped += 1
            continue

        filename = f"{split}_{offset + index:06d}_{source_id}.wav"
        relative_path = f"audio/{filename}"
        sf.write(str(audio_dir / filename), array, sample_rate, subtype="PCM_16")

        minimal_rows.append((relative_path, text))
        full_rows.append(
            {
                "audio_file": relative_path,
                "text": text,
                "split": split,
                "source_id": source_id,
                "speaker_id": speaker_id,
                "gender": str(example.get("gender") or ""),
                "duration": str(example.get("duration") or ""),
                "quality_score": str(example.get("quality_score") or ""),
            }
        )

        if (index + 1) % 500 == 0:
            print(f"  {split}: exported {index + 1}/{len(dataset)}")

    with open(output_dir / "metadata.csv", "a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="|")
        writer.writerows(minimal_rows)

    full_exists = (output_dir / "metadata_full.csv").exists()
    with open(output_dir / "metadata_full.csv", "a", encoding="utf-8", newline="") as handle:
        fieldnames = ["audio_file", "text", "split", "source_id", "speaker_id", "gender", "duration", "quality_score"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not full_exists:
            writer.writeheader()
        writer.writerows(full_rows)

    return len(minimal_rows), skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Export annotated Shona dataset to WAV files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), help="Splits to export")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    full_metadata_path = output_dir / "metadata_full.csv"
    metadata_path.unlink(missing_ok=True)
    full_metadata_path.unlink(missing_ok=True)

    exported_total = 0
    skipped_total = 0
    offset = 0
    for split in args.splits:
        exported, skipped = export_split(split, output_dir, offset)
        exported_total += exported
        skipped_total += skipped
        offset += exported

    print(f"Done. Exported {exported_total} clips, skipped {skipped_total}, output={output_dir}")


if __name__ == "__main__":
    main()
