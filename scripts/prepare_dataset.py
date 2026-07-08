"""
Prepares a Shona audio dataset for XTTSv2 fine-tuning.

Expects a directory with audio files and a metadata CSV file.
Validates audio quality, trims silence, converts to Coqui format.

Usage:
    python scripts/prepare_dataset.py --input ./raw_shona --output ./data/shona_ft

Metadata format (pipe-delimited, no header):
    audio_filename|transcription text
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

SAMPLE_RATE = 22050
MIN_DURATION_SEC = 1.0
MAX_DURATION_SEC = 30.0
MIN_RMS_DB = -40.0  # Reject audio quieter than this (likely silence/corrupt)


def get_audio_info(path: Path) -> dict:
    """Return duration and RMS dB of an audio file via ffprobe/ffmpeg."""
    # Duration
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return {"duration": 0.0, "rms_db": -99.0}

    # RMS level
    result = subprocess.run(
        [
            "ffmpeg", "-i", str(path),
            "-af", "volumedetect",
            "-vn", "-sn", "-dn",
            "-f", "null", "/dev/null",
        ],
        capture_output=True, text=True,
    )
    rms_db = -99.0
    for line in result.stderr.splitlines():
        if "mean_volume" in line:
            try:
                rms_db = float(line.split(":")[-1].strip().replace(" dB", ""))
            except ValueError:
                pass
            break

    return {"duration": duration, "rms_db": rms_db}


def convert_audio(input_path: Path, output_path: Path) -> Path:
    """Convert audio to 22050 Hz mono 16-bit WAV, trim leading/trailing silence."""
    output_path = output_path.with_suffix(".wav")
    if output_path.exists():
        return output_path

    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", "silenceremove=start_periods=1:start_silence=0.1:start_threshold=-50dB"
               ":stop_periods=1:stop_silence=0.3:stop_threshold=-50dB",
        "-ar", str(SAMPLE_RATE),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-300:]}")
    return output_path


def validate_audio(path: Path, min_dur: float = MIN_DURATION_SEC, max_dur: float = MAX_DURATION_SEC) -> tuple[bool, str]:
    """Return (ok, reason) after checking duration and level."""
    info = get_audio_info(path)
    dur = info["duration"]
    rms = info["rms_db"]

    if dur < min_dur:
        return False, f"too short ({dur:.2f}s < {min_dur}s)"
    if dur > max_dur:
        return False, f"too long ({dur:.2f}s > {max_dur}s)"
    if rms < MIN_RMS_DB:
        return False, f"too quiet / silent ({rms:.1f} dB < {MIN_RMS_DB} dB)"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser(description="Prepare Shona dataset for XTTSv2 fine-tuning")
    parser.add_argument("--input", required=True, help="Input directory with audio + metadata.csv")
    parser.add_argument("--output", required=True, help="Output directory for processed dataset")
    parser.add_argument("--min-duration", type=float, default=MIN_DURATION_SEC,
                        help=f"Minimum clip duration in seconds (default: {MIN_DURATION_SEC})")
    parser.add_argument("--max-duration", type=float, default=MAX_DURATION_SEC,
                        help=f"Maximum clip duration in seconds (default: {MAX_DURATION_SEC})")
    parser.add_argument("--no-trim", action="store_true",
                        help="Skip silence trimming")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / "wavs"
    wav_dir.mkdir(exist_ok=True)

    meta_file = input_dir / "metadata.csv"
    if not meta_file.exists():
        print(f"ERROR: metadata.csv not found in {input_dir}")
        print("Expected format (pipe-delimited, no header):")
        print("  audio_filename|transcription text")
        sys.exit(1)

    # Parse metadata
    rows = []
    with open(meta_file, encoding="utf-8") as f:
        for i, row in enumerate(csv.reader(f, delimiter="|"), start=1):
            if len(row) < 2:
                print(f"  Line {i}: skipping malformed row")
                continue
            audio_file, text = row[0].strip(), row[1].strip()
            if not text:
                print(f"  Line {i}: skipping empty transcription")
                continue
            # Resolve audio path
            for candidate in [input_dir / audio_file, input_dir / "wavs" / audio_file]:
                if candidate.exists():
                    rows.append((audio_file, text, candidate))
                    break
            else:
                print(f"  WARNING: {audio_file} not found, skipping")

    print(f"\nFound {len(rows)} entries in metadata.csv")

    accepted, skipped_convert, skipped_quality = [], [], []

    for audio_file, text, src_path in rows:
        stem = Path(audio_file).stem
        out_wav = wav_dir / f"{stem}.wav"

        # Convert
        try:
            if args.no_trim:
                subprocess.run(
                    ["ffmpeg", "-y", "-i", str(src_path),
                     "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16", str(out_wav)],
                    capture_output=True, text=True, check=True,
                )
            else:
                convert_audio(src_path, out_wav)
        except (subprocess.CalledProcessError, RuntimeError) as e:
            print(f"  SKIP (convert error) {audio_file}: {e}")
            skipped_convert.append(audio_file)
            continue

        # Validate converted file
        ok, reason = validate_audio(out_wav, args.min_duration, args.max_duration)
        if not ok:
            print(f"  SKIP (quality) {audio_file}: {reason}")
            out_wav.unlink(missing_ok=True)
            skipped_quality.append((audio_file, reason))
            continue

        accepted.append((f"wavs/{stem}.wav", text))

    # Write output metadata
    meta_out = output_dir / "metadata.csv"
    with open(meta_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")
        writer.writerow(["audio_file", "text", "speaker_name", "emotion_name"])
        for rel_path, text in accepted:
            writer.writerow([rel_path, text, "shona_speaker", "neutral"])

    # Summary
    print(f"\n{'='*50}")
    print(f"  Accepted : {len(accepted)}")
    print(f"  Skipped (convert error) : {len(skipped_convert)}")
    print(f"  Skipped (quality check) : {len(skipped_quality)}")
    if skipped_quality:
        for fname, reason in skipped_quality:
            print(f"    - {fname}: {reason}")
    print(f"\nDataset ready at: {output_dir}")

    total_dur = sum(
        get_audio_info(output_dir / p)["duration"] for p, _ in accepted
    )
    print(f"Total audio: {total_dur/60:.1f} minutes ({len(accepted)} clips)")
    if total_dur < 30 * 60:
        print(f"\n  HINT: XTTSv2 fine-tuning works best with 30+ minutes of audio.")
        print(f"  You have {total_dur/60:.1f} min — results may be inconsistent.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
