import logging
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

# Clean human speech carries very little energy above ~6kHz relative to the
# 300-3400Hz speech band. A reference clip with music, noise, or heavy
# compression artifacts mixed in reads far higher on this ratio — this
# threshold sits between a verified-clean sample (~0.017) and a verified
# music-contaminated sample (~0.183) from real uploads.
_HIGH_FREQ_RATIO_THRESHOLD = 0.08


def check_reference_quality(wav_path: Path) -> dict:
    """Flag reference audio likely to contain background music/noise, which
    corrupts XTTS's speaker conditioning and produces a voice that doesn't
    match the reference. This is a heuristic, not a hard gate."""
    audio, sr = sf.read(str(wav_path))
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    fft = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)
    mag = np.abs(fft)

    speech_band = mag[(freqs >= 300) & (freqs <= 3400)]
    high_band = mag[(freqs >= 6000) & (freqs <= 10000)]

    speech_energy = float(speech_band.mean()) if speech_band.size else 0.0
    high_energy = float(high_band.mean()) if high_band.size else 0.0
    ratio = high_energy / speech_energy if speech_energy > 0 else 0.0

    return {"high_freq_ratio": ratio, "likely_noisy": ratio > _HIGH_FREQ_RATIO_THRESHOLD}


def convert_to_wav(input_path: Path, output_path: Path, sample_rate: int = 16000) -> Path:
    output_path = output_path.with_suffix(".wav")
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", str(sample_rate),
        "-ac", "1",
        "-sample_fmt", "s16",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("ffmpeg failed: %s", result.stderr)
        raise RuntimeError(f"Audio conversion failed: {result.stderr}")

    logger.info("Converted %s -> %s", input_path, output_path)
    return output_path
