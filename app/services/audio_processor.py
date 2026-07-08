import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


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
