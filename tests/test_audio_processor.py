import struct
import wave

from app.services.audio_processor import check_reference_quality


def _write_tone_wav(path, freqs_hz, duration_s=3.0, sr=22050, amplitude=0.3):
    import math

    if isinstance(freqs_hz, (int, float)):
        freqs_hz = [freqs_hz]

    n_samples = int(duration_s * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        per_tone_amp = amplitude / len(freqs_hz)
        for i in range(n_samples):
            sample = sum(per_tone_amp * math.sin(2 * math.pi * f * i / sr) for f in freqs_hz)
            frames += struct.pack("<h", int(sample * 32767))
        w.writeframes(bytes(frames))


def test_clean_speech_band_harmonics_are_not_flagged(tmp_path):
    # A handful of harmonics spread across the speech band (300-3400Hz) with
    # no high-frequency content approximates how real voiced speech
    # distributes energy — should read as "clean".
    wav_path = tmp_path / "clean.wav"
    _write_tone_wav(wav_path, freqs_hz=[300, 800, 1500, 2500, 3200])
    result = check_reference_quality(wav_path)
    assert result["likely_noisy"] is False


def test_high_frequency_dominant_audio_is_flagged(tmp_path):
    # Same speech-band harmonics, but with strong 6-10kHz content added on
    # top — mirrors music/noise bleeding into a reference clip.
    wav_path = tmp_path / "noisy.wav"
    _write_tone_wav(wav_path, freqs_hz=[300, 800, 1500, 2500, 3200, 7000, 8500])
    result = check_reference_quality(wav_path)
    assert result["likely_noisy"] is True
