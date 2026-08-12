import io

import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, create_engine

import app.services.db as db_module
import app.services.queue as queue_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "voices_dir", tmp_path / "voices")
    settings.voices_dir.mkdir(parents=True, exist_ok=True)

    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'test.sqlite3'}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(test_engine)

    # `engine` was imported by value into every consuming module (app.services.queue,
    # app.services.db itself) before this fixture runs, so rebinding it on db_module
    # alone would not reach them — patch each module's own reference explicitly.
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(queue_module, "engine", test_engine)

    from app.main import app as fastapi_app
    return TestClient(fastapi_app)


@pytest.fixture()
def tiny_wav_bytes():
    # Minimal valid WAV header + silence, enough for ffmpeg to accept.
    import wave
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 1600)
    buf.seek(0)
    return buf.read()


@pytest.fixture()
def noisy_wav_bytes():
    # Speech-band harmonics plus strong 6-10kHz content, mirroring a
    # reference clip with music/noise bleeding into the recording.
    import math
    import struct
    import wave

    sr = 22050
    duration_s = 2.0
    freqs = [300, 800, 1500, 2500, 3200, 7000, 8500]
    amplitude = 0.3
    per_tone_amp = amplitude / len(freqs)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(int(duration_s * sr)):
            sample = sum(per_tone_amp * math.sin(2 * math.pi * f * i / sr) for f in freqs)
            frames += struct.pack("<h", int(sample * 32767))
        w.writeframes(bytes(frames))
    buf.seek(0)
    return buf.read()


def register_and_login(client, username="testuser", password="testpass123"):
    res = client.post("/api/v1/auth/register", json={"username": username, "password": password})
    assert res.status_code == 200, res.text
    return client
