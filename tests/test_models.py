from datetime import datetime

from app.models import Generation, GenerationStatus, User, Voice


def test_voice_defaults():
    voice = Voice(id="v1", label="My Voice", filename="raw.wav", wav_path="storage/voices/v1/voice.wav", user_id="u1")
    assert voice.id == "v1"
    assert voice.label == "My Voice"
    assert isinstance(voice.created_at, datetime)


def test_generation_defaults():
    gen = Generation(id="g1", voice_id="v1", text="Mhoro", language="en", user_id="u1")
    assert gen.status == GenerationStatus.PENDING
    assert gen.output_path is None
    assert gen.error is None
    assert gen.completed_at is None


def test_generation_status_values():
    assert GenerationStatus.PENDING == "pending"
    assert GenerationStatus.RUNNING == "running"
    assert GenerationStatus.DONE == "done"
    assert GenerationStatus.FAILED == "failed"


def test_user_defaults():
    user = User(id="u1", username="tariro", password_hash="hashed")
    assert user.username == "tariro"
    assert isinstance(user.created_at, datetime)


def test_voice_requires_user_id():
    voice = Voice(id="v1", label="My Voice", filename="raw.wav", wav_path="a.wav", user_id="u1")
    assert voice.user_id == "u1"


def test_generation_requires_user_id():
    gen = Generation(id="g1", voice_id="v1", text="Mhoro", language="en", user_id="u1")
    assert gen.user_id == "u1"


def test_generation_defaults_progress_to_zero():
    gen = Generation(id="g1", voice_id="v1", text="Mhoro", language="en", user_id="u1")
    assert gen.progress == 0.0
