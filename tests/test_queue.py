from unittest.mock import patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

from app.config import settings
from app.models import Generation, GenerationStatus, Voice
from app.services import db as db_module
from app.services import queue as queue_module


@pytest.fixture(autouse=True)
def _fresh_db(tmp_path, monkeypatch):
    test_engine = create_engine(
        f"sqlite:///{tmp_path / 'queue_test.sqlite3'}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(test_engine)
    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(queue_module, "engine", test_engine)
    yield test_engine


def _make_voice_and_generation(engine, text="Mhoro"):
    with Session(engine) as session:
        voice = Voice(label="Test Voice", filename="a.wav", wav_path="storage/voices/v1/voice.wav", user_id="u1")
        session.add(voice)
        session.commit()
        session.refresh(voice)

        gen = Generation(voice_id=voice.id, text=text, language="en", user_id="u1")
        session.add(gen)
        session.commit()
        session.refresh(gen)
        return voice.id, gen.id


@pytest.mark.asyncio
async def test_process_one_marks_done_on_success(tmp_path, _fresh_db):
    voice_id, gen_id = _make_voice_and_generation(_fresh_db)

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch.object(settings, "finetuned_model_path", ""):
            with patch("app.services.tts_model.TTSModelManager.generate", return_value=b"FAKEWAV"):
                await queue_module.process_one(gen_id)

    with Session(_fresh_db) as session:
        gen = session.get(Generation, gen_id)
        assert gen.status == GenerationStatus.DONE
        assert gen.output_path is not None
        assert (tmp_path / f"{gen_id}.wav").exists()
        assert gen.completed_at is not None


@pytest.mark.asyncio
async def test_process_one_marks_failed_on_exception(tmp_path, _fresh_db):
    voice_id, gen_id = _make_voice_and_generation(_fresh_db)

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch.object(settings, "finetuned_model_path", ""):
            with patch("app.services.tts_model.TTSModelManager.generate", side_effect=RuntimeError("boom")):
                await queue_module.process_one(gen_id)

    with Session(_fresh_db) as session:
        gen = session.get(Generation, gen_id)
        assert gen.status == GenerationStatus.FAILED
        assert gen.error == "boom"
        assert gen.output_path is None


@pytest.mark.asyncio
async def test_process_one_updates_progress_on_streaming_path(tmp_path, _fresh_db):
    voice_id, gen_id = _make_voice_and_generation(_fresh_db, text="Mhoro zvenyu")

    def fake_generate_streaming(text, voice_path, language, on_chunk):
        on_chunk(0.5)
        on_chunk(1.0)
        on_chunk(1.5)
        return b"FAKEWAV_STREAMED"

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch.object(settings, "finetuned_model_path", "fake.pth"):
            with patch(
                "app.services.tts_model.TTSModelManager.generate_streaming",
                side_effect=fake_generate_streaming,
            ):
                await queue_module.process_one(gen_id)

    with Session(_fresh_db) as session:
        gen = session.get(Generation, gen_id)
        assert gen.status == GenerationStatus.DONE
        assert gen.progress == 1.0
        assert (tmp_path / f"{gen_id}.wav").read_bytes() == b"FAKEWAV_STREAMED"


@pytest.mark.asyncio
async def test_process_one_progress_never_reaches_one_while_running(tmp_path, _fresh_db):
    voice_id, gen_id = _make_voice_and_generation(_fresh_db, text="a")  # 1 char, tiny estimate

    captured_progress = []

    def fake_generate_streaming(text, voice_path, language, on_chunk):
        # Deliberately overshoot the estimated total to prove the clamp holds.
        on_chunk(10.0)
        with Session(_fresh_db) as session:
            captured_progress.append(session.get(Generation, gen_id).progress)
        return b"FAKEWAV"

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch.object(settings, "finetuned_model_path", "fake.pth"):
            with patch(
                "app.services.tts_model.TTSModelManager.generate_streaming",
                side_effect=fake_generate_streaming,
            ):
                await queue_module.process_one(gen_id)

    assert captured_progress[0] <= 0.99


@pytest.mark.asyncio
async def test_process_one_base_model_path_unaffected(tmp_path, _fresh_db):
    voice_id, gen_id = _make_voice_and_generation(_fresh_db)

    with patch.object(queue_module, "GENERATIONS_DIR", tmp_path):
        with patch.object(settings, "finetuned_model_path", ""):
            with patch("app.services.tts_model.TTSModelManager.generate", return_value=b"FAKEWAV"):
                await queue_module.process_one(gen_id)

    with Session(_fresh_db) as session:
        gen = session.get(Generation, gen_id)
        assert gen.status == GenerationStatus.DONE
        assert gen.progress == 1.0  # set on completion regardless of path
