import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlmodel import Session

from app.config import settings
from app.models import Generation, GenerationStatus, Voice
from app.services.db import engine
from app.services.tts_model import OUTPUT_SECONDS_PER_CHAR, TTSModelManager

logger = logging.getLogger(__name__)

GENERATIONS_DIR = Path("./storage/generations")
GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)

generation_queue: asyncio.Queue[str] = asyncio.Queue()


async def enqueue(generation_id: str) -> None:
    await generation_queue.put(generation_id)


async def process_one(generation_id: str) -> None:
    with Session(engine) as session:
        gen = session.get(Generation, generation_id)
        if gen is None:
            logger.warning("Generation %s vanished before processing", generation_id)
            return
        gen.status = GenerationStatus.RUNNING
        session.add(gen)
        session.commit()
        voice_id, text, language = gen.voice_id, gen.text, gen.language

    with Session(engine) as session:
        voice = session.get(Voice, voice_id)
        voice_path = voice.wav_path

    manager = TTSModelManager()

    def _write_progress(progress: float) -> None:
        with Session(engine) as session:
            gen = session.get(Generation, generation_id)
            if gen is not None:
                gen.progress = progress
                session.add(gen)
                session.commit()

    try:
        if settings.finetuned_model_path:
            estimated_total_seconds = max(len(text) * OUTPUT_SECONDS_PER_CHAR, 0.01)

            def on_chunk(seconds_so_far: float) -> None:
                progress = min(0.99, seconds_so_far / estimated_total_seconds)
                _write_progress(progress)

            wav_bytes = await asyncio.to_thread(
                manager.generate_streaming, text, voice_path, language, on_chunk
            )
        else:
            wav_bytes = await asyncio.to_thread(manager.generate, text, voice_path, language)

        GENERATIONS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = GENERATIONS_DIR / f"{generation_id}.wav"
        output_path.write_bytes(wav_bytes)

        with Session(engine) as session:
            gen = session.get(Generation, generation_id)
            gen.status = GenerationStatus.DONE
            gen.progress = 1.0
            gen.output_path = str(output_path)
            gen.completed_at = datetime.now(timezone.utc)
            session.add(gen)
            session.commit()
    except Exception as exc:
        logger.exception("Generation %s failed", generation_id)
        with Session(engine) as session:
            gen = session.get(Generation, generation_id)
            gen.status = GenerationStatus.FAILED
            gen.error = str(exc)
            gen.completed_at = datetime.now(timezone.utc)
            session.add(gen)
            session.commit()


async def worker() -> None:
    while True:
        generation_id = await generation_queue.get()
        try:
            await process_one(generation_id)
        finally:
            generation_queue.task_done()
