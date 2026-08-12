import logging
import shutil
from pathlib import Path

from fastapi import APIRouter, Depends, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from sqlmodel import Session, select

from app.config import settings
from app.models import Generation, GenerationStatus, User, Voice
from app.services.audio_processor import check_reference_quality, convert_to_wav
from app.services.auth import get_current_user
from app.services.db import get_session
from app.services.queue import enqueue

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["voices"])


@router.post("/voices/upload")
async def upload_voice(
    file: UploadFile,
    label: str = Form(...),
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    if not file.filename:
        raise HTTPException(400, "No file provided")
    if not label or not label.strip():
        raise HTTPException(400, "Label is required")

    voice = Voice(label=label.strip(), filename=file.filename, wav_path="", user_id=current_user.id)
    raw_dir = settings.voices_dir / voice.id
    raw_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(file.filename).suffix or ".wav"
    raw_path = raw_dir / f"raw{ext}"

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")

    raw_path.write_bytes(content)
    wav_path = convert_to_wav(raw_path, raw_dir / "voice", sample_rate=settings.sample_rate)
    raw_path.unlink(missing_ok=True)

    voice.wav_path = str(wav_path)
    session.add(voice)
    session.commit()
    session.refresh(voice)

    quality = check_reference_quality(wav_path)
    response = {"voice_id": voice.id, "label": voice.label}
    if quality["likely_noisy"]:
        response["quality_warning"] = (
            "This sample may contain background music or noise, which can make "
            "generated speech sound less like this voice. For best results, use a "
            "clean, single-speaker recording with no music or background sound."
        )
    return response


@router.get("/voices")
async def list_voices(current_user: User = Depends(get_current_user), session: Session = Depends(get_session)):
    voices = session.exec(
        select(Voice).where(Voice.user_id == current_user.id).order_by(Voice.created_at.desc())
    ).all()
    return [
        {"id": v.id, "label": v.label, "filename": v.filename, "created_at": v.created_at.isoformat()}
        for v in voices
    ]


@router.delete("/voices/{voice_id}", status_code=204)
async def delete_voice(
    voice_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    voice = session.get(Voice, voice_id)
    if voice is None or voice.user_id != current_user.id:
        raise HTTPException(404, f"Voice '{voice_id}' not found")

    voice_dir = settings.voices_dir / voice_id
    shutil.rmtree(voice_dir, ignore_errors=True)

    session.delete(voice)
    session.commit()
    return Response(status_code=204)


class GenerateRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "en"


@router.post("/generations")
async def create_generation(
    payload: GenerateRequest,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    voice = session.get(Voice, payload.voice_id)
    if voice is None or voice.user_id != current_user.id:
        raise HTTPException(404, f"Voice '{payload.voice_id}' not found")

    gen = Generation(
        voice_id=payload.voice_id,
        text=payload.text,
        language=payload.language,
        user_id=current_user.id,
    )
    session.add(gen)
    session.commit()
    session.refresh(gen)

    await enqueue(gen.id)
    return {"generation_id": gen.id, "status": gen.status}


@router.get("/generations/{generation_id}")
async def get_generation(
    generation_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    gen = session.get(Generation, generation_id)
    if gen is None or gen.user_id != current_user.id:
        raise HTTPException(404, f"Generation '{generation_id}' not found")
    return {
        "id": gen.id,
        "voice_id": gen.voice_id,
        "text": gen.text,
        "language": gen.language,
        "status": gen.status,
        "progress": gen.progress,
        "error": gen.error,
        "created_at": gen.created_at.isoformat(),
        "completed_at": gen.completed_at.isoformat() if gen.completed_at else None,
    }


@router.get("/generations/{generation_id}/audio")
async def get_generation_audio(
    generation_id: str,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    gen = session.get(Generation, generation_id)
    if gen is None or gen.user_id != current_user.id:
        raise HTTPException(404, f"Generation '{generation_id}' not found")
    if gen.status != GenerationStatus.DONE:
        return JSONResponse(status_code=409, content={"status": gen.status})
    return FileResponse(gen.output_path, media_type="audio/wav")


@router.get("/generations")
async def list_generations(
    voice_id: str | None = None,
    current_user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
):
    query = select(Generation).where(Generation.user_id == current_user.id).order_by(
        Generation.created_at.desc()
    )
    if voice_id:
        query = query.where(Generation.voice_id == voice_id)
    gens = session.exec(query).all()
    return [
        {
            "id": g.id,
            "voice_id": g.voice_id,
            "text": g.text,
            "status": g.status,
            "created_at": g.created_at.isoformat(),
        }
        for g in gens
    ]
