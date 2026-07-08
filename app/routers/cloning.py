import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from app.config import settings
from app.services.audio_processor import convert_to_wav
from app.services.tts_model import TTSModelManager

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/voices", tags=["voices"])


class GenerateRequest(BaseModel):
    voice_id: str
    text: str
    language: str = "en"


@router.post("/upload")
async def upload_voice(file: UploadFile):
    if not file.filename:
        raise HTTPException(400, "No file provided")

    voice_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix or ".wav"
    raw_dir = settings.voices_dir / voice_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"raw{ext}"

    content = await file.read()
    if len(content) > 50 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 50MB)")

    raw_path.write_bytes(content)

    wav_path = convert_to_wav(raw_path, raw_dir / "voice", sample_rate=settings.sample_rate)

    raw_path.unlink(missing_ok=True)

    return {"voice_id": voice_id, "path": str(wav_path)}


@router.post("/generate")
async def generate_voice(payload: GenerateRequest):
    wav_path = settings.voices_dir / payload.voice_id / "voice.wav"
    if not wav_path.exists():
        raise HTTPException(404, f"Voice sample not found for id '{payload.voice_id}'")

    manager = TTSModelManager()
    wav_bytes = manager.generate(text=payload.text, voice_path=str(wav_path), language=payload.language)

    return Response(content=wav_bytes, media_type="audio/wav")
