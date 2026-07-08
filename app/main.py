import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import cloning
from app.services.tts_model import TTSModelManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _warm_model_in_background() -> None:
    try:
        logger.info("Background model warm-up started...")
        TTSModelManager().load_model()
    except Exception:
        logger.warning("Background model warm-up failed", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=_warm_model_in_background, daemon=True).start()
    logger.info("Application ready at http://0.0.0.0:8000")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Shona Voice Cloner", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cloning.router)

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/", include_in_schema=False)
async def ui():
    return FileResponse(str(_static / "index.html"))


@app.get("/health")
async def health():
    from app.config import settings as s
    return {"status": "ok", "voices_dir": str(s.voices_dir), "model": s.tts_model_name}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
