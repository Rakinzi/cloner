import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import auth, cloning
from app.services.db import create_db_and_tables
from app.services.queue import worker as queue_worker
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
    create_db_and_tables()
    threading.Thread(target=_warm_model_in_background, daemon=True).start()
    worker_task = asyncio.create_task(queue_worker())
    logger.info("Application ready at http://0.0.0.0:8000")
    yield
    worker_task.cancel()
    logger.info("Shutting down.")


app = FastAPI(title="Shona Voice Cloner", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def no_cache_static(request, call_next):
    # StaticFiles sends ETag/Last-Modified but no explicit Cache-Control, so
    # browsers can serve a stale HTML/JS page without revalidating on a plain
    # refresh — this has already caused a fix to appear not to take effect.
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path in (
        "/", "/login", "/register", "/generate", "/voices", "/history",
    ):
        response.headers["Cache-Control"] = "no-store"
    return response

app.include_router(auth.router)
app.include_router(cloning.router)

_static = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/generate")


@app.get("/login", include_in_schema=False)
async def login_page():
    return FileResponse(str(_static / "login.html"))


@app.get("/register", include_in_schema=False)
async def register_page():
    return FileResponse(str(_static / "register.html"))


@app.get("/generate", include_in_schema=False)
async def generate_page():
    return FileResponse(str(_static / "generate.html"))


@app.get("/voices", include_in_schema=False)
async def voices_page():
    return FileResponse(str(_static / "voices.html"))


@app.get("/history", include_in_schema=False)
async def history_page():
    return FileResponse(str(_static / "history.html"))


@app.get("/health")
async def health():
    from app.config import settings as s
    return {"status": "ok", "voices_dir": str(s.voices_dir), "model": s.tts_model_name}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
