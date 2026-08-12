import logging
import os
import secrets
from pathlib import Path

from dotenv import dotenv_values
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_cache_dir: Path = Path("./storage/models")
    voices_dir: Path = Path("./storage/voices")
    db_path: Path = Path("./storage/db.sqlite3")
    sample_rate: int = 22050
    device: str = "cuda"
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    finetuned_model_path: str = "./checkpoints/best_model_65420.pth"  # path to fine-tuned .pth checkpoint
    session_secret: str = Field(default_factory=lambda: secrets.token_hex(32))

    model_config = {"env_prefix": "CLONER_", "env_file": ".env"}


settings = Settings()
settings.voices_dir.mkdir(parents=True, exist_ok=True)
settings.model_cache_dir.mkdir(parents=True, exist_ok=True)

# Checking os.environ alone misses the secret when it's only set via .env
# (pydantic-settings loads .env internally without exporting it to
# os.environ), which produced a false "not set" warning even when the
# secret was correctly and stably loaded. Check both sources.
_session_secret_configured = "CLONER_SESSION_SECRET" in os.environ or "CLONER_SESSION_SECRET" in dotenv_values(
    Settings.model_config["env_file"]
)
if not _session_secret_configured:
    logging.getLogger(__name__).warning(
        "CLONER_SESSION_SECRET not set — using a random secret generated at startup. "
        "Sessions will not survive a process restart. Set CLONER_SESSION_SECRET to persist logins."
    )
