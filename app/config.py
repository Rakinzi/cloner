from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_cache_dir: Path = Path("./storage/models")
    voices_dir: Path = Path("./storage/voices")
    sample_rate: int = 22050
    device: str = "cuda"
    tts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    finetuned_model_path: str = ""  # path to fine-tuned .pth checkpoint

    model_config = {"env_prefix": "CLONER_", "env_file": ".env"}


settings = Settings()
settings.voices_dir.mkdir(parents=True, exist_ok=True)
settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
