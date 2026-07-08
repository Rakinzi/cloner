import logging
import threading
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)
_MPL_CACHE_DIR = Path("./storage/matplotlib")
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)


class TTSModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls) -> "TTSModelManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    obj = super().__new__(cls)
                    obj._model = None
                    obj._device = None
                    cls._instance = obj
        return cls._instance

    def load_model(self) -> None:
        with self._lock:
            if self._model is not None:
                return

            # Point fugashi/MeCab to the bundled unidic-lite dictionary
            import os
            import plistlib
            import subprocess
            import unidic_lite
            os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE_DIR.resolve()))
            os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE_DIR.resolve()))
            os.environ.setdefault("MPLBACKEND", "Agg")
            os.environ.setdefault("MECABRC", os.path.join(unidic_lite.DICDIR, "mecabrc"))

            original_check_output = subprocess.check_output

            def patched_check_output(cmd, *args, **kwargs):
                if list(cmd) == ["system_profiler", "-xml", "SPFontsDataType"]:
                    return plistlib.dumps([{"_items": []}])
                return original_check_output(cmd, *args, **kwargs)

            subprocess.check_output = patched_check_output
            try:
                import torch
                from TTS.api import TTS
            finally:
                subprocess.check_output = original_check_output

            device = settings.device
            if not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                device = "cpu"

            logger.info("Loading XTTSv2 model (this may take a while on first run)...")
            tts = TTS(model_name=settings.tts_model_name, progress_bar=True)

            if device == "cuda":
                tts.to(device)
                if torch.cuda.is_available():
                    tts.model.half()

            self._model = tts
            self._device = device
            logger.info("TTS model loaded successfully on %s", device)

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    @property
    def device(self) -> str:
        if self._device is None:
            self.load_model()
        return self._device

    def generate(self, text: str, voice_path: str, language: str = "en") -> bytes:
        import io

        import torch
        from scipy.io import wavfile

        with torch.inference_mode():
            wav = self.model.tts(text=text, speaker_wav=voice_path, language=language)
            wav_tensor = torch.tensor(wav, device="cpu")
            wav_int = (wav_tensor * 32767).to(torch.int16).numpy()

            buf = io.BytesIO()
            wavfile.write(buf, rate=24000, data=wav_int)
            buf.seek(0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return buf.getvalue()
