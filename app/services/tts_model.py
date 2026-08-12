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

            if settings.finetuned_model_path:
                self._model = self._load_finetuned(device)
            else:
                logger.info("Loading stock XTTSv2 model (this may take a while on first run)...")
                tts = TTS(model_name=settings.tts_model_name, progress_bar=True)
                if device == "cuda":
                    tts.to(device)
                    # TTS.api.TTS is a wrapper; the loaded model lives on its
                    # synthesizer in current Coqui TTS releases.
                    if torch.cuda.is_available() and tts.synthesizer is not None:
                        tts.synthesizer.tts_model.half()
                self._model = tts

            self._device = device
            logger.info("TTS model loaded successfully on %s", device)

    def _load_finetuned(self, device: str):
        """Load the Shona-finetuned GPT checkpoint, falling back to base XTTSv2 assets
        (dvae, mel stats, vocab) the same way scripts/finetune_xtts.py trained against."""
        import sys

        import torch

        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts

        scripts_dir = str((Path(__file__).resolve().parents[2] / "scripts"))
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)
        from finetune_xtts import download_checkpoints

        checkpoint_path = Path(settings.finetuned_model_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"CLONER_FINETUNED_MODEL_PATH does not exist: {checkpoint_path}")

        cache_dir = settings.model_cache_dir / "xtts_checkpoints"
        logger.info("Checking base XTTSv2 assets in %s ...", cache_dir)
        ckpt_paths = download_checkpoints(cache_dir)

        config = XttsConfig()
        run_config = checkpoint_path.parent / "config.json"
        config.load_json(str(run_config if run_config.exists() else ckpt_paths["config.json"]))
        config.model_args.mel_norm_file = str(ckpt_paths["mel_stats.pth"])
        config.model_args.dvae_checkpoint = str(ckpt_paths["dvae.pth"])
        config.model_args.xtts_checkpoint = str(ckpt_paths["model.pth"])
        tokenizer_path = Path(config.model_args.tokenizer_file)
        # The stock config uses an empty tokenizer path. Path("") is ".",
        # which exists but is a directory and cannot be loaded as a tokenizer.
        if not tokenizer_path.is_file():
            tokenizer_path = ckpt_paths["vocab.json"]
        config.model_args.tokenizer_file = str(tokenizer_path)

        logger.info("Loading finetuned Shona checkpoint from %s", checkpoint_path)
        model = Xtts.init_from_config(config)
        model.load_checkpoint(
            config,
            checkpoint_path=str(checkpoint_path),
            vocab_path=str(tokenizer_path),
            eval=True,
            strict=False,
        )
        model.to(device)
        self._xtts_config = config
        return model

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

        model = self.model  # triggers load_model() if needed

        with torch.inference_mode():
            if settings.finetuned_model_path:
                gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[voice_path])
                out = model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                )
                wav = out["wav"]
            else:
                wav = model.tts(text=text, speaker_wav=voice_path, language=language)

            wav_tensor = torch.tensor(wav, device="cpu")
            wav_int = (wav_tensor * 32767).to(torch.int16).numpy()

            buf = io.BytesIO()
            wavfile.write(buf, rate=24000, data=wav_int)
            buf.seek(0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return buf.getvalue()
