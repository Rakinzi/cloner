import logging
import threading
from collections import OrderedDict
from pathlib import Path

from app.config import settings

logger = logging.getLogger(__name__)
_MPL_CACHE_DIR = Path("./storage/matplotlib")
_MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_MAX_CONDITIONING_CACHE_ENTRIES = 8

# Estimated output-audio-seconds per input character, derived from two
# verified real generations ("Mangwanani" -> 2.18s/10 chars = 0.218;
# a 5-char sample -> 1.29s/5 chars = 0.258). Used only to estimate a
# generation's total expected duration for progress reporting — retune
# by hand from real (text_len, audio_seconds_produced) log lines as more
# data accumulates. Not a wall-clock timing estimate.
OUTPUT_SECONDS_PER_CHAR = 0.22


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
                    obj._conditioning_cache = OrderedDict()
                    obj._conditioning_lock = threading.Lock()
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
                    if torch.cuda.is_available():
                        tts.model.half()
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
        if not tokenizer_path.exists():
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

    def _get_conditioning_latents(self, model, voice_path: str):
        """Return cached voice conditioning, recomputing it if the WAV changed."""
        path = Path(voice_path).resolve()
        stat = path.stat()
        cache_key = (str(path), stat.st_mtime_ns, stat.st_size)

        with self._conditioning_lock:
            cached = self._conditioning_cache.get(cache_key)
            if cached is not None:
                self._conditioning_cache.move_to_end(cache_key)
                return cached

            conditioning = model.get_conditioning_latents(audio_path=[str(path)])

            # Drop a stale fingerprint for this path before storing its new one.
            for key in list(self._conditioning_cache):
                if key[0] == str(path):
                    del self._conditioning_cache[key]
            self._conditioning_cache[cache_key] = conditioning
            while len(self._conditioning_cache) > _MAX_CONDITIONING_CACHE_ENTRIES:
                self._conditioning_cache.popitem(last=False)

            return conditioning

    def generate(self, text: str, voice_path: str, language: str = "en") -> bytes:
        import io

        import torch
        from scipy.io import wavfile

        model = self.model  # triggers load_model() if needed

        with torch.inference_mode():
            if settings.finetuned_model_path:
                gpt_cond_latent, speaker_embedding = self._get_conditioning_latents(model, voice_path)
                out = model.inference(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    # The low-level XTTS API does not split text by default.  In
                    # that mode text beyond the GPT token limit is truncated,
                    # which is especially easy to hit with Shona tokenization.
                    # XTTS joins the generated sentence chunks in the returned
                    # waveform, so the caller still receives one WAV file.
                    enable_text_splitting=True,
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

    def generate_streaming(self, text: str, voice_path: str, language: str, on_chunk) -> bytes:
        """Stream-generate audio via Xtts.inference_stream, invoking on_chunk(seconds_so_far)
        after every decoded chunk. Only valid on the finetuned-checkpoint code path — the
        stock TTS.api.TTS object has no streaming equivalent in this codebase."""
        import io

        import torch
        from scipy.io import wavfile

        model = self.model  # triggers load_model() if needed
        sample_rate = 24000

        with torch.inference_mode():
            gpt_cond_latent, speaker_embedding = self._get_conditioning_latents(model, voice_path)

            chunks = []
            seconds_so_far = 0.0
            for chunk in model.inference_stream(
                text=text,
                language=language,
                gpt_cond_latent=gpt_cond_latent,
                speaker_embedding=speaker_embedding,
                enable_text_splitting=True,
            ):
                chunks.append(chunk)
                seconds_so_far += len(chunk) / sample_rate
                on_chunk(seconds_so_far)

            full_wav = torch.cat(chunks, dim=-1) if chunks else torch.zeros(0)
            wav_tensor = full_wav.to("cpu")
            wav_int = (wav_tensor * 32767).to(torch.int16).numpy()

            buf = io.BytesIO()
            wavfile.write(buf, rate=sample_rate, data=wav_int)
            buf.seek(0)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return buf.getvalue()
