from unittest.mock import Mock

import numpy as np

from app.config import settings
from app.services.tts_model import TTSModelManager


def make_manager(model):
    manager = TTSModelManager()
    manager._model = model
    manager._device = "cpu"
    manager._conditioning_cache.clear()
    return manager


def test_finetuned_generation_enables_text_splitting(monkeypatch, tmp_path):
    model = Mock()
    model.get_conditioning_latents.return_value = ("latent", "embedding")
    model.inference.return_value = {"wav": np.zeros(24, dtype=np.float32)}

    manager = make_manager(model)
    monkeypatch.setattr(settings, "finetuned_model_path", "checkpoint.pth")
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")

    wav = manager.generate("First sentence. Second sentence.", str(voice_path))

    assert wav[:4] == b"RIFF"
    model.inference.assert_called_once_with(
        text="First sentence. Second sentence.",
        language="en",
        gpt_cond_latent="latent",
        speaker_embedding="embedding",
        enable_text_splitting=True,
    )


def test_finetuned_generation_reuses_voice_conditioning(monkeypatch, tmp_path):
    model = Mock()
    model.get_conditioning_latents.return_value = ("latent", "embedding")
    model.inference.return_value = {"wav": np.zeros(24, dtype=np.float32)}
    manager = make_manager(model)
    monkeypatch.setattr(settings, "finetuned_model_path", "checkpoint.pth")
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")

    manager.generate("One.", str(voice_path))
    manager.generate("Two.", str(voice_path))

    model.get_conditioning_latents.assert_called_once_with(audio_path=[str(voice_path.resolve())])


def test_voice_conditioning_is_refreshed_when_file_changes(monkeypatch, tmp_path):
    model = Mock()
    model.get_conditioning_latents.return_value = ("latent", "embedding")
    model.inference.return_value = {"wav": np.zeros(24, dtype=np.float32)}
    manager = make_manager(model)
    monkeypatch.setattr(settings, "finetuned_model_path", "checkpoint.pth")
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")

    manager.generate("One.", str(voice_path))
    voice_path.write_bytes(b"different voice")
    manager.generate("Two.", str(voice_path))

    assert model.get_conditioning_latents.call_count == 2
