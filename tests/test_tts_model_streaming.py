from unittest.mock import MagicMock, patch

import torch

from app.services.tts_model import OUTPUT_SECONDS_PER_CHAR, TTSModelManager


def _fake_chunks(n=3, samples_per_chunk=4800):
    # 4800 samples @ 24000Hz = 0.2s per chunk
    return [torch.zeros(samples_per_chunk) for _ in range(n)]


def test_generate_streaming_calls_on_chunk_per_yielded_chunk():
    manager = TTSModelManager()
    fake_model = MagicMock()
    fake_model.inference_stream.return_value = iter(_fake_chunks(3))

    seen_progress = []

    with patch.object(TTSModelManager, "model", new=fake_model):
        with patch.object(manager, "_get_conditioning_latents", return_value=("latent", "embedding")):
            result = manager.generate_streaming(
                text="Mhoro",
                voice_path="fake.wav",
                language="en",
                on_chunk=lambda seconds_so_far: seen_progress.append(seconds_so_far),
            )

    assert len(seen_progress) == 3
    # Cumulative, strictly increasing
    assert seen_progress[0] < seen_progress[1] < seen_progress[2]
    assert isinstance(result, bytes)
    assert len(result) > 0


def test_output_seconds_per_char_is_positive_constant():
    assert OUTPUT_SECONDS_PER_CHAR > 0
