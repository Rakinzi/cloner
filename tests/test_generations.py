import asyncio
from unittest.mock import patch

from app.config import settings
from tests.conftest import register_and_login


def _upload_voice(client, tiny_wav_bytes, label="Voice A"):
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": label},
    )
    return res.json()["voice_id"]


def test_generate_unknown_voice_404(client):
    register_and_login(client)
    res = client.post("/api/v1/generations", json={"voice_id": "nope", "text": "Mhoro", "language": "en"})
    assert res.status_code == 404


def test_generate_returns_pending_and_processes(client, tiny_wav_bytes):
    register_and_login(client)
    voice_id = _upload_voice(client, tiny_wav_bytes)

    with patch.object(settings, "finetuned_model_path", ""):
        with patch("app.services.tts_model.TTSModelManager.generate", return_value=b"FAKEWAV"):
            res = client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "Mhoro", "language": "en"})
            assert res.status_code == 200
            body = res.json()
            assert body["status"] == "pending"
            gen_id = body["generation_id"]

            # Drain the queue synchronously for the test (worker isn't running in TestClient).
            from app.services.queue import process_one
            asyncio.run(process_one(gen_id))

    res = client.get(f"/api/v1/generations/{gen_id}")
    assert res.status_code == 200
    assert res.json()["status"] == "done"

    res = client.get(f"/api/v1/generations/{gen_id}/audio")
    assert res.status_code == 200
    assert res.content == b"FAKEWAV"


def test_generation_response_includes_progress(client, tiny_wav_bytes):
    register_and_login(client)
    voice_id = _upload_voice(client, tiny_wav_bytes)
    res = client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "Mhoro", "language": "en"})
    gen_id = res.json()["generation_id"]

    res = client.get(f"/api/v1/generations/{gen_id}")
    assert res.status_code == 200
    assert "progress" in res.json()
    assert res.json()["progress"] == 0.0


def test_audio_not_ready_returns_409(client, tiny_wav_bytes):
    register_and_login(client)
    voice_id = _upload_voice(client, tiny_wav_bytes)
    res = client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "Mhoro", "language": "en"})
    gen_id = res.json()["generation_id"]

    res = client.get(f"/api/v1/generations/{gen_id}/audio")
    assert res.status_code == 409
    assert res.json()["status"] == "pending"


def test_list_generations_filtered_by_voice(client, tiny_wav_bytes):
    register_and_login(client)
    voice_a = _upload_voice(client, tiny_wav_bytes, label="A")
    voice_b = _upload_voice(client, tiny_wav_bytes, label="B")
    client.post("/api/v1/generations", json={"voice_id": voice_a, "text": "one", "language": "en"})
    client.post("/api/v1/generations", json={"voice_id": voice_b, "text": "two", "language": "en"})

    res = client.get(f"/api/v1/generations?voice_id={voice_a}")
    gens = res.json()
    assert len(gens) == 1
    assert gens[0]["voice_id"] == voice_a


def test_cannot_generate_against_another_users_voice(client, tiny_wav_bytes):
    register_and_login(client, "alice", "alicepass123")
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Alice's Voice"},
    )
    voice_id = res.json()["voice_id"]
    client.post("/api/v1/auth/logout")

    register_and_login(client, "bob", "bobpass123")
    res = client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "hi", "language": "en"})
    assert res.status_code == 404


def test_generation_list_is_scoped_to_current_user(client, tiny_wav_bytes):
    register_and_login(client, "alice", "alicepass123")
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Alice's Voice"},
    )
    voice_id = res.json()["voice_id"]
    client.post("/api/v1/generations", json={"voice_id": voice_id, "text": "hi", "language": "en"})
    client.post("/api/v1/auth/logout")

    register_and_login(client, "bob", "bobpass123")
    res = client.get("/api/v1/generations")
    assert res.json() == []
