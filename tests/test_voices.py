from tests.conftest import register_and_login


def test_upload_requires_label(client, tiny_wav_bytes):
    register_and_login(client)
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": ""},
    )
    assert res.status_code == 400


def test_upload_and_list_voice(client, tiny_wav_bytes):
    register_and_login(client)
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "My Voice"},
    )
    assert res.status_code == 200
    voice_id = res.json()["voice_id"]
    assert res.json()["label"] == "My Voice"

    res = client.get("/api/v1/voices")
    assert res.status_code == 200
    voices = res.json()
    assert len(voices) == 1
    assert voices[0]["id"] == voice_id
    assert voices[0]["label"] == "My Voice"


def test_delete_voice(client, tiny_wav_bytes):
    register_and_login(client)
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Temp"},
    )
    voice_id = res.json()["voice_id"]

    res = client.delete(f"/api/v1/voices/{voice_id}")
    assert res.status_code == 204

    res = client.get("/api/v1/voices")
    assert res.json() == []


def test_delete_unknown_voice_404(client):
    register_and_login(client)
    res = client.delete("/api/v1/voices/does-not-exist")
    assert res.status_code == 404


def test_voice_list_is_scoped_to_current_user(client, tiny_wav_bytes):
    register_and_login(client, "alice", "alicepass123")
    client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Alice's Voice"},
    )
    client.post("/api/v1/auth/logout")

    register_and_login(client, "bob", "bobpass123")
    res = client.get("/api/v1/voices")
    assert res.json() == []


def test_cannot_delete_another_users_voice(client, tiny_wav_bytes):
    register_and_login(client, "alice", "alicepass123")
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Alice's Voice"},
    )
    voice_id = res.json()["voice_id"]
    client.post("/api/v1/auth/logout")

    register_and_login(client, "bob", "bobpass123")
    res = client.delete(f"/api/v1/voices/{voice_id}")
    assert res.status_code == 404


def test_upload_requires_authentication(client, tiny_wav_bytes):
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "No Auth"},
    )
    assert res.status_code == 401


def test_upload_clean_audio_has_no_quality_warning(client, tiny_wav_bytes):
    register_and_login(client)
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", tiny_wav_bytes, "audio/wav")},
        data={"label": "Clean Voice"},
    )
    assert res.status_code == 200
    assert "quality_warning" not in res.json()


def test_upload_noisy_audio_surfaces_quality_warning(client, noisy_wav_bytes):
    register_and_login(client)
    res = client.post(
        "/api/v1/voices/upload",
        files={"file": ("sample.wav", noisy_wav_bytes, "audio/wav")},
        data={"label": "Noisy Voice"},
    )
    assert res.status_code == 200
    assert "quality_warning" in res.json()
    assert res.json()["quality_warning"]
