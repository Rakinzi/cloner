def test_register_creates_user_and_sets_cookie(client):
    res = client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    assert res.status_code == 200
    assert res.json()["username"] == "tariro"
    assert "session" in res.cookies


def test_register_rejects_duplicate_username(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.post("/api/v1/auth/register", json={"username": "tariro", "password": "different"})
    assert res.status_code == 400


def test_register_rejects_blank_fields(client):
    res = client.post("/api/v1/auth/register", json={"username": "", "password": "x"})
    assert res.status_code == 400
    res = client.post("/api/v1/auth/register", json={"username": "x", "password": ""})
    assert res.status_code == 400


def test_login_succeeds_with_correct_credentials(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.post("/api/v1/auth/login", json={"username": "tariro", "password": "hunter2pass"})
    assert res.status_code == 200
    assert res.json()["username"] == "tariro"


def test_login_rejects_wrong_password(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.post("/api/v1/auth/login", json={"username": "tariro", "password": "wrong"})
    assert res.status_code == 401


def test_login_rejects_unknown_username(client):
    res = client.post("/api/v1/auth/login", json={"username": "nobody", "password": "x"})
    assert res.status_code == 401


def test_me_requires_authentication(client):
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 401


def test_me_returns_username_when_logged_in(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 200
    assert res.json()["username"] == "tariro"


def test_logout_clears_session(client):
    client.post("/api/v1/auth/register", json={"username": "tariro", "password": "hunter2pass"})
    res = client.post("/api/v1/auth/logout")
    assert res.status_code == 200
    res = client.get("/api/v1/auth/me")
    assert res.status_code == 401
