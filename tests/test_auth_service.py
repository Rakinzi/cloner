from app.services.auth import (
    SESSION_COOKIE_NAME,
    create_session_cookie,
    hash_password,
    read_session_cookie,
    verify_password,
)


def test_hash_and_verify_password_roundtrip():
    h = hash_password("correct horse battery staple")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong password", h)


def test_hash_password_produces_different_hashes_for_same_input():
    assert hash_password("same") != hash_password("same")


def test_session_cookie_roundtrip():
    token = create_session_cookie("user-123")
    assert read_session_cookie(token) == "user-123"


def test_session_cookie_rejects_tampered_token():
    token = create_session_cookie("user-123")
    tampered = token[:-1] + ("a" if token[-1] != "a" else "b")
    assert read_session_cookie(tampered) is None


def test_session_cookie_name_is_stable_constant():
    assert SESSION_COOKIE_NAME == "session"
