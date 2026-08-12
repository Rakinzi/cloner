from sqlmodel import Session, select

from app.models import Voice


def test_create_db_and_tables_creates_voice_table(tmp_path, monkeypatch):
    db_file = tmp_path / "test.sqlite3"
    monkeypatch.setenv("CLONER_DB_PATH", str(db_file))

    # Re-import with the new env var in effect
    import importlib

    import app.config
    import app.services.db as db_module

    importlib.reload(app.config)
    importlib.reload(db_module)

    db_module.create_db_and_tables()
    assert db_file.exists()

    with Session(db_module.engine) as session:
        session.add(Voice(id="v1", label="Test", filename="a.wav", wav_path="a.wav", user_id="u1"))
        session.commit()
        result = session.exec(select(Voice)).all()
        assert len(result) == 1
        assert result[0].label == "Test"
