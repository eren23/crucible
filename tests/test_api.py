"""Tests for API routes via FastAPI TestClient."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from crucible.runner.notes import NoteStore

fastapi = pytest.importorskip("fastapi")


@pytest.fixture
def client(tmp_path: Path):
    """Create a test client with a temp NoteStore."""
    from fastapi.testclient import TestClient
    from crucible.api.server import create_app

    store = NoteStore(tmp_path / ".crucible")

    app = create_app()
    with patch("crucible.api.routes._get_note_store", return_value=store):
        yield TestClient(app)


class TestApiHealth:
    def test_health(self, client) -> None:
        r = client.get("/api/v1/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


class TestApiNotes:
    def test_add_note(self, client) -> None:
        r = client.post("/api/v1/experiments/run_001/notes", json={
            "text": "Test note from API",
            "tags": ["api-test"],
            "stage": "training",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "created"
        assert data["note"]["run_id"] == "run_001"
        assert data["note"]["tags"] == ["api-test"]

    def test_get_notes(self, client) -> None:
        client.post("/api/v1/experiments/run_001/notes", json={"text": "Note A"})
        client.post("/api/v1/experiments/run_001/notes", json={"text": "Note B"})
        r = client.get("/api/v1/experiments/run_001/notes")
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_get_notes_empty(self, client) -> None:
        r = client.get("/api/v1/experiments/nonexistent/notes")
        assert r.json()["count"] == 0

    def test_add_note_empty_body(self, client) -> None:
        r = client.post("/api/v1/experiments/run_001/notes", json={"text": ""})
        assert "error" in r.json()

    def test_default_created_by(self, client) -> None:
        r = client.post("/api/v1/experiments/run_001/notes", json={"text": "Hello"})
        assert r.json()["note"]["created_by"] == "api"

    def test_notes_in_notestore_format(self, client, tmp_path: Path) -> None:
        """Notes stored as .md files, not flat JSONL in logs/."""
        client.post("/api/v1/experiments/run_001/notes", json={"text": "Test"})
        notes_dir = tmp_path / ".crucible" / "notes" / "run_001"
        assert notes_dir.exists()
        assert len(list(notes_dir.glob("*.md"))) == 1
        # No flat JSONL in logs
        logs_dir = tmp_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        assert len(list(logs_dir.glob("*.notes.jsonl"))) == 0
