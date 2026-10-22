# Per-request API keys ("use my own key") for the Web app.
#
# A caller may supply their own key via the X-Api-Key header. It is passed into
# the translation worker for that run only, never written to disk or logged, and
# is IGNORED in public server mode (where the server owns the key).
#
# Skips cleanly if FastAPI's TestClient dependency (httpx) is unavailable.
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402
from webapp import server  # noqa: E402


class _Req:
    """Minimal stand-in for a Starlette Request (only .headers is read)."""

    def __init__(self, headers=None):
        self.headers = headers or {}


def test_client_key_read_and_trimmed():
    assert server._client_api_key(_Req({"x-api-key": "  sk-abc  "})) == "sk-abc"
    assert server._client_api_key(_Req()) == ""
    assert server._client_api_key(None) == ""


def test_client_key_length_capped():
    key = server._client_api_key(_Req({"x-api-key": "k" * 5000}))
    assert len(key) == server.MAX_CLIENT_KEY_LEN


def test_client_key_ignored_in_server_mode(monkeypatch):
    monkeypatch.setattr(server, "server_mode_on", lambda: True)
    assert server._client_api_key(_Req({"x-api-key": "sk-abc"})) == ""


def test_key_precedence(monkeypatch):
    monkeypatch.setattr(server, "load_api_key_for_model",
                        lambda m: "server-key" if m == "M" else "")
    # Caller's own key wins over the server's stored key.
    assert server._resolve_translation_key("M", True, "client-key") == "client-key"
    assert server._resolve_translation_key("M", True, "") == "server-key"
    # Offline runs never carry a key.
    assert server._resolve_translation_key("M", False, "client-key") == ""


def test_translate_worker_receives_client_key(tmp_path, monkeypatch):
    captured = []

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None, **kw):
            captured.append(args)

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Thread", _FakeThread)
    monkeypatch.setattr(server, "UPLOAD_DIR", str(tmp_path / "uploads"))
    import core.log_config as log_config
    monkeypatch.setattr(log_config, "system_event", lambda *a, **kw: None)

    client = TestClient(server.app)
    r = client.post(
        "/api/translate",
        files=[("files", ("a.txt", b"hello", "text/plain"))],
        data={"src_lang": "English", "dst_lang": "中文", "model": "M"},
        headers={"X-Api-Key": "sk-user"},
    )
    assert r.status_code == 200, r.text
    # Worker args: (_run_translation, task, session, dests, ..., client_events, key)
    assert captured and captured[-1][-1] == "sk-user"


def test_translate_without_header_passes_empty_key(tmp_path, monkeypatch):
    captured = []

    class _FakeThread:
        def __init__(self, target=None, args=(), daemon=None, **kw):
            captured.append(args)

        def start(self):
            pass

    monkeypatch.setattr(server.threading, "Thread", _FakeThread)
    monkeypatch.setattr(server, "UPLOAD_DIR", str(tmp_path / "uploads"))
    import core.log_config as log_config
    monkeypatch.setattr(log_config, "system_event", lambda *a, **kw: None)

    client = TestClient(server.app)
    r = client.post(
        "/api/translate",
        files=[("files", ("a.txt", b"hello", "text/plain"))],
        data={"src_lang": "English", "dst_lang": "中文", "model": "M"},
    )
    assert r.status_code == 200, r.text
    assert captured and captured[-1][-1] == ""


def test_quick_translate_receives_client_key(tmp_path, monkeypatch):
    seen = {}

    import core.quick_translate as quick_translate

    def fake_translate(text, src_lang, dst_lang, context="", client_api_key=""):
        seen["key"] = client_api_key
        return "你好", True

    monkeypatch.setattr(quick_translate, "translate", fake_translate)
    monkeypatch.setattr(quick_translate, "add_history",
                        lambda *a, **kw: [])
    monkeypatch.setattr(server, "_quick_store_dir", lambda request: str(tmp_path))

    client = TestClient(server.app)
    r = client.post("/api/quick-translate",
                    json={"text": "hello", "src_lang": "English", "dst_lang": "中文"},
                    headers={"X-Api-Key": "sk-quick"})
    assert r.status_code == 200, r.text
    assert seen["key"] == "sk-quick"
