"""
Shared fixtures.

Nothing here needs Whisper weights, a Gemini key, or the network. The
Whisper model is replaced by a stub before the app starts (otherwise
startup would try to download it), the language model defaults to the
built-in mock provider, and "remote" audio is served by a throwaway HTTP
server on localhost.
"""

import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import audio_fetch
import llm
import main
import transcribe

API_KEY = "test-service-key"
AUTH = {"Authorization": f"Bearer {API_KEY}"}

SMALL_AUDIO = b"\x1aE\xdf\xa3" + b"\x00" * 2044  # 2 KB; never actually decoded


class FakeWhisperModel:
    """Stands in for WhisperModel: same surface, no weights."""

    feature_extractor = SimpleNamespace(sampling_rate=16000)

    def transcribe(self, audio, **kwargs):
        segment = SimpleNamespace(start=0.0, end=1.5, text=" Bruno is limping. ")
        return iter([segment]), SimpleNamespace(language="en", duration=1.5)


class FakeGemini:
    """
    Stands in for the shared genai client.

    `reply` is the text to return, or an exception to raise. Every call's
    arguments are kept so tests can inspect what would have been sent.
    """

    def __init__(self, reply):
        self.reply = reply
        self.calls: list[dict] = []

    @property
    def models(self):
        return self

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.reply, Exception):
            raise self.reply
        return SimpleNamespace(text=self.reply)


@pytest.fixture(autouse=True)
def environment(monkeypatch, tmp_path):
    """A clean, production-like configuration for every test."""
    for name in (
        "GEMINI_API_KEY", "GOOGLE_API_KEY", "ALLOW_LOCAL_AUDIO", "ALLOWED_AUDIO_HOSTS",
        "EXTRACTION_MODEL", "CHAT_MODEL", "WHISPER_MODEL", "LLM_TIMEOUT_SECONDS",
        "MAX_CONCURRENT_TRANSCRIPTIONS",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SERVICE_API_KEY", API_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    # The fake upstream lives on localhost; never send that via a proxy.
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")

    # Downloads land in a per-test directory so leaks can be asserted on.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    # Module-level singletons must not carry over between tests. The
    # semaphore in particular belongs to one event loop.
    monkeypatch.setattr(main, "_transcription_slots", None)
    monkeypatch.setattr(llm, "_client", None)
    monkeypatch.setattr(transcribe, "_model", None)

    def load_fake_model():
        transcribe._model = FakeWhisperModel()
        return transcribe._model

    # main.load_model is what startup warms; transcribe.load_model is
    # what a transcription calls. Neither may reach for real weights.
    monkeypatch.setattr(main, "load_model", load_fake_model)
    monkeypatch.setattr(transcribe, "load_model", load_fake_model)


@pytest.fixture
def client():
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def gemini(monkeypatch):
    """Switches to the Gemini code path with a fake client behind it."""

    def install(reply) -> FakeGemini:
        fake = FakeGemini(reply)
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setattr(llm, "_client", fake)
        return fake

    return install


@pytest.fixture
def leftover_audio(tmp_path):
    """Returns the downloaded audio files still on disk."""
    return lambda: list(tmp_path.glob("*.webm"))


class _Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _redirect(self, location):
        self.send_response(302)
        self.send_header("Location", location)
        self.end_headers()

    def do_GET(self):
        server = self.server
        server.hits.append(self.path)

        if self.path == "/audio.webm":
            self.send_response(200)
            self.send_header("Content-Length", str(len(SMALL_AUDIO)))
            self.end_headers()
            self.wfile.write(SMALL_AUDIO)
        elif self.path == "/redirect":
            self._redirect("/audio.webm")
        elif self.path == "/redirect-twice":
            self._redirect("/redirect")
        elif self.path == "/redirect-loop":
            self._redirect("/redirect-loop")
        elif self.path == "/redirect-offsite":
            self._redirect("https://internal.example.com/audio.webm")
        elif self.path == "/declares-too-large":
            self.send_response(200)
            self.send_header("Content-Length", str(10**9))
            self.end_headers()
        elif self.path == "/endless":
            # No Content-Length, so the only way to find out it is too
            # big is to count bytes as they arrive.
            self.send_response(200)
            self.end_headers()
            chunk = b"\x00" * 65536
            try:
                while server.bytes_streamed < server.endless_limit:
                    self.wfile.write(chunk)
                    server.bytes_streamed += len(chunk)
            except OSError:
                pass  # the client hung up, which is the point
        else:
            self.send_response(404)
            self.end_headers()


@pytest.fixture
def upstream(monkeypatch):
    """
    A local HTTP server standing in for blob storage.

    Enables ALLOW_LOCAL_AUDIO so the service will talk to it, and makes
    DNS deterministic: localhost resolves to itself and every other name
    to a public address, so no test depends on a real resolver.
    """
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    server.hits = []
    server.bytes_streamed = 0
    server.endless_limit = 64 * 1024 * 1024
    server.url = f"http://127.0.0.1:{server.server_port}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    async def fake_resolve(host, port):
        return ["127.0.0.1"] if host in ("127.0.0.1", "localhost") else ["93.184.216.34"]

    monkeypatch.setenv("ALLOW_LOCAL_AUDIO", "true")
    monkeypatch.setattr(audio_fetch, "_resolve", fake_resolve)

    yield server
    server.shutdown()
    server.server_close()
