"""Bearer-token auth, startup validation, and the two probes."""

import threading
import time

import pytest
from fastapi.testclient import TestClient

import main
import transcribe
from conftest import API_KEY, AUTH

CHAT_BODY = {"question": "When is Bruno's booster due?", "context": "Bruno: booster due 4 March."}


def post_chat(client, authorization=None):
    headers = {} if authorization is None else {"Authorization": authorization}
    return client.post("/chat", json=CHAT_BODY, headers=headers)


def test_correct_token_is_accepted(client):
    assert post_chat(client, f"Bearer {API_KEY}").status_code == 200


@pytest.mark.parametrize("scheme", ["bearer", "BEARER", "BeArEr"])
def test_bearer_scheme_is_case_insensitive(client, scheme):
    assert post_chat(client, f"{scheme} {API_KEY}").status_code == 200


def test_extra_whitespace_around_token_is_tolerated(client):
    assert post_chat(client, f"Bearer   {API_KEY} ").status_code == 200


@pytest.mark.parametrize(
    "authorization",
    [
        None,                       # header absent
        "",                         # header empty
        "Bearer",                   # scheme with no token
        "Bearer wrong-key",
        f"Bearer {API_KEY[:-1]}",   # one character short
        f"Bearer {API_KEY}x",       # one character long
        f"Bearer {API_KEY.upper()}",  # the token itself is case-sensitive
        f"Basic {API_KEY}",         # right token, wrong scheme
        API_KEY,                    # token with no scheme
    ],
)
def test_missing_or_wrong_token_is_rejected(client, authorization):
    response = post_chat(client, authorization)
    assert response.status_code == 401
    assert response.json() == {"detail": "Unauthorized"}


def test_transcribe_requires_auth_before_fetching_anything(client, upstream):
    response = client.post("/transcribe", json={"audio_url": f"{upstream.url}/audio.webm"})
    assert response.status_code == 401
    assert upstream.hits == []


def test_service_refuses_to_start_without_api_key(monkeypatch):
    monkeypatch.delenv("SERVICE_API_KEY")
    with pytest.raises(RuntimeError, match="SERVICE_API_KEY"):
        with TestClient(main.app):
            pass


def test_service_refuses_to_start_without_gemini_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        with TestClient(main.app):
            pass


def test_health_needs_no_auth(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_is_503_until_the_model_has_loaded(monkeypatch):
    release = threading.Event()

    def slow_load():
        release.wait(timeout=10)
        transcribe._model = object()

    monkeypatch.setattr(main, "load_model", slow_load)

    with TestClient(main.app) as client:
        warming = client.get("/ready")
        assert warming.status_code == 503
        assert warming.json() == {"status": "loading"}
        # Liveness must not depend on the model.
        assert client.get("/health").status_code == 200

        release.set()
        deadline = time.monotonic() + 5
        while client.get("/ready").status_code != 200:
            assert time.monotonic() < deadline, "/ready never became 200"
            time.sleep(0.02)

        assert client.get("/ready").json() == {"status": "ready"}


def test_ready_stays_503_if_the_model_fails_to_load(monkeypatch):
    def broken_load():
        raise RuntimeError("weights missing")

    monkeypatch.setattr(main, "load_model", broken_load)

    with TestClient(main.app) as client:
        time.sleep(0.1)
        assert client.get("/ready").status_code == 503
        assert client.get("/health").status_code == 200


def test_every_response_carries_a_request_id(client):
    generated = post_chat(client, f"Bearer {API_KEY}").headers["x-request-id"]
    assert len(generated) == 32

    echoed = client.post("/chat", json=CHAT_BODY, headers={**AUTH, "X-Request-ID": "trace-abc.123"})
    assert echoed.headers["x-request-id"] == "trace-abc.123"

    # Rejected requests are traceable too.
    assert "x-request-id" in post_chat(client, "Bearer wrong-key").headers
