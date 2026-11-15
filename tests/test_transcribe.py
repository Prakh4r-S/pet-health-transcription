"""/transcribe: URL guarding, the streaming size cap, redirects, and shape."""

import pytest

import audio_fetch
import main
import transcribe
from conftest import AUTH


def post_transcribe(client, url):
    return client.post("/transcribe", json={"audio_url": url}, headers=AUTH)


# --- SSRF guard, with production settings (no ALLOW_LOCAL_AUDIO) -----------


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/audio.webm",                     # loopback
        "https://localhost/audio.webm",
        "https://10.0.0.5/audio.webm",                      # private ranges
        "https://172.16.0.1/audio.webm",
        "https://192.168.1.10/audio.webm",
        "https://169.254.169.254/latest/meta-data/",        # cloud metadata
        "https://[::1]/audio.webm",                         # IPv6 loopback
        "https://[fe80::1]/audio.webm",                     # IPv6 link-local
        "https://example.com/audio.webm",                   # public, not allowlisted
        "https://blob.vercel-storage.com/audio.webm",       # apex is not a subdomain
        "https://evilblob.vercel-storage.com/audio.webm",   # suffix lookalikes
        "https://blob.vercel-storage.com.evil.com/a.webm",
    ],
)
def test_disallowed_hosts_are_rejected(client, url):
    response = post_transcribe(client, url)
    assert response.status_code == 400
    assert response.json() == {"detail": "Audio URL host is not allowed"}


def test_plain_http_is_rejected_even_for_an_allowed_host(client):
    response = post_transcribe(client, "http://abc.public.blob.vercel-storage.com/audio.webm")
    assert response.status_code == 400
    assert response.json() == {"detail": "Audio URL must use HTTPS"}


@pytest.mark.parametrize("address", ["10.0.0.5", "127.0.0.1", "169.254.169.254", "::1", "::ffff:10.0.0.5"])
def test_allowed_host_resolving_to_a_private_address_is_rejected(client, monkeypatch, address):
    # The name passes the allowlist; the address it points at must not.
    async def resolves_inward(host, port):
        return [address]

    monkeypatch.setattr(audio_fetch, "_resolve", resolves_inward)

    response = post_transcribe(client, "https://abc.public.blob.vercel-storage.com/audio.webm")
    assert response.status_code == 400
    assert response.json() == {"detail": "Audio URL resolves to a non-public address"}


def test_one_private_address_among_public_ones_is_enough_to_reject(client, monkeypatch):
    async def mixed(host, port):
        return ["93.184.216.34", "10.0.0.5"]

    monkeypatch.setattr(audio_fetch, "_resolve", mixed)

    response = post_transcribe(client, "https://abc.blob.vercel-storage.com/audio.webm")
    assert response.status_code == 400


def test_allowlist_is_configurable(client, monkeypatch):
    monkeypatch.setenv("ALLOWED_AUDIO_HOSTS", "media.example.org")

    async def public(host, port):
        return ["93.184.216.34"]

    monkeypatch.setattr(audio_fetch, "_resolve", public)

    # The default hosts no longer apply once the variable is set.
    assert post_transcribe(client, "https://abc.blob.vercel-storage.com/a.webm").json() == {
        "detail": "Audio URL host is not allowed"
    }


def test_non_url_input_fails_validation(client):
    assert post_transcribe(client, "not a url").status_code == 422
    assert post_transcribe(client, "file:///etc/passwd").status_code == 422


# --- Size cap --------------------------------------------------------------


def test_cap_is_sixty_megabytes():
    assert audio_fetch.MAX_AUDIO_BYTES == 60 * 1024 * 1024


def test_oversized_stream_is_cut_off_mid_download(client, upstream, leftover_audio):
    # The real limit, against a body that would run to 64 MB and declares
    # no length. Nothing but counting bytes as they arrive can catch it.
    response = post_transcribe(client, f"{upstream.url}/endless")

    assert response.status_code == 413
    assert response.json() == {"detail": "Recording too large"}
    assert leftover_audio() == []


def test_download_stops_as_soon_as_the_cap_is_crossed(client, upstream, monkeypatch, leftover_audio):
    cap = 256 * 1024
    monkeypatch.setattr(audio_fetch, "MAX_AUDIO_BYTES", cap)

    response = post_transcribe(client, f"{upstream.url}/endless")

    assert response.status_code == 413
    # The upstream was willing to send 64 MB. It got to send a few socket
    # buffers past the cap before the connection was dropped.
    assert upstream.bytes_streamed < cap + 16 * 1024 * 1024
    assert leftover_audio() == []


def test_declared_oversize_is_rejected_without_reading_the_body(client, upstream, leftover_audio):
    response = post_transcribe(client, f"{upstream.url}/declares-too-large")

    assert response.status_code == 413
    assert leftover_audio() == []


def test_body_exactly_at_the_cap_is_accepted(client, upstream, monkeypatch):
    from conftest import SMALL_AUDIO

    monkeypatch.setattr(audio_fetch, "MAX_AUDIO_BYTES", len(SMALL_AUDIO))
    monkeypatch.setattr(main, "transcribe_file", lambda path: transcribe.TranscriptionResult(text="ok"))

    assert post_transcribe(client, f"{upstream.url}/audio.webm").status_code == 200


# --- Redirects and upstream errors -----------------------------------------


@pytest.fixture
def stub_whisper(monkeypatch):
    """Replaces transcription and records what it was handed."""
    seen = {}

    def fake_transcribe(path):
        with open(path, "rb") as handle:
            seen["bytes"] = handle.read()
        seen["path"] = path
        return transcribe.TranscriptionResult(
            text="Bruno is limping.",
            segments=[transcribe.Segment(start=0.0, end=1.5, text="Bruno is limping.")],
            language="en",
            duration=1.5,
        )

    monkeypatch.setattr(main, "transcribe_file", fake_transcribe)
    return seen


@pytest.mark.parametrize("path", ["/redirect", "/redirect-twice"])
def test_redirects_are_followed(client, upstream, stub_whisper, path):
    from conftest import SMALL_AUDIO

    response = post_transcribe(client, f"{upstream.url}{path}")

    assert response.status_code == 200
    assert upstream.hits[-1] == "/audio.webm"
    assert stub_whisper["bytes"] == SMALL_AUDIO


def test_redirect_loop_gives_up(client, upstream, leftover_audio):
    response = post_transcribe(client, f"{upstream.url}/redirect-loop")

    assert response.status_code == 400
    assert response.json() == {"detail": "Could not fetch audio (too many redirects)"}
    # The first request plus five redirects, and no more.
    assert len(upstream.hits) == audio_fetch.MAX_REDIRECTS + 1
    assert leftover_audio() == []


def test_redirect_to_a_disallowed_host_is_blocked(client, upstream, leftover_audio):
    # An allowed first hop must not be a way to reach somewhere else.
    response = post_transcribe(client, f"{upstream.url}/redirect-offsite")

    assert response.status_code == 400
    assert response.json() == {"detail": "Audio URL host is not allowed"}
    assert leftover_audio() == []


def test_upstream_error_status_becomes_400(client, upstream, leftover_audio):
    response = post_transcribe(client, f"{upstream.url}/missing.webm")

    assert response.status_code == 400
    assert response.json() == {"detail": "Could not fetch audio (404)"}
    assert leftover_audio() == []


def test_unreachable_upstream_becomes_400(client, upstream):
    upstream.shutdown()
    upstream.server_close()

    response = post_transcribe(client, f"{upstream.url}/audio.webm")
    assert response.status_code == 400


# --- Response shape and cleanup --------------------------------------------


def test_successful_response_shape(client, upstream, stub_whisper, leftover_audio):
    response = post_transcribe(client, f"{upstream.url}/audio.webm")

    assert response.status_code == 200
    body = response.json()

    assert set(body) == {"transcript", "segments", "extraction", "language", "duration"}
    assert body["transcript"] == "Bruno is limping."
    assert body["segments"] == [
        {"start": 0.0, "end": 1.5, "text": "Bruno is limping.", "speaker": None}
    ]
    assert body["language"] == "en"
    assert body["duration"] == 1.5

    extraction = body["extraction"]
    assert set(extraction) == {
        "presenting_complaint", "examination", "assessment", "plan",
        "follow_up", "medications", "uncertain",
    }
    assert isinstance(extraction["medications"], list)
    assert isinstance(extraction["uncertain"], list)

    # The audio existed while Whisper ran and is gone afterwards.
    assert stub_whisper["path"].endswith(".webm")
    assert leftover_audio() == []


def test_extraction_from_the_model_reaches_the_response(client, upstream, stub_whisper, gemini):
    fake = gemini(
        '{"presenting_complaint": "Limping on left hind.", "plan": "Rest.",'
        ' "medications": [{"drug": "meloxicam", "dose": "1.5 mg", "frequency": "once daily",'
        ' "evidence": "start him on meloxicam"}, {"drug": "gabapentin"}]}'
    )

    body = post_transcribe(client, f"{upstream.url}/audio.webm").json()
    extraction = body["extraction"]

    assert extraction["presenting_complaint"] == "Limping on left hind."
    assert extraction["assessment"] is None
    assert extraction["medications"][0] == {
        "drug": "meloxicam", "dose": "1.5 mg", "frequency": "once daily",
        "duration_days": None, "instructions": None, "evidence": "start him on meloxicam",
    }
    # A drug with neither dose nor frequency is flagged for the vet.
    assert extraction["uncertain"] == ["gabapentin: no dose or frequency was stated."]
    # The transcript, and nothing else, is what the model was asked about.
    assert fake.calls[0]["contents"] == "<transcript>\nBruno is limping.\n</transcript>"


def test_unparseable_extraction_degrades_to_an_empty_note(client, upstream, stub_whisper, gemini):
    gemini("I'm sorry, I can't do that.")

    response = post_transcribe(client, f"{upstream.url}/audio.webm")

    assert response.status_code == 200
    extraction = response.json()["extraction"]
    assert extraction["medications"] == []
    assert extraction["uncertain"] == ["The transcript could not be parsed into a note."]


def test_silent_recording_is_422(client, upstream, monkeypatch, leftover_audio):
    monkeypatch.setattr(main, "transcribe_file", lambda path: transcribe.TranscriptionResult(text="  "))

    response = post_transcribe(client, f"{upstream.url}/audio.webm")

    assert response.status_code == 422
    assert response.json() == {"detail": "No speech detected in the recording"}
    assert leftover_audio() == []


def test_undecodable_audio_is_422(client, upstream, leftover_audio):
    # No stub here: the real decoder runs on bytes that are not audio.
    response = post_transcribe(client, f"{upstream.url}/audio.webm")

    assert response.status_code == 422
    assert response.json() == {"detail": "Recording could not be decoded as audio"}
    assert leftover_audio() == []


def test_audio_is_removed_when_transcription_crashes(upstream, monkeypatch, leftover_audio):
    from fastapi.testclient import TestClient

    def crash(path):
        raise RuntimeError("model fault")

    monkeypatch.setattr(main, "transcribe_file", crash)

    with TestClient(main.app, raise_server_exceptions=False) as client:
        response = post_transcribe(client, f"{upstream.url}/audio.webm")

    assert response.status_code == 500
    assert leftover_audio() == []


@pytest.mark.parametrize(
    "failure, expected_status",
    [("timeout", 504), ("server_error", 502)],
)
def test_model_failures_during_extraction_map_to_gateway_errors(
    client, upstream, stub_whisper, gemini, failure, expected_status
):
    import httpx
    from google.genai import errors

    gemini(httpx.ReadTimeout("slow") if failure == "timeout" else errors.ServerError(500, {}))

    assert post_transcribe(client, f"{upstream.url}/audio.webm").status_code == expected_status
