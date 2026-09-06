"""
Transcription and note-extraction service.

Takes a signed URL to a consultation recording, transcribes it, and
proposes a structured clinical note. Everything it returns is a
*proposal* — the vet accepts or rejects each field in the web app, and
nothing here writes to the clinical record.
"""

import os
import tempfile
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl

from extract import ProposedNote, extract_note
from transcribe import TranscriptionResult, load_model, transcribe_file

MAX_AUDIO_BYTES = 60 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 120


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Whisper weights take several seconds to load and hundreds of MB of
    # RAM. Loading once at startup rather than per request is the
    # difference between a 2-second response and a 20-second one.
    load_model()
    yield


app = FastAPI(title="Pet Health transcription", lifespan=lifespan)


def require_api_key(authorization: str = Header(default="")) -> None:
    """
    Shared-secret auth.

    This service holds no user context and makes no authorisation
    decisions of its own — the Next.js app has already established that
    the caller is the vet on the appointment. All this check does is
    ensure the request came from that app and not from the open
    internet.
    """
    expected = os.environ.get("SERVICE_API_KEY")
    if not expected:
        raise HTTPException(500, "SERVICE_API_KEY is not configured")
    if authorization != f"Bearer {expected}":
        raise HTTPException(401, "Unauthorized")


class TranscribeRequest(BaseModel):
    audio_url: HttpUrl


class TranscribeResponse(BaseModel):
    transcript: str
    segments: list[dict]
    extraction: ProposedNote


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe(
    request: TranscribeRequest,
    _: None = Depends(require_api_key),
) -> TranscribeResponse:
    audio = await _download(str(request.audio_url))

    # NamedTemporaryFile because faster-whisper reads from a path — it
    # shells out to ffmpeg for decoding rather than accepting bytes.
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=True) as handle:
        handle.write(audio)
        handle.flush()
        result: TranscriptionResult = transcribe_file(handle.name)

    if not result.text.strip():
        raise HTTPException(422, "No speech detected in the recording")

    proposal = extract_note(result.text)

    return TranscribeResponse(
        transcript=result.text,
        segments=result.segments,
        extraction=proposal,
    )


async def _download(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=DOWNLOAD_TIMEOUT_SECONDS) as client:
        response = await client.get(url)
        if response.status_code != 200:
            raise HTTPException(400, f"Could not fetch audio ({response.status_code})")
        if len(response.content) > MAX_AUDIO_BYTES:
            raise HTTPException(413, "Recording too large")
        return response.content
