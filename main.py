"""
Transcription and note-extraction service.

Takes a signed URL to a consultation recording, transcribes it, and
proposes a structured clinical note. Everything it returns is a
*proposal* — the vet accepts or rejects each field in the web app, and
nothing here writes to the clinical record.
"""

import asyncio
import os
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI, HTTPException, Header
from pydantic import BaseModel, HttpUrl

from audio_fetch import AudioFetchError, download_audio
from chat import ChatRequest, ChatResponse, answer_question
from extract import ProposedNote, extract_note
from transcribe import TranscriptionResult, load_model, transcribe_file

# Each running transcription holds the decoded PCM for a whole recording
# in memory. Bounding how many run at once is what stops a burst of
# requests from getting the instance OOM-killed; the rest wait their turn.
MAX_CONCURRENT_TRANSCRIPTIONS = max(
    1, int(os.environ.get("MAX_CONCURRENT_TRANSCRIPTIONS", "2"))
)
_transcription_slots = asyncio.Semaphore(MAX_CONCURRENT_TRANSCRIPTIONS)

# Strong references to in-flight transcription tasks, so one whose
# request was cancelled is not garbage-collected before it finishes.
_transcription_tasks: set[asyncio.Task] = set()


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
    try:
        path = await download_audio(str(request.audio_url))
    except AudioFetchError as exc:
        raise HTTPException(exc.status_code, exc.detail) from exc

    # Shielded so that a cancelled request (client disconnect, shutdown)
    # cannot release the slot or delete the file while the worker thread
    # is still transcribing — threads cannot be cancelled, so the task
    # has to outlive the request to keep the concurrency bound honest.
    task = asyncio.create_task(_transcribe_and_cleanup(path))
    _transcription_tasks.add(task)
    task.add_done_callback(_transcription_tasks.discard)
    result: TranscriptionResult = await asyncio.shield(task)

    if not result.text.strip():
        raise HTTPException(422, "No speech detected in the recording")

    proposal = await asyncio.to_thread(extract_note, result.text)

    return TranscribeResponse(
        transcript=result.text,
        segments=result.segments,
        extraction=proposal,
    )


async def _transcribe_and_cleanup(path: str) -> TranscriptionResult:
    """Runs Whisper in a worker thread, then always removes the audio."""
    try:
        async with _transcription_slots:
            return await asyncio.to_thread(transcribe_file, path)
    finally:
        with suppress(OSError):
            os.unlink(path)


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    _: None = Depends(require_api_key),
) -> ChatResponse:
    """
    Answers a question about a pet's records.

    The context block is assembled by the calling application, which has
    already established who is asking and filtered the data to what they
    may see. This service does no access control of its own — it cannot,
    having no notion of users — which is exactly why the shared secret
    matters.
    """
    if not request.question.strip():
        raise HTTPException(422, "No question given")
    if len(request.context) > 40_000:
        raise HTTPException(413, "Context too large")

    return await asyncio.to_thread(answer_question, request)
