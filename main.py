"""
Transcription and note-extraction service.

Takes a signed URL to a consultation recording, transcribes it, and
proposes a structured clinical note. Everything it returns is a
*proposal* — the vet accepts or rejects each field in the web app, and
nothing here writes to the clinical record.
"""

import asyncio
import hmac
import logging
import os
from contextlib import asynccontextmanager, suppress

from fastapi import Depends, FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl

import config
import llm
from audio_fetch import AudioFetchError, download_audio
from chat import ChatRequest, ChatResponse, answer_question
from extract import ProposedNote, extract_note
from transcribe import (
    AudioDecodeError,
    Segment,
    TranscriptionResult,
    is_model_loaded,
    load_model,
    transcribe_file,
)

logger = logging.getLogger(__name__)

# Each running transcription holds the decoded PCM for a whole recording
# in memory. Bounding how many run at once is what stops a burst of
# requests from getting the instance OOM-killed; the rest wait their turn.
_transcription_slots: asyncio.Semaphore | None = None

# Strong references to background tasks (model warm-up, in-flight
# transcriptions), so none is garbage-collected before it finishes.
_background_tasks: set[asyncio.Task] = set()


def _slots() -> asyncio.Semaphore:
    global _transcription_slots
    if _transcription_slots is None:
        _transcription_slots = asyncio.Semaphore(config.max_concurrent_transcriptions())
    return _transcription_slots


def _keep(task: asyncio.Task) -> asyncio.Task:
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def _log_warmup_result(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    if task.exception() is not None:
        # /ready keeps answering 503, which is what stops traffic being
        # routed here; this is the line that says why.
        logger.error("Whisper model failed to load", exc_info=task.exception())
    else:
        logger.info("Whisper model loaded (%s)", config.whisper_model())


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail on boot, not on the first request: a deploy with a missing
    # secret should never start taking traffic.
    config.validate()
    if config.llm_provider() == "gemini":
        try:
            llm.get_client()
        except Exception as exc:
            raise RuntimeError(
                "Gemini client could not be created; is GEMINI_API_KEY set?"
            ) from exc

    # Whisper weights take several seconds to load and hundreds of MB of
    # RAM, so they are loaded once, here, not per request. It happens in
    # the background so the port opens immediately and /health answers
    # during warm-up; /ready reports when the model is actually usable.
    warmup = _keep(asyncio.create_task(asyncio.to_thread(load_model)))
    warmup.add_done_callback(_log_warmup_result)
    yield


app = FastAPI(title="Pet Health transcription", lifespan=lifespan)


@app.exception_handler(llm.LLMError)
async def llm_error_handler(_: Request, exc: llm.LLMError) -> JSONResponse:
    # 504 when the model did not answer in time, 502 for any other
    # upstream failure. The cause is logged where it was raised; the
    # caller gets a stable message, not the provider's error text.
    status = 504 if isinstance(exc, llm.LLMTimeoutError) else 502
    return JSONResponse(status_code=status, content={"detail": str(exc)})


@app.exception_handler(AudioDecodeError)
async def decode_error_handler(_: Request, exc: AudioDecodeError) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": str(exc)})


def require_api_key(authorization: str = Header(default="")) -> None:
    """
    Shared-secret auth.

    This service holds no user context and makes no authorisation
    decisions of its own — the Next.js app has already established that
    the caller is the vet on the appointment. All this check does is
    ensure the request came from that app and not from the open
    internet.
    """
    expected = config.service_api_key()
    if not expected:
        # Unreachable in a running service (startup refuses to boot
        # without the key); kept so the check can never fail open.
        raise HTTPException(500, "SERVICE_API_KEY is not configured")

    scheme, _, token = authorization.strip().partition(" ")
    # Compared as bytes in constant time, so response timing does not
    # reveal how much of a guessed token was correct.
    supplied = token.strip().encode()
    if scheme.lower() != "bearer" or not hmac.compare_digest(supplied, expected.encode()):
        raise HTTPException(401, "Unauthorized")


class TranscribeRequest(BaseModel):
    audio_url: HttpUrl


class TranscribeResponse(BaseModel):
    transcript: str
    segments: list[Segment]
    extraction: ProposedNote
    language: str | None = None
    duration: float | None = None


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. Says nothing about the model."""
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    """Readiness: 200 once the Whisper model is loaded, 503 until then."""
    if is_model_loaded():
        return JSONResponse({"status": "ready"})
    return JSONResponse({"status": "loading"}, status_code=503)


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
    task = _keep(asyncio.create_task(_transcribe_and_cleanup(path)))
    result: TranscriptionResult = await asyncio.shield(task)

    if not result.text.strip():
        raise HTTPException(422, "No speech detected in the recording")

    proposal = await asyncio.to_thread(extract_note, result.text)

    return TranscribeResponse(
        transcript=result.text,
        segments=result.segments,
        extraction=proposal,
        language=result.language,
        duration=result.duration,
    )


async def _transcribe_and_cleanup(path: str) -> TranscriptionResult:
    """Runs Whisper in a worker thread, then always removes the audio."""
    try:
        async with _slots():
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
