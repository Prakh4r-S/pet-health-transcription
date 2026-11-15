"""
Settings, read from the environment each time they are asked for.

Nothing here is captured at import time. That is deliberate: a value
frozen when a module is first imported cannot be overridden afterwards,
which is how `eval/run_eval.py --provider/--model` came to be silently
ignored. Reading lazily costs a dictionary lookup and removes the whole
class of bug.
"""

import os

DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"
DEFAULT_ALLOWED_AUDIO_HOSTS = "*.public.blob.vercel-storage.com,*.blob.vercel-storage.com"

# "mock" returns canned output without calling any API. It exists so the
# service and the eval harness can be exercised with no credentials.
LLM_PROVIDERS = ("gemini", "mock")


def _get(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _positive_number(name: str, default: float) -> float:
    raw = _get(name)
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero, got {raw!r}")
    return value


def service_api_key() -> str:
    return _get("SERVICE_API_KEY")


def llm_provider() -> str:
    provider = _get("LLM_PROVIDER", "gemini").lower()
    if provider not in LLM_PROVIDERS:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER {provider!r}; expected one of {', '.join(LLM_PROVIDERS)}"
        )
    return provider


def extraction_model() -> str:
    return _get("EXTRACTION_MODEL", DEFAULT_GEMINI_MODEL)


def chat_model() -> str:
    return _get("CHAT_MODEL") or extraction_model()


def llm_timeout_seconds() -> float:
    """Deadline for a single request to the language model."""
    return _positive_number("LLM_TIMEOUT_SECONDS", 60.0)


def whisper_model() -> str:
    # "base" runs on a modest CPU in roughly real time and is adequate for
    # clear two-person speech. "small" is noticeably better on veterinary
    # vocabulary and roughly three times slower; "medium" needs more RAM
    # than most small instances allow. Set via env so the trade-off can be
    # measured rather than guessed at.
    return _get("WHISPER_MODEL", "base")


def max_concurrent_transcriptions() -> int:
    return max(1, int(_positive_number("MAX_CONCURRENT_TRANSCRIPTIONS", 2)))


def allowed_audio_hosts() -> list[str]:
    raw = _get("ALLOWED_AUDIO_HOSTS", DEFAULT_ALLOWED_AUDIO_HOSTS)
    return [h.strip().lower().rstrip(".") for h in raw.split(",") if h.strip()]


def allow_local_audio() -> bool:
    return _get("ALLOW_LOCAL_AUDIO").lower() == "true"


def validate() -> None:
    """
    Raises RuntimeError if the service cannot run as configured.

    Called once at startup so a missing secret stops the deploy rather
    than surfacing later as a 500 on every request.
    """
    if not service_api_key():
        raise RuntimeError("SERVICE_API_KEY is not configured")
    llm_provider()
    llm_timeout_seconds()
    max_concurrent_transcriptions()
