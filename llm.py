"""
The one place that talks to the language model.

Both extraction and chat go through `generate_json`, so they share a
client, a timeout, and a single definition of what an upstream failure
looks like to the rest of the service.
"""

import logging
import threading
import time

import config

logger = logging.getLogger(__name__)

_client = None
_client_lock = threading.Lock()


class LLMError(Exception):
    """The language model could not be reached or returned an error."""


class LLMTimeoutError(LLMError):
    """The language model did not answer within the deadline."""


def get_client():
    """
    Returns the shared Gemini client, creating it on first use.

    The client owns a connection pool, so building one per request throws
    away keep-alive connections and pays a TLS handshake every time. It
    reads GEMINI_API_KEY (or GOOGLE_API_KEY) from the environment.
    """
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from google import genai

                _client = genai.Client()
    return _client


def generate_json(
    *,
    model: str,
    system: str,
    contents,
    temperature: float,
    max_output_tokens: int,
    attempts: int = 1,
) -> str:
    """
    Asks Gemini for a JSON reply and returns its text.

    Retries only on 503 or an empty reply, and only up to `attempts`.
    Raises LLMTimeoutError or LLMError; never returns an empty string.
    """
    import httpx
    from google.genai import errors, types

    client = get_client()
    config_ = types.GenerateContentConfig(
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        # Constrains the decoder to emit valid JSON, which removes a
        # whole class of parse failure before it happens.
        response_mime_type="application/json",
        # Milliseconds. Without this a stalled upstream holds the request
        # (and its worker thread) open indefinitely.
        http_options=types.HttpOptions(timeout=int(config.llm_timeout_seconds() * 1000)),
    )

    last_error = "no attempt made"
    for attempt in range(attempts):
        if attempt:
            # Backoff sits before a retry rather than after an attempt,
            # so the final failure is reported without a pointless wait.
            time.sleep(2 ** (attempt - 1))
        try:
            response = client.models.generate_content(
                model=model, contents=contents, config=config_
            )
        except httpx.TimeoutException as exc:
            # Not retried: another full deadline on top of this one would
            # outlast any caller still waiting.
            logger.warning("Gemini request timed out (model=%s)", model)
            raise LLMTimeoutError("Language model timed out") from exc
        except errors.APIError as exc:
            if exc.code == 504:
                logger.warning("Gemini deadline exceeded (model=%s)", model)
                raise LLMTimeoutError("Language model timed out") from exc
            # The free tier returns 503 under load often enough that a
            # single attempt is not a fair test of the model.
            if exc.code != 503:
                logger.error("Gemini request failed (model=%s): %s", model, exc)
                raise LLMError("Language model request failed") from exc
            last_error = str(exc)
        except httpx.HTTPError as exc:
            logger.error("Gemini request failed (model=%s): %r", model, exc)
            raise LLMError("Language model could not be reached") from exc
        else:
            text = _text_from(response)
            if text.strip():
                return text
            last_error = "model returned no text"

    logger.error("Gemini failed after %d attempt(s) (model=%s): %s", attempts, model, last_error)
    raise LLMError("Language model is unavailable")


def _text_from(response) -> str:
    """
    Pulls text out of a response that may also contain thinking parts.

    Newer Gemini models return reasoning alongside the answer, and
    `response.text` warns when it has to skip non-text parts. Reading the
    parts directly avoids the warning and, more importantly, avoids
    returning an empty string when the model spent its whole output
    budget thinking.
    """
    if getattr(response, "text", None):
        return response.text

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            if getattr(part, "thought", False):
                continue
            if getattr(part, "text", None):
                parts.append(part.text)
    return "".join(parts)
