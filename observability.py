"""
Structured logging and request IDs.

One JSON object per line on stdout, which is what Cloud Run's log
collector parses: `severity` and `message` are recognised, and every
other key becomes a queryable field.

What is deliberately never logged: transcript text, chat questions,
record context, model output, audio URLs (they are signed, so the URL is
itself a credential), and API keys. Log lines carry shapes and timings,
not content.
"""

import json
import logging
import re
import sys
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone

import config

REQUEST_ID_HEADER = "x-request-id"

# A caller-supplied ID is echoed into logs and a response header, so it
# is only accepted if it is short and boring; anything else is replaced.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

# Probes fire every few seconds and would bury everything else.
_UNLOGGED_PATHS = {"/health", "/ready"}

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_request_fields: ContextVar[dict | None] = ContextVar("request_fields", default=None)

access_logger = logging.getLogger("access")


def annotate(**fields) -> None:
    """
    Adds fields to the current request's log line.

    Safe to call from anywhere under a request, including worker
    threads; a no-op outside one. Pass metadata only — never content.
    """
    current = _request_fields.get()
    if current is not None:
        current.update({k: v for k, v in fields.items() if v is not None})


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "severity": record.levelname,
            "time": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = _request_id.get()
        if request_id:
            entry["request_id"] = request_id
        entry.update(getattr(record, "fields", {}))
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(config.log_level())

    # Uvicorn installs its own plain-text handlers; route its messages
    # through ours instead so every line on stdout is JSON.
    for name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = []
        uvicorn_logger.propagate = True

    # Replaced by RequestLogMiddleware, which knows the request ID,
    # route template, and model metadata that uvicorn's line does not.
    uvicorn_access = logging.getLogger("uvicorn.access")
    uvicorn_access.handlers = []
    uvicorn_access.propagate = False

    # At INFO these log every outbound URL — including signed audio URLs.
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


class RequestLogMiddleware:
    """
    Assigns each request an ID, returns it as X-Request-ID, and writes
    one log line when the request finishes.

    Written as plain ASGI rather than BaseHTTPMiddleware so the handler
    runs in the same task and context: the request ID set here is visible
    to every log call made while handling the request.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        supplied = dict(scope["headers"]).get(REQUEST_ID_HEADER.encode(), b"").decode("latin-1")
        request_id = supplied if _SAFE_REQUEST_ID.match(supplied) else uuid.uuid4().hex

        fields: dict = {}
        _request_id.set(request_id)
        _request_fields.set(fields)

        status = 500
        started = time.perf_counter()

        async def send_with_request_id(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", []).append(
                    (REQUEST_ID_HEADER.encode(), request_id.encode())
                )
            await send(message)

        error: BaseException | None = None
        try:
            await self.app(scope, receive, send_with_request_id)
        except BaseException as exc:
            # Unhandled: the server's error layer answers 500 after this
            # re-raises, and logs the traceback itself.
            status, error = 500, exc
            raise
        finally:
            if scope["path"] not in _UNLOGGED_PATHS:
                # The route template ("/transcribe"), not the raw path:
                # stable to group by, and never carries caller input.
                route = getattr(scope.get("route"), "path", None) or "unmatched"
                entry = {
                    "method": scope["method"],
                    "route": route,
                    "status": status,
                    "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    **fields,
                }
                if error is not None:
                    entry["error_type"] = type(error).__name__
                level = logging.ERROR if status >= 500 else logging.INFO
                access_logger.log(
                    level, "%s %s %d", scope["method"], route, status, extra={"fields": entry}
                )
