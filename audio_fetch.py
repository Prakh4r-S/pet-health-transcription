"""
Fetches a consultation recording to a temp file, safely.

The URL comes from the caller, so it is treated as hostile: it must be
HTTPS, on an allowlisted host, and resolve only to public addresses, and
those checks are repeated on every redirect hop. The body is streamed
straight to disk and the size cap is enforced as bytes arrive, so an
oversized recording is never held in memory.
"""

import asyncio
import ipaddress
import os
import socket
import tempfile
from contextlib import suppress

import httpx

import config

MAX_AUDIO_BYTES = 60 * 1024 * 1024
MAX_REDIRECTS = 5
# Read timeout is per chunk, so a server dripping bytes could hold a
# request open indefinitely without an overall deadline as well.
DOWNLOAD_TIMEOUT_SECONDS = 120
TIMEOUT = httpx.Timeout(60.0, connect=10.0)

class AudioFetchError(Exception):
    """A fetch failure that maps directly onto an HTTP error response."""

    def __init__(self, status_code: int, detail: str):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def _host_allowed(host: str) -> bool:
    for pattern in config.allowed_audio_hosts():
        if pattern.startswith("*."):
            # "*.example.com" matches subdomains only, not the apex.
            if host.endswith(pattern[1:]) and len(host) > len(pattern) - 1:
                return True
        elif host == pattern:
            return True
    return False


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address.split("%")[0])
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    # is_global is False for private, loopback, link-local (including the
    # cloud metadata address), CGNAT, and reserved ranges.
    return ip.is_global


async def _resolve(host: str, port: int) -> list[str]:
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(
            host, port, type=socket.SOCK_STREAM
        )
    except (socket.gaierror, UnicodeError) as exc:
        raise AudioFetchError(400, "Audio URL host could not be resolved") from exc
    return [info[4][0] for info in infos]


async def validate_audio_url(url: httpx.URL) -> None:
    """
    Raises AudioFetchError(400) unless the URL may be fetched.

    ALLOW_LOCAL_AUDIO=true is for local tests only: it additionally
    permits plain HTTP and hosts that resolve to non-public addresses
    (e.g. http://localhost:9000/sample.webm). Public hosts still have to
    be on the allowlist.
    """
    allow_local = config.allow_local_audio()

    if url.scheme != "https" and not (allow_local and url.scheme == "http"):
        raise AudioFetchError(400, "Audio URL must use HTTPS")

    host = (url.host or "").lower().rstrip(".")
    if not host:
        raise AudioFetchError(400, "Audio URL has no host")

    allowlisted = _host_allowed(host)
    if not allowlisted and not allow_local:
        # Rejected before any DNS lookup, so the service cannot be made
        # to resolve arbitrary names either.
        raise AudioFetchError(400, "Audio URL host is not allowed")

    port = url.port or (443 if url.scheme == "https" else 80)
    addresses = await _resolve(host, port)
    if not addresses:
        raise AudioFetchError(400, "Audio URL host could not be resolved")
    public = [_is_public(a) for a in addresses]

    if allowlisted:
        if all(public) or allow_local:
            return
        raise AudioFetchError(400, "Audio URL resolves to a non-public address")

    if allow_local and not any(public):
        return

    raise AudioFetchError(400, "Audio URL host is not allowed")


async def _guard_request(request: httpx.Request) -> None:
    # Runs for the initial request and for every redirect hop, so a
    # redirect cannot be used to reach a host the first URL could not.
    await validate_audio_url(request.url)


async def download_audio(url: str) -> str:
    """
    Streams the recording to a temp file and returns its path.

    The caller owns the file and must delete it. On any failure the file
    is removed here before the error propagates.
    """
    handle = tempfile.NamedTemporaryFile(suffix=".webm", delete=False)
    path = handle.name
    try:
        async with asyncio.timeout(DOWNLOAD_TIMEOUT_SECONDS):
            async with httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                event_hooks={"request": [_guard_request]},
            ) as client:
                async with client.stream("GET", url) as response:
                    if response.status_code != 200:
                        raise AudioFetchError(
                            400, f"Could not fetch audio ({response.status_code})"
                        )

                    declared = response.headers.get("content-length", "")
                    if declared.isdigit() and int(declared) > MAX_AUDIO_BYTES:
                        raise AudioFetchError(413, "Recording too large")

                    received = 0
                    async for chunk in response.aiter_bytes(64 * 1024):
                        received += len(chunk)
                        if received > MAX_AUDIO_BYTES:
                            raise AudioFetchError(413, "Recording too large")
                        handle.write(chunk)
        handle.close()
        return path
    except BaseException as exc:
        handle.close()
        with suppress(OSError):
            os.unlink(path)
        if isinstance(exc, TimeoutError):
            raise AudioFetchError(400, "Could not fetch audio (timed out)") from exc
        if isinstance(exc, httpx.TooManyRedirects):
            raise AudioFetchError(400, "Could not fetch audio (too many redirects)") from exc
        if isinstance(exc, (httpx.HTTPError, httpx.InvalidURL)):
            raise AudioFetchError(400, "Could not fetch audio") from exc
        raise
