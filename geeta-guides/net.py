"""
net.py — the HTTP fetching this project kept getting wrong, fixed once.

Every lesson here was learned by something breaking:

  * SSL — the python.org macOS build ships an empty CA bundle until you run
    `Install Certificates.command`, so HTTPS dies with
    CERTIFICATE_VERIFY_FAILED. We prefer certifi's bundle when importable and
    fail with the fix in the message. We never disable verification, because
    that swaps a loud error for a silent vulnerability.

  * Retries — GitHub Pages throttles at roughly 60 rapid requests, and one of
    our jobs makes 700. Exponential backoff on 429/502/503/504.

  * 404 vs 503 — completely different meanings. A 404 says this resource does
    not exist (for us: a verse absent from the source's recension, a real
    finding). A 503 says slow down. Conflating them hides bugs.

  * URLError wrapping — urllib wraps SSL failures inside URLError, so you have
    to inspect `.reason`. Catching ssl.SSLCertVerificationError directly does
    not work, and HTTPError must be caught before URLError because it is a
    subclass of it.

  * Atomic cache writes — a run killed mid-write leaves a truncated file that
    poisons every later run. Write to a temp path, then os.replace.

  * Corrupt cache recovery — treat an unparseable cache entry as a miss rather
    than trusting the cache blindly.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.error
import urllib.request

DEFAULT_USER_AGENT = "geeta-guides/0.1 (personal research project)"


def ssl_context() -> ssl.SSLContext | None:
    """Prefer certifi's CA bundle; fall back to the system default.

    Returns None to mean "use urllib's default context", which is correct on a
    properly configured machine.
    """
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


SSL_CONTEXT = ssl_context()

_CERT_HELP = (
    "SSL certificate verification failed.\n\n"
    "  This Python cannot verify HTTPS certificates. On a python.org macOS\n"
    "  build the CA bundle ships empty. Cheapest fix, no sudo needed:\n\n"
    "      .venv/bin/pip install certifi\n\n"
    "  This module then picks it up automatically. The system-wide alternative\n"
    "  is `/Applications/Python\\ 3.x/Install\\ Certificates.command`, which may\n"
    "  need sudo.\n"
)


class FetchError(RuntimeError):
    """Raised when a fetch fails in a way retrying will not fix."""


class NotFound(FetchError):
    """HTTP 404 — the resource genuinely is not there. Usually a real finding."""


def fetch_bytes(
    url: str,
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout: float = 30.0,
    max_retries: int = 6,
    polite_delay: float = 0.15,
) -> bytes:
    """GET `url`, retrying transient failures with exponential backoff.

    Raises
    ------
    NotFound    on 404 — do not retry, the resource is absent
    FetchError  on a certificate problem, an unretryable status, or exhaustion
    """
    req = urllib.request.Request(url, headers={"User-Agent": user_agent})
    delay = 1.0
    last_err: Exception | None = None

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CONTEXT) as resp:
                data = resp.read()
            time.sleep(polite_delay)
            return data

        # HTTPError first: it subclasses URLError.
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code == 404:
                raise NotFound(f"HTTP 404: {url}") from e
            if e.code in (429, 502, 503, 504):
                print(f"\n    {e.code} on {url.rsplit('/', 2)[-2:]}, "
                      f"backing off {delay:.0f}s ({attempt + 1}/{max_retries})",
                      end="", flush=True)
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
                continue
            raise FetchError(f"HTTP {e.code}: {url}") from e

        except urllib.error.URLError as e:
            last_err = e
            if isinstance(e.reason, ssl.SSLCertVerificationError):
                raise FetchError(_CERT_HELP) from e
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue

        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(delay)
            delay = min(delay * 2, 30.0)
            continue

    raise FetchError(f"gave up on {url} after {max_retries} attempts: {last_err}")


def fetch_text(url: str, *, encoding: str = "utf-8", **kw) -> str:
    """GET `url` and decode as text."""
    return fetch_bytes(url, **kw).decode(encoding, errors="replace")


def fetch_json(url: str, *, cache_path: str | None = None, **kw) -> dict:
    """GET `url` as JSON, optionally caching to `cache_path`.

    A cached file is used when it parses. When it does not — the signature of a
    run killed mid-write — it is treated as a miss and refetched.
    """
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(f"\n    corrupt cache {os.path.basename(cache_path)}, refetching",
                  end="", flush=True)
            try:
                os.remove(cache_path)
            except OSError:
                pass  # read-only mount; the atomic write below replaces it

    data = json.loads(fetch_text(url, **kw))

    if cache_path:
        write_json_atomic(cache_path, data)
    return data


def write_json_atomic(path: str, obj) -> None:
    """Write JSON so an interruption cannot leave a half-written file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)
