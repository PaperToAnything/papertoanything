"""The local bridge: an HTTP server on 127.0.0.1 that serves only data.

It serves no web page. The Lab (https://lab.papertoanything.com) is opened in
your browser with ``?bridge=http://127.0.0.1:PORT&token=...`` and reads these
endpoints directly:

    GET /spec              BridgeHello JSON
    GET /trace             latest TraceFrame JSON (204 when none yet)
    GET /health?since=N    {"frames": [HealthFrame, ...]} with step > N
    GET /events            Server-Sent Events: hello, spec, frame, health, bye
    GET /                  a plain-text note (no token needed)

The data endpoints require the random per-session token, as ``?token=``, the
header ``X-PTA-Token`` or ``Authorization: Bearer``. Requests whose Host
header is not this loopback address are refused (DNS-rebinding guard). A
request that carries an Origin header is refused unless the origin is this
server itself or one of the allowed Lab origins; only those origins receive
CORS headers (never ``*``).

Runs in a daemon thread, so it never blocks a notebook or script.
"""

from __future__ import annotations

import hmac
import json
import os
import queue
import secrets
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Deque, Dict, List, Optional, Tuple, Union
from urllib.parse import parse_qs, urlsplit

from .spec import BridgeHello, HealthFrame, ModelSpec, TraceFrame, dumps

__all__ = ["LocalServer", "LAB_URL", "LAB_ORIGINS"]

HEALTH_HISTORY = 5000
KEEPALIVE_SECONDS = 15.0

#: The hosted Lab that opens a bridge.
LAB_URL = "https://lab.papertoanything.com"
#: Origins allowed to read the bridge (exact match): the hosted Lab and the Lab dev server.
LAB_ORIGINS = ("https://lab.papertoanything.com", "http://localhost:5180")


def _origin_of(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}".lower()


class _Subscriber:
    def __init__(self) -> None:
        self.q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)

    def put(self, msg: Optional[bytes]) -> None:
        while True:
            try:
                self.q.put_nowait(msg)
                return
            except queue.Full:  # a slow tab drops the oldest, never blocks training
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass


def _sse(event: str, data: Any, id_: Optional[Union[int, str]] = None) -> bytes:
    lines = [f"event: {event}"]
    if id_ is not None:
        lines.append(f"id: {id_}")
    lines.append("data: " + (data if isinstance(data, str) else dumps(data)))
    return ("\n".join(lines) + "\n\n").encode("utf-8")


class LocalServer:
    """Local same-origin bridge. Use ``start()``; it returns ``self``."""

    def __init__(
        self,
        spec: Optional[ModelSpec] = None,
        lab_url: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 0,
        token: Optional[str] = None,
        name: Optional[str] = None,
    ) -> None:
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("the bridge only listens on loopback (127.0.0.1)")
        self.host = "127.0.0.1" if host == "localhost" else host
        self.requested_port = port
        self.token = token or secrets.token_urlsafe(24)
        self.lab_url = (lab_url or os.environ.get("PTA_LAB_URL") or LAB_URL).rstrip("/")
        self.allowed_origins = {o.lower() for o in LAB_ORIGINS} | {_origin_of(self.lab_url)}
        self.name = name or (spec.name if spec else "model")
        self._spec = spec
        self._trace: Optional[TraceFrame] = None
        self._trace_json: Optional[str] = None
        self._health: Deque[Tuple[int, str]] = deque(maxlen=HEALTH_HISTORY)
        self._subs: List[_Subscriber] = []
        self._lock = threading.Lock()
        self._closed = threading.Event()
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> "LocalServer":
        if self._httpd is not None:
            return self
        server = self

        class Handler(_Handler):
            bridge = server

        httpd = ThreadingHTTPServer((self.host, self.requested_port), Handler)
        httpd.daemon_threads = True
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever, name="pta-bridge", daemon=True)
        self._thread.start()
        return self

    @property
    def port(self) -> int:
        if self._httpd is None:
            raise RuntimeError("server not started")
        return int(self._httpd.server_address[1])

    @property
    def origin(self) -> str:
        h = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{h}:{self.port}"

    @property
    def url(self) -> str:
        """The Lab page to open: ``https://lab.papertoanything.com/?bridge=http://127.0.0.1:PORT&token=...``."""
        return f"{self.lab_url}/?bridge={self.origin}&token={self.token}"

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._broadcast(_sse("bye", {"reason": "closed"}))
        with self._lock:
            for s in self._subs:
                s.put(None)
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()

    def __enter__(self) -> "LocalServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def serve_forever(self) -> None:
        """Block until Ctrl+C (for the CLI)."""
        self.start()
        try:
            while not self._closed.wait(0.5):
                pass
        except KeyboardInterrupt:
            pass
        finally:
            self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed.is_set() else ("running" if self._httpd else "new")
        return f"<ptoa.LocalServer {state} {self.url if self._httpd else ''}>"

    # ── data ─────────────────────────────────────────────────────────────────

    @property
    def spec(self) -> Optional[ModelSpec]:
        return self._spec

    def hello(self) -> Optional[BridgeHello]:
        return None if self._spec is None else BridgeHello(self._spec)

    def set_spec(self, spec: ModelSpec) -> None:
        self._spec = spec
        self._broadcast(_sse("spec", BridgeHello(spec)))

    def push(self, frame: Union[TraceFrame, Dict[str, Any]]) -> None:
        """Publish a TraceFrame (``event: frame``) and make it ``/trace``."""
        if isinstance(frame, dict):
            frame = TraceFrame.from_dict(frame)
        data = dumps(frame)
        self._trace, self._trace_json = frame, data
        self._broadcast(_sse("frame", data, frame.step))

    def push_health(self, frame: Union[HealthFrame, Dict[str, Any]]) -> None:
        """Publish a HealthFrame (``event: health``) and keep it in history."""
        if isinstance(frame, dict):
            frame = HealthFrame.from_dict(frame)
        data = dumps(frame)
        with self._lock:
            self._health.append((frame.step, data))
        self._broadcast(_sse("health", data, frame.step))

    def watch(self, model: Any, optimizer: Any = None, every: int = 10, **kwargs: Any) -> Any:
        """``ptoa.watch`` streaming into this server."""
        from .health import Watch

        kwargs.setdefault("open", False)
        return Watch(model, optimizer, every, server=self, **kwargs)

    def watch_trace(self, model: Any, example_input: Any, every: int = 10) -> Any:
        """Push a full TraceFrame (activations) every N training steps."""
        from .torch import TraceWatch

        return TraceWatch(model, example_input, self.push, every=every, spec=self._spec)

    def health_since(self, since: int) -> List[str]:
        with self._lock:
            return [d for s, d in self._health if s > since]

    def _broadcast(self, msg: bytes) -> None:
        with self._lock:
            subs = list(self._subs)
        for s in subs:
            s.put(msg)

    def _subscribe(self) -> _Subscriber:
        s = _Subscriber()
        with self._lock:
            self._subs.append(s)
        return s

    def _unsubscribe(self, s: _Subscriber) -> None:
        with self._lock:
            if s in self._subs:
                self._subs.remove(s)


_DATA_PATHS = {"/spec", "/trace", "/events", "/health"}


class _Handler(BaseHTTPRequestHandler):
    bridge: LocalServer
    server_version = "pta-bridge"
    sys_version = ""

    def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
        if os.environ.get("PTA_DEBUG"):
            super().log_message(fmt, *args)

    # security checks ----------------------------------------------------------

    def _allowed_hosts(self) -> set:
        port = self.bridge.port
        return {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    def _allowed_origins(self) -> set:
        return {"http://" + h for h in self._allowed_hosts()} | self.bridge.allowed_origins

    def _cors(self) -> Dict[str, str]:
        origin = (self.headers.get("Origin") or "").lower()
        if origin and origin in self.bridge.allowed_origins:
            h = {"Access-Control-Allow-Origin": origin, "Vary": "Origin"}
            if self.headers.get("Access-Control-Request-Private-Network"):
                h["Access-Control-Allow-Private-Network"] = "true"
            return h
        return {}

    def _check_common(self) -> bool:
        host = (self.headers.get("Host") or "").lower()
        if host not in self._allowed_hosts():
            self._send(421, "text/plain", b"wrong host\n")
            return False
        origin = self.headers.get("Origin")
        if origin is not None and origin.lower() not in self._allowed_origins():
            self._send(403, "text/plain", b"cross-origin request refused\n")
            return False
        return True

    def _check_token(self, query: Dict[str, List[str]]) -> bool:
        site = self.headers.get("Sec-Fetch-Site")
        if site is not None and site not in ("same-origin", "none") and self.headers.get("Origin") is None:
            self._send(403, "text/plain", b"cross-site request refused\n")
            return False
        auth = self.headers.get("Authorization") or ""
        bearer = auth[7:] if auth.lower().startswith("bearer ") else ""
        given = (query.get("token") or [self.headers.get("X-PTA-Token") or bearer])[0]
        if not given:
            self._send(401, "text/plain", b"missing token\n")
            return False
        if not hmac.compare_digest(given.encode(), self.bridge.token.encode()):
            self._send(403, "text/plain", b"bad token\n")
            return False
        return True

    # responses -----------------------------------------------------------------

    def _headers(self, status: int, ctype: str, length: Optional[int] = None, extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        merged = dict(self._cors())
        merged.update(extra or {})
        for k, v in merged.items():
            self.send_header(k, v)
        self.end_headers()

    def _send(self, status: int, ctype: str, body: bytes) -> None:
        self._headers(status, ctype, len(body))
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, body: str) -> None:
        self._send(200, "application/json; charset=utf-8", body.encode("utf-8"))

    # methods -------------------------------------------------------------------

    def do_OPTIONS(self) -> None:  # CORS preflight, only for the allowed Lab origins
        if not self._check_common():
            return
        cors = self._cors()
        if not cors:
            self._send(403, "text/plain", b"cross-origin request refused\n")
            return
        cors.update({"Access-Control-Allow-Methods": "GET, HEAD, OPTIONS", "Access-Control-Allow-Headers": "Authorization, X-PTA-Token", "Access-Control-Max-Age": "600"})
        self._headers(204, "text/plain", 0, cors)

    def do_POST(self) -> None:
        self._send(405, "text/plain", b"method not allowed\n")

    do_PUT = do_DELETE = do_PATCH = do_POST

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._check_common():
            return
        parts = urlsplit(self.path)
        path = parts.path
        query = parse_qs(parts.query)
        if path in _DATA_PATHS:
            if not self._check_token(query):
                return
            if path == "/spec":
                hello = self.bridge.hello()
                if hello is None:
                    self._send(204, "text/plain", b"")
                else:
                    self._json(dumps(hello))
            elif path == "/trace":
                data = self.bridge._trace_json
                if data is None:
                    self._send(204, "text/plain", b"")
                else:
                    self._json(data)
            elif path == "/health":
                try:
                    since = int((query.get("since") or ["-1"])[0])
                except ValueError:
                    self._send(400, "text/plain", b"since must be an integer step\n")
                    return
                self._json('{"frames":[' + ",".join(self.bridge.health_since(since)) + "]}")
            else:
                self._events()
            return
        if path in ("/", "/index.html"):
            self._send(200, "text/plain; charset=utf-8", b"papertoanything bridge. Open the Lab with the link that ptoa printed.\n")
        else:
            self._send(404, "text/plain", b"not found\n")

    def _events(self) -> None:
        b = self.bridge
        sub = b._subscribe()
        try:
            self._headers(200, "text/event-stream; charset=utf-8", extra={"Connection": "keep-alive", "X-Accel-Buffering": "no"})
            if self.command == "HEAD":
                return
            self.wfile.write(b"retry: 2000\n\n")
            hello = b.hello()
            if hello is not None:
                self.wfile.write(_sse("hello", hello))
            if b._trace_json is not None and b._trace is not None:
                self.wfile.write(_sse("frame", b._trace_json, b._trace.step))
            self.wfile.flush()
            while True:
                try:
                    msg = sub.q.get(timeout=KEEPALIVE_SECONDS)
                except queue.Empty:
                    if b._closed.is_set():
                        break
                    msg = b": keepalive\n\n"
                if msg is None:
                    break
                self.wfile.write(msg)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            pass
        finally:
            b._unsubscribe(sub)
            self.close_connection = True
