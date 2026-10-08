import http.client
import json
import os
import time
import unittest
import urllib.error
import urllib.request

from papertoanything.server import LocalServer
from papertoanything.spec import FrameTensor, HealthFrame, ModelSpec, TraceFrame, encode_f32

SPEC = ModelSpec.from_dict(
    {"v": 1, "name": "s", "blocks": [{"id": "t0", "kind": "tokens", "params": {}, "inputs": []}], "output": "t0", "origin": {"kind": "lab"}}
)


def get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def raw_get(port, path, headers=None, method="GET", skip_host=False):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.putrequest(method, path, skip_host=skip_host)
    for k, v in (headers or {}).items():
        conn.putheader(k, v)
    conn.endheaders()
    r = conn.getresponse()
    body = r.read()
    conn.close()
    return r, body


class SSE:
    """Minimal SSE reader over http.client."""

    def __init__(self, server, path):
        self.conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=5)
        self.conn.request("GET", path)
        self.resp = self.conn.getresponse()

    def next_event(self, name):
        event, data = None, []
        deadline = time.time() + 5
        while time.time() < deadline:
            line = self.resp.fp.readline().decode("utf-8").rstrip("\n")
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                data.append(line[6:])
            elif line == "":
                if event == name:
                    return json.loads("\n".join(data))
                event, data = None, []
        raise AssertionError(f"no {name} event")

    def close(self):
        self.conn.close()


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.s = LocalServer(spec=SPEC).start()
        self.base = f"http://127.0.0.1:{self.s.port}"
        self.tok = "token=" + self.s.token

    def tearDown(self):
        self.s.close()

    def test_url_shape(self):
        self.assertEqual(self.s.url, f"https://lab.papertoanything.com/?bridge={self.base}&token={self.s.token}")
        self.assertGreaterEqual(len(self.s.token), 24)
        self.assertNotEqual(LocalServer().token, self.s.token)

    def test_spec_requires_token(self):
        self.assertEqual(get(self.base + "/spec")[0], 401)
        self.assertEqual(get(self.base + "/spec?token=nope")[0], 403)
        status, headers, body = get(self.base + "/spec?" + self.tok)
        self.assertEqual(status, 200)
        hello = json.loads(body)
        self.assertEqual(hello["protocol"], "pta-bridge")
        self.assertEqual(hello["spec"]["name"], "s")
        self.assertIsNone(headers.get("Access-Control-Allow-Origin"))
        self.assertEqual(headers.get("Cache-Control"), "no-store")
        self.assertEqual(headers.get("Referrer-Policy"), "no-referrer")
        self.assertEqual(get(self.base + "/spec", {"X-PTA-Token": self.s.token})[0], 200)

    def test_trace_events_health_require_token(self):
        for path in ("/trace", "/events", "/health"):
            self.assertEqual(get(self.base + path)[0], 401, path)
            self.assertEqual(get(self.base + path + "?token=x")[0], 403, path)

    def test_cross_origin_refused(self):
        self.assertEqual(get(self.base + "/spec?" + self.tok, {"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(get(self.base + "/spec?" + self.tok, {"Origin": self.base})[0], 200)
        self.assertEqual(get(self.base + "/spec?" + self.tok, {"Sec-Fetch-Site": "cross-site"})[0], 403)
        self.assertEqual(get(self.base + "/spec?" + self.tok, {"Sec-Fetch-Site": "same-origin"})[0], 200)

    def test_lab_origin_gets_exact_cors(self):
        for origin in ("https://lab.papertoanything.com", "http://localhost:5180"):
            status, headers, _ = get(self.base + "/spec?" + self.tok, {"Origin": origin, "Sec-Fetch-Site": "cross-site"})
            self.assertEqual(status, 200, origin)
            self.assertEqual(headers.get("Access-Control-Allow-Origin"), origin)
            self.assertEqual(headers.get("Vary"), "Origin")
        for origin in ("https://lab.papertoanything.com.evil.example", "http://localhost:5181", "null"):
            self.assertEqual(get(self.base + "/spec?" + self.tok, {"Origin": origin})[0], 403, origin)
        # the token is still required, even from the Lab
        self.assertEqual(get(self.base + "/spec", {"Origin": "https://lab.papertoanything.com"})[0], 401)
        # bearer header works
        self.assertEqual(get(self.base + "/spec", {"Authorization": "Bearer " + self.s.token})[0], 200)

    def test_preflight(self):
        r, _ = raw_get(self.s.port, "/events", {"Origin": "https://lab.papertoanything.com", "Access-Control-Request-Method": "GET", "Access-Control-Request-Headers": "authorization", "Access-Control-Request-Private-Network": "true"}, method="OPTIONS")
        self.assertEqual(r.status, 204)
        self.assertEqual(r.getheader("Access-Control-Allow-Origin"), "https://lab.papertoanything.com")
        self.assertEqual(r.getheader("Access-Control-Allow-Private-Network"), "true")
        self.assertNotIn("*", r.getheader("Access-Control-Allow-Origin"))

    def test_custom_lab_url_is_allowed(self):
        with LocalServer(spec=SPEC, lab_url="http://localhost:9999/") as s:
            self.assertTrue(s.url.startswith("http://localhost:9999/?bridge=http://127.0.0.1:"))
            base = f"http://127.0.0.1:{s.port}"
            self.assertEqual(get(base + "/spec?token=" + s.token, {"Origin": "http://localhost:9999"})[0], 200)

    def test_dns_rebinding_host_refused(self):
        r, _ = raw_get(self.s.port, "/spec?" + self.tok, {"Host": f"evil.example:{self.s.port}"}, skip_host=True)
        self.assertEqual(r.status, 421)

    def test_no_cors_for_unknown_origin_preflight(self):
        r, _ = raw_get(self.s.port, "/spec", {"Origin": "https://evil.example"}, method="OPTIONS")
        self.assertEqual(r.status, 403)
        self.assertIsNone(r.getheader("Access-Control-Allow-Origin"))

    def test_loopback_only(self):
        with self.assertRaises(ValueError):
            LocalServer(host="0.0.0.0")
        self.assertEqual(self.s._httpd.server_address[0], "127.0.0.1")

    def test_trace_push(self):
        self.assertEqual(get(self.base + "/trace?" + self.tok)[0], 204)
        self.s.push(TraceFrame(step=7, tensors=[]))
        status, _, body = get(self.base + "/trace?" + self.tok)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["step"], 7)

    def test_events_stream(self):
        sse = SSE(self.s, "/events?" + self.tok)
        try:
            self.assertEqual(sse.resp.status, 200)
            self.assertTrue(sse.resp.getheader("Content-Type").startswith("text/event-stream"))
            hello = sse.next_event("hello")
            self.assertEqual(hello["spec"]["name"], "s")
            self.s.push(TraceFrame(step=1, loss=0.5, tensors=[FrameTensor("t0:out", [2], encode_f32([1, 2]))]))
            frame = sse.next_event("frame")
            self.assertEqual(frame["step"], 1)
            self.assertEqual(frame["tensors"][0]["shape"], [2])
            self.s.push_health(HealthFrame(step=10, t=1.0, loss=2.0))
            self.assertEqual(sse.next_event("health")["step"], 10)
            self.s.set_spec(SPEC)
            self.assertEqual(sse.next_event("spec")["protocol"], "pta-bridge")
        finally:
            sse.close()

    def test_health_history(self):
        for step in (0, 10, 20):
            self.s.push_health(HealthFrame(step=step, t=step, loss=1.0))
        frames = json.loads(get(self.base + "/health?" + self.tok)[2])["frames"]
        self.assertEqual([f["step"] for f in frames], [0, 10, 20])
        frames = json.loads(get(self.base + "/health?since=5&" + self.tok)[2])["frames"]
        self.assertEqual([f["step"] for f in frames], [10, 20])
        self.assertEqual(get(self.base + "/health?since=x&" + self.tok)[0], 400)

    def test_serves_no_web_page(self):
        status, _, body = get(self.base + "/")
        self.assertEqual(status, 200)
        self.assertNotIn(b"<html", body.lower())
        self.assertEqual(get(self.base + "/nope.js")[0], 404)
        self.assertEqual(get(self.base + "/assets/app.js")[0], 404)

    def test_health_ring_buffer(self):
        from papertoanything import server as srv

        old = srv.HEALTH_HISTORY
        srv.HEALTH_HISTORY = 4
        try:
            with LocalServer(spec=SPEC) as s:
                for step in range(10):
                    s.push_health(HealthFrame(step=step, t=step, loss=1.0))
                base = f"http://127.0.0.1:{s.port}"
                frames = json.loads(get(base + "/health?since=-1&token=" + s.token)[2])["frames"]
                self.assertEqual([f["step"] for f in frames], [6, 7, 8, 9])
        finally:
            srv.HEALTH_HISTORY = old

    def test_events_readable_from_lab_origin(self):
        conn = http.client.HTTPConnection("127.0.0.1", self.s.port, timeout=5)
        conn.request("GET", "/events?" + self.tok, headers={"Origin": "https://lab.papertoanything.com", "Sec-Fetch-Site": "cross-site"})
        r = conn.getresponse()
        self.assertEqual(r.status, 200)
        self.assertEqual(r.getheader("Access-Control-Allow-Origin"), "https://lab.papertoanything.com")
        conn.close()

    def test_close_ends_stream(self):
        sse = SSE(self.s, "/events?" + self.tok)
        sse.next_event("hello")
        self.s.close()
        sse.next_event("bye")
        sse.close()


class PackagingTests(unittest.TestCase):
    def test_no_lab_bundled(self):
        import importlib.util
        import papertoanything

        root = os.path.dirname(papertoanything.__file__)
        self.assertFalse(os.path.exists(os.path.join(root, "_lab")))
        self.assertIsNone(importlib.util.find_spec("papertoanything._viewer"))


if __name__ == "__main__":
    unittest.main()
