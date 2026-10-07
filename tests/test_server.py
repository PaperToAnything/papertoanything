import http.client
import json
import os
import tempfile
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
        self.s = LocalServer(spec=SPEC, lab_dir=os.devnull).start()
        self.base = f"http://127.0.0.1:{self.s.port}"
        self.tok = "token=" + self.s.token

    def tearDown(self):
        self.s.close()

    def test_url_shape(self):
        self.assertEqual(self.s.url, f"{self.base}/?bridge=local&token={self.s.token}")
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

    def test_dns_rebinding_host_refused(self):
        r, _ = raw_get(self.s.port, "/spec?" + self.tok, {"Host": f"evil.example:{self.s.port}"}, skip_host=True)
        self.assertEqual(r.status, 421)

    def test_no_cors_preflight(self):
        r, _ = raw_get(self.s.port, "/spec", {"Origin": "https://evil.example"}, method="OPTIONS")
        self.assertIn(r.status, (403, 405))
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

    def test_builtin_viewer_without_lab(self):
        status, headers, body = get(self.base + "/?bridge=local&" + self.tok)
        self.assertEqual(status, 200)
        self.assertIn(b"EventSource", body)
        self.assertIn("default-src 'self'", headers.get("Content-Security-Policy"))
        self.assertEqual(get(self.base + "/nope.js")[0], 404)

    def test_close_ends_stream(self):
        sse = SSE(self.s, "/events?" + self.tok)
        sse.next_event("hello")
        self.s.close()
        sse.next_event("bye")
        sse.close()


class LabDirTests(unittest.TestCase):
    def test_serves_lab_build_and_blocks_traversal(self):
        with tempfile.TemporaryDirectory() as d:
            lab = os.path.join(d, "lab")
            os.makedirs(os.path.join(lab, "assets"))
            with open(os.path.join(lab, "index.html"), "w") as fh:
                fh.write("<!doctype html><title>lab</title>")
            with open(os.path.join(lab, "assets", "app.js"), "w") as fh:
                fh.write("console.log(1)")
            with open(os.path.join(d, "secret.txt"), "w") as fh:
                fh.write("secret")
            with LocalServer(spec=SPEC, lab_dir=lab) as s:
                base = f"http://127.0.0.1:{s.port}"
                status, _, body = get(base + "/?bridge=local&token=" + s.token)
                self.assertEqual((status, body), (200, b"<!doctype html><title>lab</title>"))
                status, headers, _ = get(base + "/assets/app.js")
                self.assertEqual(status, 200)
                self.assertTrue(headers.get("Content-Type").startswith("text/javascript"))
                self.assertEqual(get(base + "/some/route")[2], b"<!doctype html><title>lab</title>")
                for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt"):
                    _, body = raw_get(s.port, path)
                    self.assertNotEqual(body, b"secret", path)

    def test_env_lab_dir(self):
        from papertoanything.server import find_lab_dir

        with tempfile.TemporaryDirectory() as d:
            open(os.path.join(d, "index.html"), "w").close()
            os.environ["PTA_LAB_DIR"] = d
            try:
                self.assertEqual(os.path.realpath(str(find_lab_dir())), os.path.realpath(d))
            finally:
                del os.environ["PTA_LAB_DIR"]

    def test_bundled_placeholder_is_not_a_lab(self):
        from papertoanything.server import bundled_lab_dir, find_lab_dir

        self.assertTrue((bundled_lab_dir() / "README.md").is_file())
        os.environ.pop("PTA_LAB_DIR", None)
        self.assertIsNone(find_lab_dir())


if __name__ == "__main__":
    unittest.main()
