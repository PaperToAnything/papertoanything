import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest

from papertoanything import __version__, load, show
from papertoanything.cli import main
from papertoanything.frames import clip_shape, frame_tensor_from_values
from papertoanything.link import from_url
from papertoanything.spec import decode_f32

SPEC = {"v": 1, "name": "s", "blocks": [{"id": "t0", "kind": "tokens", "params": {"vocab": 4}, "inputs": []}], "output": "t0", "origin": {"kind": "lab"}}
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


class FrameTests(unittest.TestCase):
    def test_clip_shape(self):
        self.assertEqual(clip_shape([2, 3], 100), [2, 3])
        self.assertEqual(clip_shape([4, 8, 16], 64), [1, 4, 16])
        self.assertEqual(clip_shape([4, 1000], 64), [1, 64])
        self.assertEqual(clip_shape([10], 3), [3])

    def test_clipped_values_are_real_corner(self):
        flat = list(range(24))  # shape [2, 3, 4]
        ft = frame_tensor_from_values("x", [2, 3, 4], flat, max_elems=8)
        self.assertTrue(ft.clipped)
        self.assertEqual(ft.shape, [2, 3, 4])
        self.assertEqual(ft.dataShape, [1, 2, 4])
        self.assertEqual(decode_f32(ft.b64), [0, 1, 2, 3, 4, 5, 6, 7])
        full = frame_tensor_from_values("x", [2, 3, 4], flat, max_elems=24)
        self.assertFalse(full.clipped)
        self.assertIsNone(full.dataShape)
        self.assertIn('"clipped":false', json.dumps(full.to_dict(), separators=(",", ":")))


class ShowTests(unittest.TestCase):
    def test_link_mode(self):
        with contextlib.redirect_stdout(io.StringIO()):
            url = show(SPEC, mode="link", open_browser=False)
        self.assertEqual(from_url(url).to_dict(), SPEC)

    def test_local_mode(self):
        with contextlib.redirect_stdout(io.StringIO()):
            s = show(SPEC, mode="local", open_browser=False)
        try:
            self.assertIn("?bridge=local&token=", s.url)
            self.assertEqual(s.spec.to_dict(), SPEC)
        finally:
            s.close()

    def test_file_mode_and_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.pta")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(show(SPEC, mode="file", path=path), path)
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
            self.assertEqual(doc["protocol"], "pta-file")
            self.assertEqual(doc["spec"], SPEC)
            spec, trace = load(path)
            self.assertEqual(spec.to_dict(), SPEC)
            self.assertIsNone(trace)

    def test_auto_without_lab_is_link(self):
        os.environ.pop("PTA_LAB_DIR", None)
        with contextlib.redirect_stdout(io.StringIO()):
            out = show(SPEC, open_browser=False)
        self.assertTrue(isinstance(out, str) and "#s=" in out)


class CliTests(unittest.TestCase):
    def run_cli(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(list(argv))
        return code, buf.getvalue()

    def test_version(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), self.assertRaises(SystemExit):
            main(["--version"])
        self.assertIn(__version__, buf.getvalue())

    def test_link_decode_save_inspect(self):
        with tempfile.TemporaryDirectory() as d:
            sp = os.path.join(d, "spec.json")
            with open(sp, "w") as fh:
                json.dump(SPEC, fh)
            code, out = self.run_cli("link", sp, "--no-open")
            self.assertEqual(code, 0)
            url = out.strip()
            self.assertEqual(from_url(url).to_dict(), SPEC)
            code, out = self.run_cli("decode", url)
            self.assertEqual(json.loads(out), SPEC)
            self.run_cli("save", sp, "-o", os.path.join(d, "x.pta"))
            self.assertEqual(load(os.path.join(d, "x.pta"))[0].to_dict(), SPEC)
            code, out = self.run_cli("inspect", sp)
            self.assertIn("t0", out)

    def test_help(self):
        code, out = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("inspect", out)

    def test_module_entry_point(self):
        env = dict(os.environ, PYTHONPATH=SRC)
        r = subprocess.run([sys.executable, "-m", "papertoanything", "--version"], capture_output=True, text=True, env=env, cwd=tempfile.gettempdir())
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn(__version__, r.stdout)

    def test_import_without_torch_side_effects(self):
        code = "import sys, papertoanything; assert 'torch' not in sys.modules, 'torch imported eagerly'; print('ok')"
        env = dict(os.environ, PYTHONPATH=SRC)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=tempfile.gettempdir())
        self.assertEqual(r.returncode, 0, r.stderr)


if __name__ == "__main__":
    unittest.main()
