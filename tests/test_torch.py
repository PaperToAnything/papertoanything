import contextlib
import io
import json
import os
import tempfile
import unittest
import warnings

try:
    import torch
    import torch.nn as nn

    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False

try:
    import transformers  # noqa: F401

    HAVE_HF = HAVE_TORCH
except Exception:
    HAVE_HF = False


def kinds(spec):
    return [(b.kind, b.label) for b in spec.blocks]


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class FromModuleTests(unittest.TestCase):
    def setUp(self):
        from tinygpt import TinyGPT

        torch.manual_seed(0)
        self.TinyGPT = TinyGPT
        self.x = torch.randint(0, 16, (2, 8))

    def analyze(self, *a, **k):
        from papertoanything.torch import analyze

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return analyze(*a, **k)

    def test_nanogpt_structure(self):
        an = self.analyze(self.TinyGPT(), self.x)
        self.assertEqual(an.strategy, "fx")
        spec = an.spec
        self.assertEqual(spec.origin.kind, "torch")
        self.assertEqual(spec.origin.source, "tinygpt:TinyGPT")
        by = {b.id: b for b in spec.blocks}
        k = [b.kind for b in spec.blocks if b.kind != "group"]
        self.assertEqual(k[:3], ["tokens", "embedding", "posenc"])
        self.assertEqual(k.count("attention"), 2)
        self.assertEqual(k.count("mlp"), 2)
        self.assertEqual(k.count("add"), 4)
        self.assertEqual(k.count("norm"), 5)
        self.assertEqual(k[-1], "unembed")
        e0 = next(b for b in spec.blocks if b.kind == "embedding")
        self.assertEqual((e0.params.vocab, e0.params.d), (16, 16))
        p0 = next(b for b in spec.blocks if b.kind == "posenc")
        self.assertEqual((p0.params.variant, p0.params.maxLen, p0.inputs), ("learned", 8, [e0.id]))
        att = next(b for b in spec.blocks if b.kind == "attention")
        self.assertEqual((att.params.heads, att.params.d, att.params.causal), (2, 16, True))
        mlp = next(b for b in spec.blocks if b.kind == "mlp")
        self.assertEqual((mlp.params.hidden, mlp.params.variant), (64, "gelu"))
        u = by[spec.output]
        self.assertEqual((u.kind, u.params.tiedTo), ("unembed", e0.id))
        # Residual adds join the stream and the sublayer.
        r0 = next(b for b in spec.blocks if b.kind == "add")
        self.assertEqual(r0.inputs, [p0.id, att.id])
        # Blocks inside a transformer layer point at its group.
        self.assertEqual(by[att.group].kind, "group")
        self.assertIn("transformer.h.0", by[att.group].label)
        # Every input refers to an earlier block (topological order).
        seen = set()
        for b in spec.blocks:
            for i in b.inputs:
                self.assertIn(i, seen, b.id)
            seen.add(b.id)
        json.loads(spec.to_json())

    def test_fx_and_hooks_agree(self):
        a = self.analyze(self.TinyGPT(), self.x, strategy="fx").spec
        b = self.analyze(self.TinyGPT(), self.x, strategy="hooks").spec
        self.assertEqual(a.to_dict(), b.to_dict())
        self.assertEqual(a.bindings, b.bindings)

    def test_control_flow_falls_back_to_hooks(self):
        an = self.analyze(self.TinyGPT(strict=True), self.x)
        self.assertEqual(an.strategy, "hooks")
        self.assertTrue(any(n.startswith("fx:") for n in an.warnings))
        ref = self.analyze(self.TinyGPT(), self.x).spec
        self.assertEqual(kinds(an.spec), kinds(ref))

    def test_no_input_and_no_trace_uses_tree(self):
        from papertoanything import from_module

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = from_module(self.TinyGPT(strict=True))
        self.assertTrue(any("registration order" in str(x.message) for x in w))
        self.assertIn("attention", [b.kind for b in spec.blocks])

    def test_flash_attention_variant(self):
        spec = self.analyze(self.TinyGPT(flash=True), self.x).spec
        self.assertEqual([b.kind for b in spec.blocks].count("attention"), 2)

    def test_unknown_modules_become_groups_and_never_fail(self):
        class Weird(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.ones(4))

            def forward(self, x):
                return x * self.w

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv1d(4, 4, 1)
                self.weird = Weird()
                self.fc = nn.Linear(4, 2)
                self.act = nn.ReLU()

            def forward(self, x):
                h = self.conv(x.transpose(1, 2)).transpose(1, 2)
                h = self.weird(h)
                return torch.sigmoid(self.act(self.fc(h)))

        x = torch.randn(2, 3, 4)
        spec = self.analyze(Net(), x).spec
        ks = kinds(spec)
        self.assertIn(("input", "x"), ks)
        self.assertIn(("group", "Conv1d"), ks)
        self.assertIn(("group", "Weird"), ks)
        self.assertIn(("linear", "fc"), ks)
        self.assertIn(("activation", "act"), ks)
        self.assertEqual(spec.block(spec.output).kind, "activation")
        self.assertEqual(spec.block(spec.output).params.variant, "sigmoid")
        inp = next(b for b in spec.blocks if b.kind == "input")
        self.assertEqual(inp.params.in_, 4)

    def test_multihead_attention(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(10, 8)
                self.mha = nn.MultiheadAttention(8, 2, batch_first=True)
                self.norm = nn.LayerNorm(8)

            def forward(self, idx):
                x = self.emb(idx)
                a, _ = self.mha(x, x, x, need_weights=False)
                return self.norm(x + a)

        spec = self.analyze(M(), torch.randint(0, 10, (1, 5))).spec
        att = next(b for b in spec.blocks if b.kind == "attention")
        self.assertEqual((att.params.heads, att.params.d), (2, 8))
        self.assertIn("add", [b.kind for b in spec.blocks])

    def test_model_state_unchanged(self):
        m = self.TinyGPT()
        m.train()
        before = {k: v.clone() for k, v in m.state_dict().items()}
        self.analyze(m, self.x)
        self.assertTrue(m.training)
        for k, v in m.state_dict().items():
            self.assertTrue(torch.equal(v, before[k]), k)

    @unittest.skipUnless(HAVE_HF, "transformers not installed")
    def test_hf_gpt2(self):
        from transformers import GPT2Config, GPT2LMHeadModel

        cfg = GPT2Config(vocab_size=32, n_positions=16, n_embd=16, n_layer=2, n_head=2)
        model = GPT2LMHeadModel(cfg)
        spec = self.analyze(model, torch.randint(0, 32, (1, 8))).spec
        ks = [b.kind for b in spec.blocks]
        self.assertEqual(ks.count("attention"), 2, kinds(spec))
        att = next(b for b in spec.blocks if b.kind == "attention")
        self.assertEqual(att.params.heads, 2)
        self.assertEqual(spec.block(spec.output).kind, "unembed")


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class CaptureTests(unittest.TestCase):
    def setUp(self):
        from tinygpt import TinyGPT

        torch.manual_seed(0)
        self.model = TinyGPT()
        self.x = torch.randint(0, 16, (2, 8))

    def test_capture_keys_match_spec(self):
        from papertoanything import capture, from_module

        spec = from_module(self.model, self.x)
        frame = capture(self.model, self.x, spec=spec)
        keys = {t.key for t in frame.tensors}
        for b in spec.blocks:
            if b.kind != "group":
                self.assertIn(f"{b.id}:out", keys)
        logits = frame.tensor(f"{spec.output}:out")
        self.assertEqual(logits.shape, [2, 8, 16])
        self.assertFalse(logits.clipped)
        ref, _ = self.model.eval()(self.x)
        got = torch.tensor(logits.values()).view(2, 8, 16)
        self.assertTrue(torch.allclose(got, ref.float(), atol=1e-5))
        self.assertIsNone(frame.uncaptured)

    def test_capture_clips_honestly(self):
        from papertoanything import capture

        frame = capture(self.model, self.x, max_elems=20)
        clipped = [t for t in frame.tensors if t.clipped]
        self.assertTrue(clipped)
        for t in clipped:
            n = 1
            for s in t.dataShape:
                n *= s
            self.assertLessEqual(n, 20)
            self.assertEqual(len(t.values()), n)
            self.assertNotEqual(t.dataShape, t.shape)
        self.assertEqual(frame.clipped, [t.key for t in clipped])

    def test_capture_loss_and_frame_budget(self):
        from papertoanything import capture

        frame = capture(self.model, (self.x, self.x), max_frame_elems=300)
        self.assertIsNotNone(frame.loss)
        empty = [t for t in frame.tensors if t.b64 == ""]
        self.assertTrue(empty)
        self.assertTrue(all(t.clipped and set(t.dataShape) == {0} for t in empty))

    def test_trace_watch_and_save(self):
        from papertoanything import load, save
        from papertoanything.torch import TraceWatch

        frames = []
        opt = torch.optim.SGD(self.model.parameters(), lr=0.1)
        with TraceWatch(self.model, self.x, frames.append, every=2):
            for _ in range(4):
                _, loss = self.model(self.x, self.x)
                loss.backward()
                opt.step()
                opt.zero_grad()
        self.assertEqual([f.step for f in frames], [2, 4])
        self.assertIsNotNone(frames[-1].loss)
        with tempfile.TemporaryDirectory() as d:
            p = save(self.model, os.path.join(d, "m.pta"), example_input=self.x)
            spec, trace = load(p)
            self.assertEqual(spec.name, "TinyGPT")
            self.assertTrue(trace.tensors)


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class WatchTests(unittest.TestCase):
    def setUp(self):
        from tinygpt import TinyGPT

        torch.manual_seed(0)
        self.TinyGPT = TinyGPT
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def train(self, w, model, opt, steps, report=True):
        for _ in range(steps):
            x = torch.randint(0, 16, (4, 8))
            _, loss = model(x, x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if report:
                w.step(loss)

    def test_watch_with_optimizer(self):
        import papertoanything as pta
        from papertoanything.health import RunFile

        model = self.TinyGPT()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        rf = os.path.join(self.tmp.name, "run.pta")
        w = pta.watch(model, opt, every=5, open=False, run_file=rf, val_loss=lambda: 1.25, val_every=5)
        with w:
            self.train(w, model, opt, 12)
        self.assertEqual([f.step for f in w.frames], [0, 5, 10])
        f = w.frames[-1]
        self.assertIsNotNone(f.loss)
        self.assertEqual(f.lr, 1e-2)
        self.assertEqual(f.valLoss, 1.25)
        self.assertGreater(f.gradNorm, 0)
        self.assertTrue({"forward", "backward", "optim"} <= set(f.timing))
        ids = {b.id: b for b in f.blocks}
        spec = w.spec
        att = next(b.id for b in spec.blocks if b.kind == "attention")
        mlp = next(b.id for b in spec.blocks if b.kind == "mlp")
        self.assertEqual(len(ids[att].headEntropy), 2)
        self.assertEqual(len(ids[att].headPattern), 2)
        self.assertEqual(ids[att].headPattern[0].size, 8)
        self.assertIsNotNone(ids[att].act)
        self.assertIsNotNone(ids[att].grad)
        self.assertGreater(ids[att].updateRatio, 0)
        self.assertIsNotNone(ids[mlp].hidden)
        self.assertEqual(len(ids[att].act.hist), 32)
        for e in ids[att].headEntropy:
            self.assertGreaterEqual(e, 0)
            self.assertLessEqual(e, 2.0795)  # ln(8)
        spec_d, frames = RunFile.read(rf)
        self.assertEqual(spec_d["name"], "TinyGPT")
        self.assertEqual([x.step for x in frames], [0, 5, 10])

    def test_watch_without_optimizer(self):
        import papertoanything as pta

        model = self.TinyGPT()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with pta.watch(model, every=3, open=False, run_file=False) as w:
            self.train(w, model, opt, 7)
        self.assertEqual([f.step for f in w.frames], [0, 3, 6])
        self.assertIsNotNone(w.frames[0].loss)
        self.assertTrue(any(b.updateRatio for b in w.frames[-1].blocks))

    def test_hooks_idle_between_samples(self):
        import papertoanything as pta
        from papertoanything import health

        calls = []
        orig = health.tensor_stats
        health.tensor_stats = lambda *a, **k: calls.append(1) or orig(*a, **k)
        try:
            model = self.TinyGPT()
            opt = torch.optim.SGD(model.parameters(), lr=0.1)
            with pta.watch(model, opt, every=100, open=False, run_file=False) as w:
                self.train(w, model, opt, 1)
                n = len(calls)
                self.assertGreater(n, 0)
                self.train(w, model, opt, 20)
                self.assertEqual(len(calls), n)
        finally:
            health.tensor_stats = orig

    def test_dead_relu_visible(self):
        import papertoanything as pta

        model = self.TinyGPT(act=nn.ReLU())
        for blk in model.transformer.h:
            nn.init.constant_(blk.mlp.c_fc.bias, -50.0)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        with pta.watch(model, opt, every=1, open=False, run_file=False) as w:
            self.train(w, model, opt, 2)
        mlp = next(b.id for b in w.spec.blocks if b.kind == "mlp")
        self.assertEqual(w.frames[-1].block(mlp).hidden.zeroFrac, 1.0)

    def test_nan_warning(self):
        import papertoanything as pta

        model = self.TinyGPT()
        with torch.no_grad():
            model.transformer.h[0].mlp.c_proj.weight[0, 0] = float("nan")
        opt = torch.optim.SGD(model.parameters(), lr=0.01)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pta.watch(model, opt, every=1, open=False, run_file=False) as w:
                self.train(w, model, opt, 1)
        self.assertTrue(any("NaN" in str(c.message) for c in caught))
        mlp = next(b.id for b in w.spec.blocks if b.kind == "mlp")
        self.assertGreater(w.frames[0].block(mlp).act.nanCount, 0)
        json.loads(json.dumps(w.frames[0].to_dict(), allow_nan=False))

    def test_streams_to_server(self):
        import urllib.request

        import papertoanything as pta
        from papertoanything.server import LocalServer

        model = self.TinyGPT()
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        with LocalServer() as s:
            with contextlib.redirect_stdout(io.StringIO()):
                w = pta.watch(model, opt, every=2, open=False, server=s, run_file=False)
            self.train(w, model, opt, 5)
            w.close()
            with urllib.request.urlopen(f"http://127.0.0.1:{s.port}/health?since=-1&token={s.token}", timeout=5) as r:
                frames = json.loads(r.read())["frames"]
            self.assertEqual([f["step"] for f in frames], [0, 2, 4])
            with urllib.request.urlopen(f"http://127.0.0.1:{s.port}/spec?token={s.token}", timeout=5) as r:
                self.assertEqual(json.loads(r.read())["spec"]["name"], "TinyGPT")

    def test_integrations_import_guarded(self):
        if HAVE_HF:
            from papertoanything.integrations.hf import PTACallback

            self.assertTrue(callable(PTACallback))
        try:
            import lightning  # noqa: F401
        except ImportError:
            with self.assertRaises(ImportError):
                import papertoanything.integrations.lightning  # noqa: F401


@unittest.skipUnless(HAVE_TORCH, "torch not installed")
class CliTorchTests(unittest.TestCase):
    def test_inspect_file_module(self):
        from papertoanything.cli import main

        here = os.path.dirname(os.path.abspath(__file__))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = main(["inspect", os.path.join(here, "tinygpt.py") + ":TinyGPT", "--input-shape", "1,8", "--json"])
        self.assertEqual(code, 0)
        spec = json.loads(buf.getvalue())
        self.assertEqual(spec["origin"], {"kind": "torch", "source": "tinygpt.py:TinyGPT"})
        self.assertIn("attention", [b["kind"] for b in spec["blocks"]])


if __name__ == "__main__":
    unittest.main()
