import json
import unittest

from papertoanything.spec import (
    BlockHealth,
    BridgeHello,
    Diagnosis,
    HealthFrame,
    ModelSpec,
    TensorStats,
    TraceFrame,
    decode_f32,
    encode_f32,
)

SPEC = {
    "v": 1,
    "name": "two-layer",
    "blocks": [
        {"id": "t0", "kind": "tokens", "params": {"vocab": 50}, "inputs": []},
        {"id": "e0", "kind": "embedding", "label": "embed", "params": {"vocab": 50, "d": 64}, "inputs": ["t0"]},
        {"id": "g1", "kind": "group", "label": "layer 1", "params": {}, "inputs": []},
        {"id": "l0", "kind": "linear", "params": {"in": 64, "out": 64, "bias": False, "futureKey": [1, 2]}, "inputs": ["e0"], "group": "g1", "at": {"x": 1.5, "y": 2}},
        {"id": "u0", "kind": "unembed", "params": {"vocab": 50, "d": 64, "tiedTo": "e0"}, "inputs": ["l0"]},
    ],
    "output": "u0",
    "dataset": {"id": "repeat-tokens", "provenance": "synthetic", "seed": 1, "options": {"vocab": 50, "len": 32}},
    "train": {"optimizer": "adamw", "lr": 0.003, "weightDecay": 0.1, "batch": 32, "steps": 100, "seed": 0},
    "origin": {"kind": "torch", "source": "model:GPT"},
}


class SpecTests(unittest.TestCase):
    def test_round_trip_dict(self):
        spec = ModelSpec.from_dict(SPEC)
        self.assertEqual(spec.to_dict(), SPEC)
        self.assertEqual(ModelSpec.from_json(spec.to_json()), spec)

    def test_in_keyword_and_unknown_params(self):
        spec = ModelSpec.from_dict(SPEC)
        p = spec.block("l0").params
        self.assertEqual(p.in_, 64)
        self.assertEqual(p.extra, {"futureKey": [1, 2]})
        self.assertIn('"in":64', spec.to_json())

    def test_compact_json_matches_js(self):
        text = ModelSpec.from_dict(SPEC).to_json()
        self.assertNotIn(" ", text.replace("layer 1", ""))
        self.assertEqual(json.loads(text), SPEC)
        # Key order follows the TypeScript interfaces.
        self.assertTrue(text.startswith('{"v":1,"name":"two-layer","blocks":[{"id":"t0","kind":"tokens","params"'))

    def test_optional_fields_omitted(self):
        spec = ModelSpec.from_dict({"v": 1, "name": "x", "blocks": [], "output": "", "origin": {"kind": "lab"}})
        self.assertEqual(spec.to_dict(), {"v": 1, "name": "x", "blocks": [], "output": "", "origin": {"kind": "lab"}})

    def test_bad_version(self):
        with self.assertRaises(ValueError):
            ModelSpec.from_dict(dict(SPEC, v=2))

    def test_nan_rejected(self):
        spec = ModelSpec.from_dict(SPEC)
        spec.train.lr = float("nan")
        with self.assertRaises(ValueError):
            spec.to_json()

    def test_hello_and_frames(self):
        hello = BridgeHello(ModelSpec.from_dict(SPEC))
        d = hello.to_dict()
        self.assertEqual(d["protocol"], "pta-bridge")
        self.assertEqual(d["v"], 1)
        self.assertEqual(d["producer"]["name"], "papertoanything")
        self.assertEqual(BridgeHello.from_dict(d).spec, hello.spec)
        frame = TraceFrame.from_dict({"step": 3, "loss": 1.5, "tensors": [{"key": "e0:out", "shape": [2], "b64": encode_f32([1.0, -2.5])}]})
        self.assertEqual(frame.tensor("e0:out").values(), [1.0, -2.5])
        self.assertEqual(TraceFrame.from_dict(frame.to_dict()), frame)
        self.assertEqual(frame.clipped, [])

    def test_f32_little_endian(self):
        import base64
        import struct

        self.assertEqual(base64.b64decode(encode_f32([1.0])), struct.pack("<f", 1.0))
        self.assertEqual(decode_f32(encode_f32([0.5, 3.0])), [0.5, 3.0])

    def test_health_round_trip(self):
        st = TensorStats(0.1, 1.0, 3.0, 0.25, 0, 0, satFrac=0.5, hist=[1.0] * 32)
        fr = HealthFrame(step=10, t=12.5, loss=None, blocks=[BlockHealth("a0", act=st, headEntropy=[0.5, 1.0], updateRatio=1e-3)], lr=1e-3, timing={"forward": 2.0})
        d = fr.to_dict()
        self.assertIn("loss", d)
        self.assertIsNone(d["loss"])
        self.assertNotIn("valLoss", d)
        self.assertEqual(HealthFrame.from_dict(json.loads(json.dumps(d))), fr)
        dg = Diagnosis("dead-relu", "warn", ["m0"], 40, "MLP units are dead", {"zeroFrac": 0.93})
        self.assertEqual(Diagnosis.from_dict(dg.to_dict()), dg)


if __name__ == "__main__":
    unittest.main()
