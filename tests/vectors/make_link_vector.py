"""Regenerate link.json. Run from python/: `python tests/vectors/make_link_vector.py`.

The vector is the cross-language contract for spec <-> fragment links. Only
regenerate it when the format deliberately changes, and tell the TypeScript
side (packages/engine link tests read the same file).
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from papertoanything.link import deflate_raw, spec_to_payload, to_url, LAB_URL  # noqa: E402
from papertoanything.spec import dumps  # noqa: E402

SPEC = {
    "v": 1,
    "name": "tiny µ-attn — ü",
    "blocks": [
        {"id": "t0", "kind": "tokens", "params": {"vocab": 50}, "inputs": []},
        {"id": "e0", "kind": "embedding", "label": "embed", "params": {"vocab": 50, "d": 64}, "inputs": ["t0"]},
        {"id": "a1", "kind": "attention", "label": "L1 attention", "params": {"heads": 4, "causal": True}, "inputs": ["e0"]},
        {"id": "u0", "kind": "unembed", "params": {"vocab": 50, "d": 64, "tiedTo": "e0"}, "inputs": ["a1"], "at": {"x": 0, "y": 120.5}},
    ],
    "output": "u0",
    "train": {"optimizer": "adamw", "lr": 0.003, "batch": 32, "steps": 500, "seed": 42},
    "origin": {"kind": "lab"},
}


NODE_PAYLOAD = "hY9LTsNAEESvYtW6icb5sJgzsGSHsmh7Wsko9ozlaQdCZIlDcBV2rOAmnAQNSRQjIbFrlbpeVR2xhy0JgVuBhfpwKD7eblg1FF8vr8XnOwhVE-tdgn04wrv8ZUDY-fBzx52EBELHPbcJ9oh9rLmCXZmR4EM3aLauRzq7ZeKWthLnfNiA0HAlzUX7G0hwsLfLKTeXubK5vLJZVYL6GCbsu7KYyteIrbBLsEtCzUPiBlb7QX4FyTRomIwYwv-VCerF3cfT_l9cLrEmsGbbE6whHGDLuZmtxnFNiIN2g14itWcf8mfs1Lf-Wfq81HH7mFf2sGZmzIJQsdZb2MWckFS6lKsYQhJxsMv5SIi935xQ5xkNVxjHbw"


def main():
    text = dumps(SPEC)
    raw = text.encode("utf-8")
    vec = {
        "about": (
            "payload = base64url_nopad(deflate_raw(utf8(json))). 'json' is the exact string to compress "
            "(JSON.stringify(spec) gives it). Decoders MUST turn 'fragment' back into 'spec' (deep-equal). "
            "Encoders SHOULD produce 'fragment'; a different DEFLATE implementation may legitimately produce "
            "different bytes, which is fine as long as the other side decodes them to the same spec."
        ),
        "spec": SPEC,
        "json": text,
        "utf8Length": len(raw),
        "deflateRawHex": deflate_raw(raw).hex(),
        "payload": spec_to_payload(SPEC),
        "fragment": "#s=" + spec_to_payload(SPEC),
        "url": to_url(SPEC, base=LAB_URL),
        "zlib": "level 6, wbits -15, memLevel 8, default strategy",
        # Produced by Node 24 CompressionStream("deflate-raw") from the same 'json'.
        # Different bytes, same spec: every decoder must accept both.
        "payloadNode": NODE_PAYLOAD,
    }
    with open(os.path.join(HERE, "link.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(vec, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    print(vec["url"])


if __name__ == "__main__":
    main()
