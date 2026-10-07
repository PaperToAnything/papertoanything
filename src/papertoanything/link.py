"""Spec <-> URL fragment, matching ``packages/engine/src/link.ts``.

Byte format (normative; the TypeScript engine must agree)::

    url      = LAB_URL + "#s=" + payload
    payload  = base64url( deflate_raw( utf8( json ) ) )

    json         Compact JSON of the ModelSpec: no whitespace, separators
                 "," and ":", non-ASCII characters written literally (not
                 \\u-escaped), keys in the order they appear in the object.
                 This is what ``JSON.stringify(spec)`` produces.
    utf8         UTF-8 bytes of that string, no BOM.
    deflate_raw  Raw DEFLATE stream (RFC 1951): no zlib header, no adler32,
                 no gzip wrapper. ``CompressionStream("deflate-raw")`` in the
                 browser; ``zlib.compressobj(level=6, wbits=-15)`` here.
    base64url    RFC 4648 section 5 alphabet (``-`` and ``_``), padding
                 ``=`` stripped. Decoders must accept input with or without
                 padding.

Decoding is the reverse; any extra ``&key=value`` pairs after the payload in
the fragment are ignored.

Compatibility is defined by *decoding*, not by identical compressed bytes:
two DEFLATE encoders (stock zlib here, Chromium's zlib fork in the browser)
may emit different, equally valid streams for the same input. Each side must
decode the other's output to the same spec. ``tests/vectors/link.json``
carries a spec, its exact JSON string, the deflate bytes in hex and the
fragment Python produces, so the TypeScript side can check both directions.

The fragment (everything after ``#``) is never sent to any server by a
browser, so a spec in a link stays between the terminal and the tab.
"""

from __future__ import annotations

import base64
import json
import os
import zlib
from typing import Any, Mapping, Union

from .spec import ModelSpec, dumps

__all__ = ["LAB_URL", "lab_url", "encode", "decode", "to_url", "from_url", "spec_to_payload", "payload_to_spec"]

LAB_URL = "https://lab.papertoanything.com/"
FRAGMENT_KEY = "s"
#: Refuse to inflate a payload beyond this many bytes (zip-bomb guard).
MAX_JSON_BYTES = 32 * 1024 * 1024

SpecLike = Union[ModelSpec, Mapping[str, Any]]


def lab_url() -> str:
    """The Lab origin links point at. Override with ``PTA_LAB_URL``."""
    return os.environ.get("PTA_LAB_URL", LAB_URL)


def _json_bytes(spec: SpecLike) -> bytes:
    obj = spec.to_dict() if isinstance(spec, ModelSpec) else dict(spec)
    return dumps(obj).encode("utf-8")


def deflate_raw(data: bytes) -> bytes:
    c = zlib.compressobj(level=6, method=zlib.DEFLATED, wbits=-15, memLevel=8, strategy=zlib.Z_DEFAULT_STRATEGY)
    return c.compress(data) + c.flush()


def inflate_raw(data: bytes, limit: int = MAX_JSON_BYTES) -> bytes:
    d = zlib.decompressobj(wbits=-15)
    out = d.decompress(data, limit)
    if d.unconsumed_tail:
        raise ValueError(f"payload inflates to more than {limit} bytes")
    out += d.flush()
    if not d.eof:
        raise ValueError("truncated deflate stream")
    return out


def b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64url_decode(text: str) -> bytes:
    text = text.strip()
    if any(c in text for c in "+/"):
        raise ValueError("payload uses the standard base64 alphabet; expected base64url")
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def spec_to_payload(spec: SpecLike) -> str:
    """The fragment payload: base64url(deflate_raw(utf8(json)))."""
    return b64url_encode(deflate_raw(_json_bytes(spec)))


encode = spec_to_payload


def payload_to_dict(payload: str) -> dict:
    return json.loads(inflate_raw(b64url_decode(payload)).decode("utf-8"))


def payload_to_spec(payload: str) -> ModelSpec:
    return ModelSpec.from_dict(payload_to_dict(payload))


def to_url(spec: SpecLike, base: str | None = None) -> str:
    """``https://lab.papertoanything.com/#s=<payload>``."""
    return (base or lab_url()) + "#" + FRAGMENT_KEY + "=" + spec_to_payload(spec)


def extract_payload(url_or_fragment: str) -> str:
    """Accept a full URL, ``#s=...``, ``s=...`` or a bare payload."""
    text = url_or_fragment.strip()
    if "#" in text:
        text = text.split("#", 1)[1]
    for part in text.split("&"):
        if part.startswith(FRAGMENT_KEY + "="):
            return part[len(FRAGMENT_KEY) + 1 :]
    if "=" in text.rstrip("="):
        raise ValueError("no s= payload in the fragment")
    return text


def from_url(url_or_fragment: str) -> ModelSpec:
    return payload_to_spec(extract_payload(url_or_fragment))


def decode(url_or_fragment: str) -> ModelSpec:
    return from_url(url_or_fragment)
