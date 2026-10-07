"""The model spec, mirrored from ``packages/engine/src/spec.ts``.

``spec.ts`` is the contract; this module is a faithful Python copy of the
parts that cross the bridge. JSON field names are identical to the
TypeScript ones (camelCase included). The only renamed attribute is
``BlockParams.in_``, because ``in`` is a Python keyword; it is still ``"in"``
in JSON.

Every dataclass has ``to_dict()`` / ``from_dict()``; ``ModelSpec`` also has
``to_json()`` / ``from_json()``. Round-tripping a spec through JSON gives an
equal spec. Optional fields that are ``None`` are omitted from JSON, exactly as
an absent optional property in TypeScript. Unknown keys in ``params`` are kept
in ``BlockParams.extra`` and written back out, so a newer spec survives a trip
through an older Python package.

Stdlib only.
"""

from __future__ import annotations

import base64
import json
import struct
import sys
from array import array
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

__all__ = [
    "SPEC_VERSION",
    "LIVE_BUDGET",
    "BLOCK_KINDS",
    "BlockParams",
    "Block",
    "DatasetRef",
    "TrainConfig",
    "Origin",
    "ModelSpec",
    "TensorRecord",
    "MatmulRecord",
    "Trace",
    "FrameTensor",
    "TraceFrame",
    "BridgeHello",
    "TensorStats",
    "HeadPattern",
    "BlockHealth",
    "HealthFrame",
    "Diagnosis",
    "SEVERITIES",
    "encode_f32",
    "decode_f32",
    "dumps",
]

SPEC_VERSION = 1

LIVE_BUDGET = {"params": 2_000_000, "seqLen": 256, "vocab": 4096}

BLOCK_KINDS = (
    "tokens",
    "input",
    "embedding",
    "posenc",
    "attention",
    "mlp",
    "norm",
    "add",
    "linear",
    "activation",
    "unembed",
    "softmax",
    "loss",
    "group",
)

JSON = Dict[str, Any]


def dumps(obj: Any) -> str:
    """Compact JSON exactly as the bridge sends it.

    ``separators=(",", ":")`` and ``ensure_ascii=False`` match
    ``JSON.stringify`` byte for byte for the values a spec holds (strings,
    integers, booleans, finite floats). NaN and Infinity are rejected, since
    JSON has no spelling for them.
    """
    if hasattr(obj, "to_dict"):
        obj = obj.to_dict()
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _put(d: JSON, key: str, value: Any) -> None:
    if value is not None:
        d[key] = value


# ── Blocks ───────────────────────────────────────────────────────────────────

_PARAM_KEYS = (
    "vocab",
    "d",
    "variant",
    "maxLen",
    "heads",
    "causal",
    "attnOnly",
    "hidden",
    "in",
    "out",
    "bias",
    "tiedTo",
)


@dataclass
class BlockParams:
    vocab: Optional[int] = None
    d: Optional[int] = None
    variant: Optional[str] = None
    maxLen: Optional[int] = None
    heads: Optional[int] = None
    causal: Optional[bool] = None
    attnOnly: Optional[bool] = None
    hidden: Optional[int] = None
    in_: Optional[int] = None
    out: Optional[int] = None
    bias: Optional[bool] = None
    tiedTo: Optional[str] = None
    #: Keys this package does not know yet; preserved on round trip.
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> JSON:
        d: JSON = {}
        for key in _PARAM_KEYS:
            _put(d, key, getattr(self, "in_" if key == "in" else key))
        for key, value in self.extra.items():
            if key not in d:
                d[key] = value
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BlockParams":
        kwargs = {("in_" if k == "in" else k): d[k] for k in _PARAM_KEYS if k in d}
        extra = {k: v for k, v in d.items() if k not in _PARAM_KEYS}
        return cls(extra=extra, **kwargs)


@dataclass
class Block:
    id: str
    kind: str
    params: BlockParams = field(default_factory=BlockParams)
    inputs: List[str] = field(default_factory=list)
    label: Optional[str] = None
    group: Optional[str] = None
    at: Optional[Dict[str, float]] = None

    def to_dict(self) -> JSON:
        # Key order follows the TypeScript interface.
        d: JSON = {"id": self.id, "kind": self.kind}
        _put(d, "label", self.label)
        d["params"] = self.params.to_dict()
        d["inputs"] = list(self.inputs)
        _put(d, "group", self.group)
        if self.at is not None:
            d["at"] = {"x": self.at["x"], "y": self.at["y"]}
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Block":
        at = d.get("at")
        return cls(
            id=d["id"],
            kind=d["kind"],
            label=d.get("label"),
            params=BlockParams.from_dict(d.get("params") or {}),
            inputs=list(d.get("inputs") or []),
            group=d.get("group"),
            at=None if at is None else {"x": at["x"], "y": at["y"]},
        )


@dataclass
class DatasetRef:
    id: str
    provenance: str  # "synthetic" | "real" | "yours"
    seed: Optional[int] = None
    options: Optional[Dict[str, Union[int, float, str, bool]]] = None

    def to_dict(self) -> JSON:
        d: JSON = {"id": self.id, "provenance": self.provenance}
        _put(d, "seed", self.seed)
        _put(d, "options", None if self.options is None else dict(self.options))
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "DatasetRef":
        opts = d.get("options")
        return cls(d["id"], d["provenance"], d.get("seed"), None if opts is None else dict(opts))


@dataclass
class TrainConfig:
    optimizer: str  # "adamw" | "sgd"
    lr: float
    batch: int
    steps: int
    seed: int
    weightDecay: Optional[float] = None

    def to_dict(self) -> JSON:
        d: JSON = {"optimizer": self.optimizer, "lr": self.lr}
        _put(d, "weightDecay", self.weightDecay)
        d.update(batch=self.batch, steps=self.steps, seed=self.seed)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrainConfig":
        return cls(
            optimizer=d["optimizer"],
            lr=d["lr"],
            batch=d["batch"],
            steps=d["steps"],
            seed=d["seed"],
            weightDecay=d.get("weightDecay"),
        )


@dataclass
class Origin:
    kind: str  # "lab" | "torch" | "template"
    source: Optional[str] = None

    def to_dict(self) -> JSON:
        d: JSON = {"kind": self.kind}
        _put(d, "source", self.source)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Origin":
        return cls(d["kind"], d.get("source"))


@dataclass
class ModelSpec:
    name: str
    blocks: List[Block]
    output: str
    origin: Origin = field(default_factory=lambda: Origin("lab"))
    v: int = SPEC_VERSION
    dataset: Optional[DatasetRef] = None
    train: Optional[TrainConfig] = None
    #: Not part of the spec and never serialised. ``from_module`` fills it
    #: with block id -> where that block came from in the nn.Module, so that
    #: ``capture`` can key tensors by the same block ids.
    bindings: Dict[str, str] = field(default_factory=dict, repr=False, compare=False)

    def to_dict(self) -> JSON:
        d: JSON = {
            "v": self.v,
            "name": self.name,
            "blocks": [b.to_dict() for b in self.blocks],
            "output": self.output,
        }
        _put(d, "dataset", None if self.dataset is None else self.dataset.to_dict())
        _put(d, "train", None if self.train is None else self.train.to_dict())
        d["origin"] = self.origin.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ModelSpec":
        v = d.get("v", SPEC_VERSION)
        if v != SPEC_VERSION:
            raise ValueError(f"unsupported spec version {v!r}; this package reads v{SPEC_VERSION}")
        return cls(
            v=v,
            name=d["name"],
            blocks=[Block.from_dict(b) for b in d["blocks"]],
            output=d["output"],
            dataset=None if d.get("dataset") is None else DatasetRef.from_dict(d["dataset"]),
            train=None if d.get("train") is None else TrainConfig.from_dict(d["train"]),
            origin=Origin.from_dict(d.get("origin") or {"kind": "lab"}),
        )

    def to_json(self, indent: Optional[int] = None) -> str:
        if indent is None:
            return dumps(self)
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False, allow_nan=False)

    @classmethod
    def from_json(cls, text: Union[str, bytes]) -> "ModelSpec":
        return cls.from_dict(json.loads(text))

    def block(self, block_id: str) -> Block:
        for b in self.blocks:
            if b.id == block_id:
                return b
        raise KeyError(block_id)


def to_json(spec: ModelSpec) -> str:
    return spec.to_json()


def from_json(text: Union[str, bytes]) -> ModelSpec:
    return ModelSpec.from_json(text)


# ── Tensors on the wire ──────────────────────────────────────────────────────


def encode_f32(values: Sequence[float]) -> str:
    """Standard base64 (with padding) of little-endian float32 values."""
    arr = array("f", values)
    if sys.byteorder != "little":
        arr.byteswap()
    return base64.b64encode(arr.tobytes()).decode("ascii")


def decode_f32(b64: str) -> List[float]:
    raw = base64.b64decode(b64)
    if len(raw) % 4:
        raise ValueError("float32 payload length is not a multiple of 4")
    return list(struct.unpack(f"<{len(raw) // 4}f", raw))


# ── Trace (live mode, in-memory form) ────────────────────────────────────────


@dataclass
class TensorRecord:
    key: str
    shape: List[int]
    #: Row-major float32 values. A Float32Array in TypeScript; a list here.
    data: List[float]

    def to_dict(self) -> JSON:
        return {"key": self.key, "shape": list(self.shape), "data": list(self.data)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TensorRecord":
        return cls(d["key"], list(d["shape"]), list(d["data"]))


@dataclass
class MatmulRecord:
    key: str
    a: str
    b: str
    c: str
    transposeB: bool

    def to_dict(self) -> JSON:
        return {"key": self.key, "a": self.a, "b": self.b, "c": self.c, "transposeB": self.transposeB}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "MatmulRecord":
        return cls(d["key"], d["a"], d["b"], d["c"], bool(d["transposeB"]))


@dataclass
class Trace:
    specHash: str
    tokens: List[int]
    tensors: List[TensorRecord]
    matmuls: List[MatmulRecord] = field(default_factory=list)
    loss: Optional[float] = None
    interventions: List[str] = field(default_factory=list)

    def to_dict(self) -> JSON:
        d: JSON = {
            "specHash": self.specHash,
            "tokens": list(self.tokens),
            "tensors": [t.to_dict() for t in self.tensors],
            "matmuls": [m.to_dict() for m in self.matmuls],
        }
        _put(d, "loss", self.loss)
        d["interventions"] = list(self.interventions)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Trace":
        return cls(
            specHash=d["specHash"],
            tokens=list(d["tokens"]),
            tensors=[TensorRecord.from_dict(t) for t in d["tensors"]],
            matmuls=[MatmulRecord.from_dict(m) for m in d.get("matmuls", [])],
            loss=d.get("loss"),
            interventions=list(d.get("interventions", [])),
        )


# ── Bridge ───────────────────────────────────────────────────────────────────


@dataclass
class FrameTensor:
    """One tensor in a TraceFrame.

    ``key``, ``shape`` and ``b64`` are exactly the TypeScript fields.
    ``shape`` is always the tensor's true, full shape.

    The two extra fields keep the frame honest when a tensor was too large to
    send whole:

    ``clipped``    True when ``b64`` holds fewer values than ``shape`` implies.
                   Always present, so a reader never has to guess.
    ``dataShape``  Present only when clipped: the shape of the values that
                   were sent. They are the leading corner of the tensor, i.e.
                   ``tensor[0:dataShape[0], 0:dataShape[1], ...]``, row-major.
                   Every value sent is a real value at its true index.
    """

    key: str
    shape: List[int]
    b64: str
    clipped: bool = False
    dataShape: Optional[List[int]] = None

    def to_dict(self) -> JSON:
        d: JSON = {"key": self.key, "shape": list(self.shape), "b64": self.b64, "clipped": self.clipped}
        _put(d, "dataShape", None if self.dataShape is None else list(self.dataShape))
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FrameTensor":
        ds = d.get("dataShape")
        return cls(d["key"], list(d["shape"]), d["b64"], bool(d.get("clipped", False)), None if ds is None else list(ds))

    def values(self) -> List[float]:
        return decode_f32(self.b64)


@dataclass
class TraceFrame:
    """Streamed over /events, one per forward pass or training checkpoint.

    ``uncaptured`` (extra, optional) lists block ids that exist in the spec
    but have no tensor in this frame, so the Lab can say "not captured"
    instead of drawing nothing silently.
    """

    step: int
    tensors: List[FrameTensor] = field(default_factory=list)
    loss: Optional[float] = None
    uncaptured: Optional[List[str]] = None

    def to_dict(self) -> JSON:
        d: JSON = {"step": self.step}
        _put(d, "loss", self.loss)
        d["tensors"] = [t.to_dict() for t in self.tensors]
        _put(d, "uncaptured", None if self.uncaptured is None else list(self.uncaptured))
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TraceFrame":
        unc = d.get("uncaptured")
        return cls(
            step=int(d["step"]),
            loss=d.get("loss"),
            tensors=[FrameTensor.from_dict(t) for t in d.get("tensors", [])],
            uncaptured=None if unc is None else list(unc),
        )

    @property
    def clipped(self) -> List[str]:
        """Keys of tensors that were clipped in this frame."""
        return [t.key for t in self.tensors if t.clipped]

    def tensor(self, key: str) -> FrameTensor:
        for t in self.tensors:
            if t.key == key:
                return t
        raise KeyError(key)


@dataclass
class BridgeHello:
    spec: ModelSpec
    producer: Dict[str, str] = field(default_factory=lambda: {"name": "papertoanything", "version": _version()})
    protocol: str = "pta-bridge"
    v: int = 1

    def to_dict(self) -> JSON:
        return {
            "protocol": self.protocol,
            "v": self.v,
            "producer": dict(self.producer),
            "spec": self.spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BridgeHello":
        if d.get("protocol") != "pta-bridge":
            raise ValueError("not a pta-bridge hello")
        return cls(spec=ModelSpec.from_dict(d["spec"]), producer=dict(d["producer"]), protocol=d["protocol"], v=d["v"])


# ── Watch: live training health ──────────────────────────────────────────────
#
# Mirrors TensorStats / BlockHealth / HealthFrame / Diagnosis in spec.ts. The
# producer (this package) only computes reductions; diagnoses are derived by
# the viewer from frames, so Diagnosis is here for parsing only.

SEVERITIES = ("info", "warn", "error")


def _clean(d: JSON) -> JSON:
    return {k: v for k, v in d.items() if v is not None}


@dataclass
class TensorStats:
    mean: float
    std: float
    absMax: float
    zeroFrac: float
    nanCount: int
    infCount: int
    satFrac: Optional[float] = None
    #: 32-bin histogram over [-absMax, absMax].
    hist: Optional[List[float]] = None

    def to_dict(self) -> JSON:
        return _clean(
            {
                "mean": self.mean,
                "std": self.std,
                "absMax": self.absMax,
                "zeroFrac": self.zeroFrac,
                "satFrac": self.satFrac,
                "nanCount": self.nanCount,
                "infCount": self.infCount,
                "hist": None if self.hist is None else list(self.hist),
            }
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TensorStats":
        h = d.get("hist")
        return cls(d["mean"], d["std"], d["absMax"], d["zeroFrac"], int(d["nanCount"]), int(d["infCount"]), d.get("satFrac"), None if h is None else list(h))


@dataclass
class HeadPattern:
    """One head's attention pattern: ``size`` x ``size`` float32, row-major,
    base64 little-endian. It is the top-left corner (queries 0..size-1, keys
    0..size-1) of batch element 0, i.e. real values at their real indices."""

    head: int
    size: int
    b64: str

    def to_dict(self) -> JSON:
        return {"head": self.head, "size": self.size, "b64": self.b64}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HeadPattern":
        return cls(int(d["head"]), int(d["size"]), d["b64"])


@dataclass
class BlockHealth:
    """Per-block health. ``hidden`` is an extension to spec.ts: for an mlp
    block, the stats of its inner activation's output (where dead ReLUs
    and saturated units are visible; ``act`` is the block's output)."""

    id: str
    act: Optional[TensorStats] = None
    grad: Optional[TensorStats] = None
    weightNorm: Optional[float] = None
    weightGradNorm: Optional[float] = None
    updateRatio: Optional[float] = None
    headEntropy: Optional[List[float]] = None
    headPattern: Optional[List[HeadPattern]] = None
    hidden: Optional[TensorStats] = None

    def to_dict(self) -> JSON:
        return _clean(
            {
                "id": self.id,
                "act": None if self.act is None else self.act.to_dict(),
                "hidden": None if self.hidden is None else self.hidden.to_dict(),
                "grad": None if self.grad is None else self.grad.to_dict(),
                "weightNorm": self.weightNorm,
                "weightGradNorm": self.weightGradNorm,
                "updateRatio": self.updateRatio,
                "headEntropy": None if self.headEntropy is None else list(self.headEntropy),
                "headPattern": None if self.headPattern is None else [p.to_dict() for p in self.headPattern],
            }
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BlockHealth":
        hp = d.get("headPattern")
        he = d.get("headEntropy")
        return cls(
            id=d["id"],
            act=None if d.get("act") is None else TensorStats.from_dict(d["act"]),
            grad=None if d.get("grad") is None else TensorStats.from_dict(d["grad"]),
            weightNorm=d.get("weightNorm"),
            weightGradNorm=d.get("weightGradNorm"),
            updateRatio=d.get("updateRatio"),
            headEntropy=None if he is None else list(he),
            headPattern=None if hp is None else [HeadPattern.from_dict(p) for p in hp],
            hidden=None if d.get("hidden") is None else TensorStats.from_dict(d["hidden"]),
        )


@dataclass
class HealthFrame:
    """One training checkpoint's health.

    ``loss`` is required in spec.ts. When the producer could not learn the
    loss (no ``w.step(loss)`` call and the model does not return a scalar),
    it is serialised as JSON ``null`` rather than invented.
    """

    step: int
    t: float
    loss: Optional[float]
    blocks: List[BlockHealth] = field(default_factory=list)
    valLoss: Optional[float] = None
    lr: Optional[float] = None
    gradNorm: Optional[float] = None
    timing: Optional[Dict[str, float]] = None
    memory: Optional[int] = None

    def to_dict(self) -> JSON:
        d: JSON = {"step": self.step, "t": self.t, "loss": self.loss}
        d.update(
            _clean(
                {
                    "valLoss": self.valLoss,
                    "lr": self.lr,
                    "gradNorm": self.gradNorm,
                    "timing": None if not self.timing else dict(self.timing),
                    "memory": self.memory,
                }
            )
        )
        d["blocks"] = [b.to_dict() for b in self.blocks]
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "HealthFrame":
        return cls(
            step=int(d["step"]),
            t=d["t"],
            loss=d.get("loss"),
            blocks=[BlockHealth.from_dict(b) for b in d.get("blocks", [])],
            valLoss=d.get("valLoss"),
            lr=d.get("lr"),
            gradNorm=d.get("gradNorm"),
            timing=None if d.get("timing") is None else dict(d["timing"]),
            memory=d.get("memory"),
        )

    def block(self, block_id: str) -> BlockHealth:
        for b in self.blocks:
            if b.id == block_id:
                return b
        raise KeyError(block_id)


@dataclass
class Diagnosis:
    rule: str
    severity: str
    blocks: List[str]
    since: int
    title: str
    evidence: Dict[str, Union[float, str]]
    suggestion: Optional[str] = None
    reference: Optional[str] = None

    def to_dict(self) -> JSON:
        d: JSON = {
            "rule": self.rule,
            "severity": self.severity,
            "blocks": list(self.blocks),
            "since": self.since,
            "title": self.title,
            "evidence": dict(self.evidence),
        }
        _put(d, "suggestion", self.suggestion)
        _put(d, "reference", self.reference)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Diagnosis":
        return cls(d["rule"], d["severity"], list(d["blocks"]), int(d["since"]), d["title"], dict(d["evidence"]), d.get("suggestion"), d.get("reference"))


def _version() -> str:
    from . import __version__

    return __version__
