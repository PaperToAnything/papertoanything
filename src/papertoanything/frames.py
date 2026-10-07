"""Encode tensors into TraceFrame entries, clipping honestly.

Works on torch tensors, numpy arrays, or (shape, flat list) pairs. torch and
numpy are imported lazily and never required.

Clipping rule. A tensor with more than ``max_elems`` values is cut to its
leading corner: leading axes are reduced first (batch before sequence before
features), so the trailing axes, which carry features, survive whole whenever
they fit. The entry keeps the true ``shape``, sets ``clipped=True`` and records
``dataShape``; every value sent is the real value at its real index.
"""

from __future__ import annotations

import base64
import math
import sys
from array import array
from typing import Any, List, Optional, Sequence, Tuple

from .spec import FrameTensor, encode_f32

__all__ = ["clip_shape", "encode_tensor", "frame_tensor_from_values"]


def clip_shape(shape: Sequence[int], max_elems: int) -> List[int]:
    """Shape of the leading corner that fits in ``max_elems`` values."""
    if max_elems < 1:
        raise ValueError("max_elems must be at least 1")
    d = [int(s) for s in shape]
    if math.prod(d) <= max_elems:
        return d
    for i in range(len(d)):
        rest = math.prod(d[i + 1 :])
        if rest <= max_elems:
            d[i] = max(1, min(d[i], max_elems // rest))
            return d
        d[i] = 1
    return d


def frame_tensor_from_values(key: str, shape: Sequence[int], flat: Sequence[float], max_elems: int = 65536) -> FrameTensor:
    """Build an entry from a row-major flat list (no torch needed)."""
    shape = [int(s) for s in shape]
    if len(flat) != math.prod(shape):
        raise ValueError(f"{key}: {len(flat)} values for shape {shape}")
    ds = clip_shape(shape, max_elems)
    if ds == shape:
        return FrameTensor(key, shape, encode_f32(flat), clipped=False)
    # Gather the leading corner from the flat row-major list.
    strides = [math.prod(shape[i + 1 :]) for i in range(len(shape))]
    out: List[float] = []

    def walk(axis: int, offset: int) -> None:
        if axis == len(shape):
            out.append(flat[offset])
            return
        for j in range(ds[axis]):
            walk(axis + 1, offset + j * strides[axis])

    walk(0, 0)
    return FrameTensor(key, shape, encode_f32(out), clipped=True, dataShape=ds)


def _torch_bytes(t: Any) -> Tuple[bytes, int]:
    import torch

    t = t.detach()
    if t.is_complex():
        raise TypeError("complex tensors are not representable as float32")
    t = t.to(device="cpu", dtype=torch.float32).contiguous()
    try:
        import numpy as np  # noqa: F401

        raw = t.numpy().astype("<f4", copy=False).tobytes()
    except Exception:
        arr = array("f", t.reshape(-1).tolist())
        if sys.byteorder != "little":
            arr.byteswap()
        raw = arr.tobytes()
    return raw, t.numel()


def encode_tensor(key: str, t: Any, max_elems: int = 65536) -> FrameTensor:
    """Encode a torch tensor or numpy array as a FrameTensor."""
    if hasattr(t, "detach"):  # torch
        shape = [int(s) for s in t.shape]
        ds = clip_shape(shape, max_elems)
        clipped = ds != shape
        if clipped:
            t = t[tuple(slice(0, n) for n in ds)]
        raw, _ = _torch_bytes(t)
        return FrameTensor(key, shape, base64.b64encode(raw).decode("ascii"), clipped=clipped, dataShape=ds if clipped else None)
    if hasattr(t, "shape") and hasattr(t, "astype"):  # numpy
        shape = [int(s) for s in t.shape]
        ds = clip_shape(shape, max_elems)
        clipped = ds != shape
        if clipped:
            t = t[tuple(slice(0, n) for n in ds)]
        raw = t.astype("<f4").tobytes(order="C")
        return FrameTensor(key, shape, base64.b64encode(raw).decode("ascii"), clipped=clipped, dataShape=ds if clipped else None)
    raise TypeError(f"{key}: cannot encode {type(t).__name__}; pass a tensor, an array, or use frame_tensor_from_values")


def scalar(x: Any) -> Optional[float]:
    """A Python float from a 0-d tensor / number, or None."""
    try:
        if hasattr(x, "detach"):
            if x.numel() != 1:
                return None
            v = float(x.detach().reshape(()).item())
        else:
            v = float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None
