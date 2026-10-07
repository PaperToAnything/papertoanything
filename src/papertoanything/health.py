"""``pta.watch``: live training health for a real PyTorch loop.

One line in a training script or notebook::

    w = pta.watch(model, optimizer)          # opens a local viewer tab
    for batch in data:
        loss = model(batch)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        w.step(loss)                         # optional: report the loss

Every ``every`` steps it emits one HealthFrame (see ``spec.HealthFrame``):
per-block activation and gradient statistics, parameter and gradient norms,
the update ratio ||dtheta|| / ||theta||, attention entropy and a sampled
pattern where the attention probabilities can be observed, timings and device
memory. Frames go to the local same-origin server (SSE ``event: health``,
history at ``GET /health?since=``) and to a JSONL run file for replay.

Cost model. Between sampled steps every hook returns after one integer
comparison. On a sampled step the statistics are on-device reductions that
are copied to the host once per tensor. Gradients are observed through
tensor hooks attached to module outputs only on sampled steps, so no
``register_full_backward_hook`` wrapper sits in the graph on the other steps.
The update ratio needs a copy of the parameters on sampled steps; pass
``update_ratio=False`` on very large models.

Diagnoses (dead ReLUs, exploding updates, collapsed attention, ...) are NOT
computed here: the viewer derives them from frames with rules written once in
the TypeScript engine. The only judgement made in Python is an optional
console warning when a NaN or Inf appears.
"""

from __future__ import annotations

import json
import math
import os
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union

from .spec import BlockHealth, HeadPattern, HealthFrame, ModelSpec, TensorStats, dumps

__all__ = ["Watch", "watch", "tensor_stats", "RunFile"]

HIST_BINS = 32
PATTERN_SIZE = 32
_SAT = {"tanh": 0.99, "sigmoid": 0.99}


def _torch():
    try:
        import torch
    except ImportError:  # pragma: no cover
        raise ImportError("pta.watch needs PyTorch: pip install 'papertoanything[torch]'") from None
    return torch


# ── Statistics ───────────────────────────────────────────────────────────────


def tensor_stats(t: Any, hist: bool = True, sat: Optional[str] = None) -> TensorStats:
    """Reduce a tensor to TensorStats with one device-to-host copy.

    mean/std/absMax/hist are over finite values only; NaN and Inf are
    counted separately so one bad value does not hide the rest.
    ``satFrac`` is only computed for bounded activations (``sat`` is
    "tanh" or "sigmoid"): the fraction with |x| > 0.99 (tanh) or with x
    outside [0.01, 0.99] (sigmoid).
    """
    torch = _torch()
    with torch.no_grad():
        x = t.detach()
        if not x.is_floating_point():
            x = x.float()
        x = x.reshape(-1)
        n = x.numel()
        if n == 0:
            return TensorStats(0.0, 0.0, 0.0, 0.0, 0, 0)
        x = x.float()
        nan = torch.isnan(x)
        inf = torch.isinf(x)
        finite = ~(nan | inf)
        xf = torch.where(finite, x, torch.zeros((), dtype=x.dtype, device=x.device))
        nf = finite.sum()
        nf_safe = nf.clamp(min=1).to(x.dtype)
        mean = xf.sum() / nf_safe
        var = (torch.where(finite, (x - mean) ** 2, torch.zeros((), dtype=x.dtype, device=x.device))).sum() / nf_safe
        absmax = xf.abs().max()
        parts = [mean, var.sqrt(), absmax, (x == 0).sum().to(x.dtype) / n, nan.sum().to(x.dtype), inf.sum().to(x.dtype)]
        if sat == "tanh":
            parts.append((xf.abs() > _SAT["tanh"]).sum().to(x.dtype) / n)
        elif sat == "sigmoid":
            parts.append(((xf > 0.99) | (xf < 0.01)).sum().to(x.dtype) / n)
        vals = torch.stack(parts)
        h = None
        if hist:
            am = float(absmax)
            if am > 0 and math.isfinite(am):
                h = torch.histc(xf[finite], bins=HIST_BINS, min=-am, max=am)
            else:
                h = torch.zeros(HIST_BINS, device=x.device)
                h[HIST_BINS // 2] = float(nf)
        vals = vals.cpu().tolist()
        hist_list = None if h is None else [float(v) for v in h.cpu().tolist()]
    st = TensorStats(
        mean=_f(vals[0]),
        std=_f(vals[1]),
        absMax=_f(vals[2]),
        zeroFrac=_f(vals[3]),
        nanCount=int(vals[4]),
        infCount=int(vals[5]),
        satFrac=_f(vals[6]) if len(vals) > 6 else None,
        hist=hist_list,
    )
    return st


def _f(v: float) -> float:
    v = float(v)
    return v if math.isfinite(v) else 0.0


def _first_tensor(out: Any) -> Any:
    torch = _torch()
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (list, tuple)):
        for o in out:
            t = _first_tensor(o)
            if t is not None:
                return t
    if isinstance(out, dict):
        for o in out.values():
            t = _first_tensor(o)
            if t is not None:
                return t
    if hasattr(out, "to_tuple"):
        return _first_tensor(out.to_tuple())
    return None


def _scalar_loss(out: Any) -> Any:
    """A 0-d float tensor in a model's output (nanoGPT returns (logits, loss))."""
    torch = _torch()
    if isinstance(out, torch.Tensor):
        return out if out.dim() == 0 and out.is_floating_point() else None
    if isinstance(out, (list, tuple)):
        for o in out:
            r = _scalar_loss(o)
            if r is not None:
                return r
    if isinstance(out, dict) or hasattr(out, "keys"):
        try:
            v = out["loss"]
            return v if isinstance(v, torch.Tensor) and v.dim() == 0 else None
        except Exception:
            return None
    return None


def _num(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x.detach().float().item()) if hasattr(x, "detach") else float(x)
    except Exception:
        return None
    return v if math.isfinite(v) else None


def _attention_from_probs(p: Any) -> tuple:
    """Per-head mean row entropy (nats) and a sampled pattern per head.

    ``p`` is [B, H, T, S] attention probabilities.
    """
    from .spec import encode_f32

    torch = _torch()
    with torch.no_grad():
        p = p.detach().float()
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)  # [B, H, T]
        ent = ent.mean(dim=(0, 2)).cpu().tolist()
        size = min(PATTERN_SIZE, p.shape[-1], p.shape[-2])
        sample = p[0, :, :size, :size].cpu()
    patterns = [HeadPattern(head=h, size=size, b64=encode_f32(sample[h].reshape(-1).tolist())) for h in range(sample.shape[0])]
    return [_f(e) for e in ent], patterns


# ── Run file ─────────────────────────────────────────────────────────────────


class RunFile:
    """JSONL replay file. Line 1 is a header, then one event per line::

        {"protocol":"pta-run","v":1,"producer":{...},"spec":{...}}
        {"event":"health","data":{HealthFrame}}
    """

    def __init__(self, path: str, spec: Optional[ModelSpec]) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "w", encoding="utf-8", newline="\n")
        self._header_written = False
        if spec is not None:
            self.write_header(spec)

    def write_header(self, spec: ModelSpec) -> None:
        if self._header_written:
            return
        from . import __version__

        header = {"protocol": "pta-run", "v": 1, "producer": {"name": "papertoanything", "version": __version__}, "spec": spec.to_dict()}
        self._fh.write(dumps(header) + "\n")
        self._header_written = True
        self._fh.flush()

    def write(self, event: str, data: Any) -> None:
        if self._fh.closed:
            return
        self._fh.write(dumps({"event": event, "data": data.to_dict() if hasattr(data, "to_dict") else data}) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    @staticmethod
    def read(path: str) -> tuple:
        """Return (spec dict or None, list of HealthFrame)."""
        spec = None
        frames: List[HealthFrame] = []
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                if not line.strip():
                    continue
                obj = json.loads(line)
                if i == 0 and obj.get("protocol") == "pta-run":
                    spec = obj.get("spec")
                elif obj.get("event") == "health":
                    frames.append(HealthFrame.from_dict(obj["data"]))
        return spec, frames


# ── The watcher ──────────────────────────────────────────────────────────────


class _Tracked:
    __slots__ = ("bid", "path", "module", "kind", "sat", "params", "is_attention", "heads", "stats")

    def __init__(self, bid: str, path: str, module: Any, kind: str, sat: Optional[str], heads: Optional[int]) -> None:
        self.bid = bid
        self.path = path
        self.module = module
        self.kind = kind
        self.sat = sat
        self.params = [p for p in module.parameters() if p.requires_grad]
        self.is_attention = kind == "attention"
        self.heads = heads
        # A learned posenc is an nn.Embedding whose output is the positional
        # table, not the block's output (x + table): track its weights only.
        self.stats = not (kind == "posenc")


class Watch:
    """Handle returned by ``pta.watch``. See the module docstring."""

    def __init__(
        self,
        model: Any,
        optimizer: Any = None,
        every: int = 10,
        val_loss: Union[None, float, Callable[[], float]] = None,
        open: bool = True,
        *,
        example_input: Any = None,
        server: Any = None,
        run_file: Union[None, bool, str] = True,
        hist: bool = True,
        update_ratio: bool = True,
        attention: bool = True,
        val_every: Optional[int] = None,
        warn_nonfinite: bool = True,
        spec: Optional[ModelSpec] = None,
        sinks: Optional[List[Callable[[HealthFrame], None]]] = None,
        lab_dir: Optional[str] = None,
    ) -> None:
        torch = _torch()
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"pta.watch expects an nn.Module, got {type(model).__name__}")
        if every < 1:
            raise ValueError("every must be at least 1")
        self.model = model
        self.optimizer = optimizer
        self.every = int(every)
        self.val_loss = val_loss
        self.val_every = val_every or self.every * 10
        self.hist = hist
        self.want_update_ratio = update_ratio
        self.want_attention = attention
        self.warn_nonfinite = warn_nonfinite
        self.sinks: List[Callable[[HealthFrame], None]] = list(sinks or [])
        self.frames: List[HealthFrame] = []
        self.step_count = 0
        self.closed = False
        self._deferred: Optional[HealthFrame] = None
        self._loss_tensor: Any = None
        self.spec: Optional[ModelSpec] = spec
        self.tracked: Dict[str, _Tracked] = {}
        self._t0 = time.perf_counter()
        self._handles: List[Any] = []
        self._block_handles: List[Any] = []
        self._pending: Dict[str, BlockHealth] = {}
        self._loss: Optional[float] = None
        self._last_val: Optional[float] = None
        self._snapshot: Optional[Dict[str, List[Any]]] = None
        self._grad_norms: Optional[Dict[str, float]] = None
        self._total_grad_norm: Optional[float] = None
        self._warned: set = set()
        self._busy = False
        self._active = False  # inside a sampled training forward
        self._mode: Any = None
        self._attn_stack: List[str] = []
        self._times: Dict[str, float] = {}
        self._timing: Dict[str, float] = {}
        self._cuda = any(p.is_cuda for p in model.parameters())
        self.server = server
        self.run_file: Optional[RunFile] = None
        self.url: Optional[str] = None

        if run_file:
            path = run_file if isinstance(run_file, str) else os.path.join("pta-runs", f"{type(model).__name__}-{time.strftime('%Y%m%d-%H%M%S')}.pta")
            self.run_file = RunFile(path, None)

        if self.spec is None and example_input is not None:
            self._build_spec(example_input)
        if self.spec is not None:
            self._attach_blocks()

        self._handles.append(model.register_forward_pre_hook(self._root_pre, with_kwargs=True))
        self._handles.append(model.register_forward_hook(self._root_post))
        if optimizer is not None:
            self._handles.append(optimizer.register_step_pre_hook(self._opt_pre))
            self._handles.append(optimizer.register_step_post_hook(self._opt_post))

        if open or server is not None:
            self._ensure_server(open_browser=open, lab_dir=lab_dir)

    # ── setup ────────────────────────────────────────────────────────────────

    def _build_spec(self, example_input: Any) -> None:
        from .torch import analyze

        self._busy = True
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.spec = analyze(self.model, example_input).spec
        except Exception as e:  # never break the user's training loop
            warnings.warn(f"pta.watch: could not map the model ({e}); block stats disabled", stacklevel=3)
            from .spec import Block, BlockParams, Origin

            self.spec = ModelSpec(type(self.model).__name__, [Block("g0", "group", BlockParams(), [], label=type(self.model).__name__)], "g0", Origin("torch"))
        finally:
            self._busy = False
        if self.run_file is not None:
            self.run_file.write_header(self.spec)
        if self.server is not None:
            self.server.set_spec(self.spec)

    def _attach_blocks(self) -> None:
        assert self.spec is not None
        if self.run_file is not None:
            self.run_file.write_header(self.spec)
        modules = dict(self.model.named_modules())
        for bid, binding in self.spec.bindings.items():
            if not binding.startswith("m:"):
                continue
            path, _, call = binding[2:].rpartition("#")
            if call != "0" or path not in modules:
                continue
            block = self.spec.block(bid)
            if block.kind in ("group",) and block.inputs == []:
                continue
            sat = block.params.variant if block.kind == "activation" and block.params.variant in _SAT else None
            tr = _Tracked(bid, path, modules[path], block.kind, sat, block.params.heads)
            self.tracked[bid] = tr
            self._block_handles.append(tr.module.register_forward_pre_hook(self._make_block_pre(tr)))
            self._block_handles.append(tr.module.register_forward_hook(self._make_block_post(tr)))
            if block.kind == "mlp":
                self._attach_hidden(tr)

    def _attach_hidden(self, tr: _Tracked) -> None:
        from .torch import classify_module

        for child in tr.module.modules():
            role = classify_module(child)
            if child is not tr.module and role.kind == "activation":
                v = role.params.variant
                sat = v if v in _SAT else None

                def post(module: Any, args: tuple, output: Any, tr: _Tracked = tr, sat: Optional[str] = sat) -> None:
                    if not self._active or self._busy:
                        return
                    out = _first_tensor(output)
                    if out is None:
                        return
                    try:
                        bh = self._pending.setdefault(tr.bid, BlockHealth(tr.bid))
                        bh.hidden = tensor_stats(out, hist=self.hist, sat=sat)
                        self._check_finite(tr.bid, "hidden activation", bh.hidden)
                    except Exception:
                        pass

                self._block_handles.append(child.register_forward_hook(post))
                return

    def _ensure_server(self, open_browser: bool, lab_dir: Optional[str]) -> None:
        from .server import LocalServer

        if self.server is None:
            self.server = LocalServer(spec=self.spec, lab_dir=lab_dir, name=type(self.model).__name__).start()
        elif self.spec is not None:
            self.server.set_spec(self.spec)
        self.url = self.server.url
        print(f"papertoanything: watching {type(self.model).__name__} at {self.url}")
        if open_browser and not os.environ.get("PTA_NO_BROWSER"):
            import webbrowser

            try:
                webbrowser.open(self.url)
            except Exception:
                pass

    # ── hooks ────────────────────────────────────────────────────────────────

    def _sampling(self) -> bool:
        return self.step_count % self.every == 0

    def _root_pre(self, module: Any, args: tuple, kwargs: dict) -> None:
        if self._busy or self.closed or not module.training:
            return
        torch = _torch()
        if not torch.is_grad_enabled():
            return
        self._flush()
        if self.spec is None:
            from .torch import Inputs

            self._build_spec(Inputs(args, kwargs))
            self._attach_blocks()
        if not self._sampling():
            self._active = False
            return
        self._active = True
        now = time.perf_counter()
        if "step_end" in self._times:
            self._timing["data"] = (now - self._times["step_end"]) * 1000
        self._sync()
        self._times["fwd_start"] = time.perf_counter()
        if self.want_update_ratio and self.optimizer is None:
            self._take_snapshot()
        if self.want_attention and any(t.is_attention for t in self.tracked.values()):
            self._enter_mode()

    def _root_post(self, module: Any, args: tuple, output: Any) -> None:
        if not self._active or self._busy:
            return
        self._exit_mode()
        self._sync()
        now = time.perf_counter()
        self._times["fwd_end"] = now
        if "fwd_start" in self._times:
            self._timing["forward"] = (now - self._times["fwd_start"]) * 1000
        loss = _scalar_loss(output)
        if loss is not None:
            self._loss_tensor = loss
            if loss.requires_grad:
                loss.register_hook(self._on_loss_grad)

    def _on_loss_grad(self, g: Any) -> Any:
        self._times["bwd_start"] = time.perf_counter()
        return None

    def _make_block_pre(self, tr: _Tracked) -> Callable:
        def pre(module: Any, args: tuple) -> None:
            if not self._active or self._busy:
                return
            if tr.is_attention:
                self._attn_stack.append(tr.bid)

        return pre

    def _make_block_post(self, tr: _Tracked) -> Callable:
        def post(module: Any, args: tuple, output: Any) -> None:
            if not self._active or self._busy or not tr.stats:
                return
            try:
                if tr.is_attention and self._attn_stack and self._attn_stack[-1] == tr.bid:
                    self._attn_stack.pop()
                out = _first_tensor(output)
                if out is None:
                    return
                bh = self._pending.setdefault(tr.bid, BlockHealth(tr.bid))
                bh.act = tensor_stats(out, hist=self.hist, sat=tr.sat)
                self._check_finite(tr.bid, "activation", bh.act)
                if tr.is_attention and bh.headEntropy is None:
                    w = self._weights_from_output(module, output, tr)
                    if w is not None:
                        bh.headEntropy, bh.headPattern = _attention_from_probs(w)
                if out.requires_grad:
                    out.register_hook(self._make_grad_hook(tr))
            except Exception as e:  # stats must never break training
                self._warn_once(f"stats:{tr.bid}", f"pta.watch: skipped stats for {tr.bid}: {e}")

        return post

    def _make_grad_hook(self, tr: _Tracked) -> Callable:
        def hook(g: Any) -> None:
            try:
                bh = self._pending.setdefault(tr.bid, BlockHealth(tr.bid))
                bh.grad = tensor_stats(g, hist=self.hist)
                self._check_finite(tr.bid, "gradient", bh.grad)
            except Exception:
                pass
            return None

        return hook

    def _weights_from_output(self, module: Any, output: Any, tr: _Tracked) -> Any:
        """Per-head probabilities the module itself returned, if any."""
        torch = _torch()
        if not isinstance(output, (tuple, list)) or len(output) < 2:
            return None
        w = output[1]
        if not isinstance(w, torch.Tensor):
            return None
        if w.dim() == 4:
            return w
        if w.dim() == 3 and isinstance(module, torch.nn.MultiheadAttention) and module.num_heads == 1:
            return w.unsqueeze(1)
        return None  # head-averaged weights: per-head entropy unknowable, so omitted

    # Attention probabilities computed inside a module (softmax or SDPA). Only
    # active during sampled forwards.
    def _enter_mode(self) -> None:
        try:
            from torch.overrides import TorchFunctionMode
        except ImportError:  # pragma: no cover
            return
        watcher = self

        class _AttnMode(TorchFunctionMode):
            def __torch_function__(self, func, types, args=(), kwargs=None):  # type: ignore[override]
                kwargs = kwargs or {}
                out = func(*args, **kwargs)
                if watcher._attn_stack:
                    try:
                        watcher._observe(func, args, kwargs, out)
                    except Exception:
                        pass
                return out

        self._mode = _AttnMode()
        self._mode.__enter__()

    def _exit_mode(self) -> None:
        if self._mode is not None:
            mode, self._mode = self._mode, None
            try:
                mode.__exit__(None, None, None)
            except Exception:
                pass
        self._attn_stack.clear()

    def _observe(self, func: Any, args: tuple, kwargs: dict, out: Any) -> None:
        torch = _torch()
        bid = self._attn_stack[-1]
        bh = self._pending.setdefault(bid, BlockHealth(bid))
        if bh.headEntropy is not None:
            return
        name = getattr(func, "__name__", "")
        heads = self.tracked[bid].heads
        if name == "softmax" and isinstance(out, torch.Tensor) and out.dim() == 4 and (heads is None or out.shape[1] == heads):
            bh.headEntropy, bh.headPattern = _attention_from_probs(out)
        elif name == "scaled_dot_product_attention" and len(args) >= 2:
            q, k = args[0], args[1]
            if q.dim() != 4 or kwargs.get("attn_mask") is not None or len(args) > 3:
                return
            # SDPA never materialises the pattern; recompute it for batch
            # element 0 from the same q and k (same math, pre-dropout).
            with torch.no_grad():
                qq, kk = q[:1].float(), k[:1].float()
                scale = kwargs.get("scale") or 1.0 / math.sqrt(qq.shape[-1])
                s = (qq @ kk.transpose(-2, -1)) * scale
                if kwargs.get("is_causal"):
                    T, S = s.shape[-2], s.shape[-1]
                    mask = torch.ones(T, S, dtype=torch.bool, device=s.device).tril()
                    s = s.masked_fill(~mask, float("-inf"))
                p = torch.softmax(s, dim=-1)
            bh.headEntropy, bh.headPattern = _attention_from_probs(p)

    def _opt_pre(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._busy or self.closed or not self._sampling():
            return
        now = time.perf_counter()
        self._sync()
        start = self._times.get("bwd_start", self._times.get("fwd_end"))
        if start is not None:
            self._timing["backward"] = (time.perf_counter() - start) * 1000
        self._times["opt_start"] = time.perf_counter() if start is None else now
        self._grad_norms, self._total_grad_norm = self._compute_grad_norms()
        if self.want_update_ratio:
            self._take_snapshot()

    def _opt_post(self, optimizer: Any, args: Any, kwargs: Any) -> None:
        if self._busy or self.closed:
            return
        if self._sampling():
            self._sync()
            if "opt_start" in self._times:
                self._timing["optim"] = (time.perf_counter() - self._times["opt_start"]) * 1000
            self._deferred = self._finish()
        self.step_count += 1
        self._times["step_end"] = time.perf_counter()

    # ── public API ───────────────────────────────────────────────────────────

    def step(self, loss: Any = None) -> Optional[HealthFrame]:
        """Mark the end of a training step, optionally reporting its loss.

        Call it after ``optimizer.step()``. When an optimizer was passed to
        ``watch``, its step hook already counts steps, and ``step(loss)``
        only attaches the loss to the frame that step produced (frames are
        held until the loss arrives or the next forward pass starts).
        Without an optimizer, each call counts one step.
        Returns the frame emitted by this call, if any.
        """
        if self.closed:
            return None
        v = None
        if loss is not None:
            v = _num(loss)
            if v is None and self.warn_nonfinite and _is_nonfinite(loss):
                self._warn_once("loss", f"pta.watch: loss is not finite at step {self.step_count}")
        if self.optimizer is not None:
            frame = self._deferred
            if frame is not None and v is not None:
                frame.loss = v
            self._flush()
            return frame
        self._loss = v
        frame = None
        if self._sampling():
            frame = self._finish()
            self._deferred = frame
            self._flush()
        self._loss = None
        self.step_count += 1
        self._times["step_end"] = time.perf_counter()
        return frame

    def _flush(self) -> None:
        frame, self._deferred = self._deferred, None
        if frame is not None:
            self._emit(frame)

    def log(self, loss: Any = None, val_loss: Any = None) -> None:
        """Record a loss or validation loss without counting a step."""
        if loss is not None:
            self._loss = _num(loss)
        if val_loss is not None:
            self._last_val = _num(val_loss)

    def push(self, frame: HealthFrame) -> None:
        self._emit(frame)

    def close(self) -> None:
        if self.closed:
            return
        self._flush()
        self.closed = True
        self._exit_mode()
        for h in self._handles + self._block_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._handles.clear()
        self._block_handles.clear()
        if self.run_file is not None:
            self.run_file.close()

    def __enter__(self) -> "Watch":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<pta.Watch {type(self.model).__name__} step={self.step_count} frames={len(self.frames)} url={self.url}>"

    # ── internals ────────────────────────────────────────────────────────────

    def _sync(self) -> None:
        if self._cuda:
            torch = _torch()
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

    def _take_snapshot(self) -> None:
        torch = _torch()
        with torch.no_grad():
            self._snapshot = {bid: [p.detach().clone() for p in tr.params] for bid, tr in self.tracked.items() if tr.params}

    def _compute_grad_norms(self) -> tuple:
        torch = _torch()
        norms: Dict[str, float] = {}
        total_sq = None
        with torch.no_grad():
            for bid, tr in self.tracked.items():
                gs = [p.grad for p in tr.params if p.grad is not None]
                if not gs:
                    continue
                sq = sum((g.detach().float() ** 2).sum() for g in gs)
                norms[bid] = sq
            seen = set()
            for p in self.model.parameters():
                if p.grad is None or id(p) in seen:
                    continue
                seen.add(id(p))
                s = (p.grad.detach().float() ** 2).sum()
                total_sq = s if total_sq is None else total_sq + s
        out = {bid: math.sqrt(float(v)) for bid, v in norms.items()}
        total = math.sqrt(float(total_sq)) if total_sq is not None else None
        return out, total

    def _finish(self) -> Optional[HealthFrame]:
        torch = _torch()
        if self._grad_norms is None:
            self._grad_norms, self._total_grad_norm = self._compute_grad_norms()
        blocks: List[BlockHealth] = []
        with torch.no_grad():
            for bid, tr in self.tracked.items():
                bh = self._pending.get(bid) or BlockHealth(bid)
                if tr.params:
                    wsq = sum((p.detach().float() ** 2).sum() for p in tr.params)
                    bh.weightNorm = _f(math.sqrt(float(wsq)))
                    if bid in self._grad_norms:
                        bh.weightGradNorm = _f(self._grad_norms[bid])
                    snap = (self._snapshot or {}).get(bid)
                    if snap is not None:
                        dsq = sum(((p.detach().float() - s.float()) ** 2).sum() for p, s in zip(tr.params, snap))
                        if float(wsq) > 0:
                            bh.updateRatio = _f(math.sqrt(float(dsq)) / math.sqrt(float(wsq)))
                if bh.to_dict() != {"id": bid}:
                    blocks.append(bh)
        loss = self._loss
        if loss is None and getattr(self, "_loss_tensor", None) is not None:
            loss = _num(self._loss_tensor)
        lr = None
        if self.optimizer is not None:
            try:
                lr = float(self.optimizer.param_groups[0]["lr"])
            except Exception:
                lr = None
        if self.val_loss is not None and self.step_count % self.val_every == 0:
            try:
                self._last_val = _num(self.val_loss() if callable(self.val_loss) else self.val_loss)
            except Exception as e:
                self._warn_once("val", f"pta.watch: val_loss callback failed: {e}")
        memory = None
        if self._cuda:
            try:
                memory = int(torch.cuda.memory_allocated())
            except Exception:
                memory = None
        frame = HealthFrame(
            step=self.step_count,
            t=round((time.perf_counter() - self._t0) * 1000, 3),
            loss=loss,
            blocks=blocks,
            valLoss=self._last_val,
            lr=lr,
            gradNorm=None if self._total_grad_norm is None else _f(self._total_grad_norm),
            timing={k: round(v, 3) for k, v in self._timing.items()} or None,
            memory=memory,
        )
        self._last_val = None  # a validation loss is reported once, on the frame after it was measured
        self._pending = {}
        self._snapshot = None
        self._grad_norms = None
        self._total_grad_norm = None
        self._timing = {}
        self._loss_tensor = None
        self._active = False
        return frame

    def _emit(self, frame: HealthFrame) -> None:
        self.frames.append(frame)
        if len(self.frames) > 10000:
            del self.frames[:5000]
        if self.run_file is not None:
            self.run_file.write("health", frame)
        if self.server is not None:
            self.server.push_health(frame)
        for sink in self.sinks:
            try:
                sink(frame)
            except Exception as e:
                self._warn_once(f"sink:{id(sink)}", f"pta.watch: sink failed: {e}")

    def _check_finite(self, bid: str, what: str, st: TensorStats) -> None:
        if self.warn_nonfinite and (st.nanCount or st.infCount):
            self._warn_once(f"nf:{bid}:{what}", f"pta.watch: {st.nanCount} NaN / {st.infCount} Inf in the {what} of block {bid} at step {self.step_count}")

    def _warn_once(self, key: str, msg: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)


def _is_nonfinite(x: Any) -> bool:
    try:
        v = float(x.detach().float().item()) if hasattr(x, "detach") else float(x)
        return not math.isfinite(v)
    except Exception:
        return False


def watch(
    model: Any,
    optimizer: Any = None,
    every: int = 10,
    val_loss: Union[None, float, Callable[[], float]] = None,
    open: bool = True,
    **kwargs: Any,
) -> Watch:
    """Watch a PyTorch training run live. Returns a ``Watch`` handle.

    ``optimizer``  when given, steps are detected with its step hooks; without
                   it, call ``w.step(loss)`` once per training step.
    ``every``      emit one HealthFrame every N steps (hooks are idle between).
    ``val_loss``   a float, or a zero-argument callable evaluated every
                   ``val_every`` steps (default ``10 * every``).
    ``open``       start the local viewer on 127.0.0.1 and open a browser tab.

    Keyword options: ``example_input`` (map the model now instead of on the
    first training forward), ``server`` (reuse a LocalServer), ``run_file``
    (path, or False to disable; default ``./pta-runs/<Model>-<time>.pta``),
    ``hist``, ``update_ratio``, ``attention``, ``val_every``,
    ``warn_nonfinite``, ``sinks`` (callables receiving each HealthFrame),
    ``lab_dir``.
    """
    return Watch(model, optimizer, every, val_loss, open, **kwargs)
