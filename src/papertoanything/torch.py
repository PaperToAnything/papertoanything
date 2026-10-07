"""PyTorch -> ModelSpec, and live activations -> TraceFrame.

``from_module(model, example_input)`` maps a real ``nn.Module`` onto the
Lab's block vocabulary. It tries three strategies, in order, and never raises
for an unfamiliar model:

1. ``fx``    ``torch.fx`` symbolic trace, with recognised modules kept as
             leaves. Gives the true dataflow graph, including residual adds
             written as ``x + f(x)``. Fails on data-dependent control flow
             (``assert t <= block_size``, ``if x.sum() > 0``), which is common.
2. ``hooks`` Run one forward pass on ``example_input`` with forward hooks on
             every module plus a ``TorchFunctionMode`` that sees functional
             ops between modules. Same dataflow graph, and it survives
             control flow because it records what actually ran.
3. ``tree``  No input and no trace: walk the module tree in registration
             order and chain recognised modules. Edges are a guess and a
             warning says so.

Recognised modules (by type or by duck-typing, never by import):

* ``nn.Embedding``                  -> embedding (posenc "learned" when it is
                                       indexed by ``arange``)
* ``nn.MultiheadAttention``, and anything named like attention that carries
  a head count (nanoGPT ``CausalSelfAttention``, HF ``GPT2Attention``,
  ``BertSelfAttention``, ...)       -> attention
* modules named MLP / FeedForward / FFN with 2-3 linear children -> mlp
* ``nn.Linear``, HF ``Conv1D``      -> linear (unembed when its weight is an
                                       embedding's weight, or when it is the
                                       output layer and its width is a vocab)
* LayerNorm / RMSNorm (torch or custom, by name) -> norm
* ReLU, GELU, Tanh, Sigmoid, SiLU   -> activation
* Softmax / LogSoftmax              -> softmax;  CrossEntropyLoss -> loss
* Dropout, Identity                 -> transparent
* anything else that is a leaf      -> ``group`` labelled with its class name

Functional ops between modules: ``a + b`` of two different block outputs ->
add; ``F.relu`` etc. -> activation; ``F.softmax`` -> softmax;
``F.cross_entropy`` -> loss; ``F.layer_norm`` -> norm; ``F.linear`` -> linear;
``F.scaled_dot_product_attention`` outside a recognised module -> attention.
Every other op is transparent: its output inherits its inputs' sources.

Containers (``Block``, ``Sequential``, the root) become ``group`` blocks with
no inputs, and the blocks inside them point at them through ``Block.group``.

torch is imported lazily so ``import papertoanything`` works without it.
"""

from __future__ import annotations

import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .frames import encode_tensor, scalar
from .spec import Block, BlockParams, FrameTensor, ModelSpec, Origin, TraceFrame

__all__ = ["from_module", "capture", "analyze", "Inputs", "TraceWatch", "watch_trace", "classify_module"]

DEFAULT_MAX_ELEMS = 65536
DEFAULT_MAX_FRAME_ELEMS = 4_000_000


def _torch():
    try:
        import torch
    except ImportError:  # pragma: no cover - exercised only without torch
        raise ImportError("papertoanything.torch needs PyTorch: pip install 'papertoanything[torch]'") from None
    return torch


# ── Module classification ────────────────────────────────────────────────────

_ACTIVATIONS = {"relu": "relu", "gelu": "gelu", "tanh": "tanh", "sigmoid": "sigmoid", "silu": "silu", "swish": "silu"}
_HEAD_ATTRS = ("n_head", "num_heads", "n_heads", "num_attention_heads", "nhead", "heads")
_DIM_ATTRS = ("n_embd", "embed_dim", "hidden_size", "d_model", "all_head_size", "embed_size", "dim")
_QKV_ATTRS = ("c_attn", "qkv", "in_proj", "Wqkv", "to_qkv", "q_proj", "query", "wq", "q_lin")


@dataclass
class Role:
    """What a module becomes. ``role`` is block | pass | leafgroup | container."""

    role: str
    kind: Optional[str] = None
    params: BlockParams = field(default_factory=BlockParams)


def _linear_dims(m: Any) -> Optional[Tuple[int, int, bool]]:
    torch = _torch()
    if isinstance(m, torch.nn.Linear):
        return m.in_features, m.out_features, m.bias is not None
    # HF transformers Conv1D: a linear layer storing weight as [in, out].
    if type(m).__name__ == "Conv1D" and hasattr(m, "nf") and getattr(m, "weight", None) is not None and m.weight.dim() == 2:
        return int(m.weight.shape[0]), int(m.nf), getattr(m, "bias", None) is not None
    return None


def _activation_variant(m: Any) -> Optional[str]:
    torch = _torch()
    nn = torch.nn
    table = ((nn.ReLU, "relu"), (nn.GELU, "gelu"), (nn.Tanh, "tanh"), (nn.Sigmoid, "sigmoid"), (nn.SiLU, "silu"))
    for cls, v in table:
        if isinstance(m, cls):
            return v
    if list(m.children()):
        return None
    name = type(m).__name__.lower()
    if "gelu" in name:  # HF NewGELUActivation, GELUActivation, QuickGELU
        return "gelu"
    if name in ("silu", "swish", "siluactivation"):
        return "silu"
    return None


def _is_lower_triangular_mask(t: Any) -> bool:
    try:
        if t is None or t.dim() < 2 or t.shape[-1] != t.shape[-2] or t.shape[-1] < 2:
            return False
        m = t.reshape(-1, t.shape[-2], t.shape[-1])[0]
        n = min(int(m.shape[-1]), 8)
        m = m[:n, :n]
        return bool((m.triu(1) == 0).all()) and bool((m.tril() != 0).all())
    except Exception:
        return False


def _attention_params(m: Any) -> Optional[BlockParams]:
    torch = _torch()
    if isinstance(m, torch.nn.MultiheadAttention):
        return BlockParams(heads=m.num_heads, d=m.embed_dim, bias=m.in_proj_bias is not None)
    cname = type(m).__name__.lower()
    if "attention" not in cname and "attn" not in cname:
        return None
    heads = next((getattr(m, a) for a in _HEAD_ATTRS if isinstance(getattr(m, a, None), int)), None)
    if not heads:
        return None
    d = next((getattr(m, a) for a in _DIM_ATTRS if isinstance(getattr(m, a, None), int)), None)
    bias = None
    for a in _QKV_ATTRS:
        sub = getattr(m, a, None)
        dims = _linear_dims(sub) if isinstance(sub, torch.nn.Module) else None
        if dims:
            d = d or dims[0]
            bias = dims[2]
            break
    causal = None
    for a in ("is_causal", "causal"):
        if isinstance(getattr(m, a, None), bool):
            causal = getattr(m, a)
            break
    if causal is None and "causal" in cname:
        causal = True
    if causal is None:
        for bname in ("bias", "mask", "causal_mask", "tril", "attn_mask"):
            buf = dict(m.named_buffers(recurse=False)).get(bname)
            if buf is not None and _is_lower_triangular_mask(buf):
                causal = True
                break
    return BlockParams(heads=int(heads), d=d, causal=causal, bias=bias)


def _mlp_params(m: Any) -> Optional[BlockParams]:
    cname = type(m).__name__.lower()
    if not any(k in cname for k in ("mlp", "feedforward", "feed_forward", "ffn", "ffw")):
        return None
    children = list(m.children())
    lins = [d for d in (_linear_dims(c) for c in children) if d]
    if len(lins) not in (2, 3) or any(list(c.children()) and not _activation_variant(c) for c in children if not _linear_dims(c)):
        return None
    variant = next((v for v in (_activation_variant(c) for c in children) if v), None)
    return BlockParams(d=lins[0][0], hidden=lins[0][1], bias=lins[0][2], variant=variant)


def classify_module(m: Any) -> Role:
    """Decide what a module becomes in the spec. Never raises."""
    try:
        return _classify(m)
    except Exception:
        return Role("leafgroup") if not list(m.children()) else Role("container")


def _classify(m: Any) -> Role:
    torch = _torch()
    nn = torch.nn
    if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.Identity)):
        return Role("pass")
    if isinstance(m, nn.Embedding):
        return Role("block", "embedding", BlockParams(vocab=m.num_embeddings, d=m.embedding_dim))
    p = _attention_params(m)
    if p is not None:
        return Role("block", "attention", p)
    p = _mlp_params(m)
    if p is not None:
        return Role("block", "mlp", p)
    dims = _linear_dims(m)
    if dims:
        return Role("block", "linear", BlockParams(in_=dims[0], out=dims[1], bias=dims[2]))
    cname = type(m).__name__
    rms = getattr(nn, "RMSNorm", None)
    if (rms is not None and isinstance(m, rms)) or "rmsnorm" in cname.lower():
        w = getattr(m, "weight", None)
        return Role("block", "norm", BlockParams(variant="rmsnorm", d=int(w.shape[-1]) if w is not None else None))
    if isinstance(m, nn.LayerNorm):
        return Role("block", "norm", BlockParams(variant="layernorm", d=int(m.normalized_shape[-1]), bias=getattr(m, "bias", None) is not None))
    if "layernorm" in cname.lower() and not list(m.children()):
        w = getattr(m, "weight", None)
        return Role("block", "norm", BlockParams(variant="layernorm", d=int(w.shape[-1]) if w is not None else None, bias=getattr(m, "bias", None) is not None))
    v = _activation_variant(m)
    if v:
        return Role("block", "activation", BlockParams(variant=v))
    if isinstance(m, (nn.Softmax, nn.LogSoftmax)):
        return Role("block", "softmax")
    if isinstance(m, (nn.CrossEntropyLoss, nn.NLLLoss)):
        return Role("block", "loss")
    if list(m.children()):
        return Role("container")
    return Role("leafgroup")


# ── Intermediate graph ───────────────────────────────────────────────────────


@dataclass
class _Node:
    nid: int
    op: str  # input | param | module | func | output
    name: str  # input name, param path, module path, or canonical op name
    srcs: List[int] = field(default_factory=list)
    chain: Tuple[str, ...] = ()  # enclosing container paths, outer -> inner
    module: Any = None
    call: int = 0
    value: Any = None  # first output tensor (or the parameter for op=param)
    aux: Dict[str, Any] = field(default_factory=dict)
    kw: Dict[str, Any] = field(default_factory=dict)


_NON_TENSOR_ATTRS = {"shape", "dtype", "device", "ndim", "is_cuda", "requires_grad", "layout"}
_NON_TENSOR_OPS = {"size", "dim", "numel", "nelement", "item", "tolist", "stride", "data_ptr"}


def _canon(name: str) -> str:
    n = name.strip("_")
    if n in ("iadd", "radd"):
        return "add"
    return n


def _flatten(obj: Any, out: List[Any]) -> List[Any]:
    torch = _torch()
    if isinstance(obj, torch.Tensor):
        out.append(obj)
    elif isinstance(obj, (list, tuple)):
        for o in obj:
            _flatten(o, out)
    elif isinstance(obj, dict):
        for o in obj.values():
            _flatten(o, out)
    elif hasattr(obj, "to_tuple") and callable(obj.to_tuple):  # HF ModelOutput
        _flatten(obj.to_tuple(), out)
    return out


@dataclass
class Inputs:
    """Positional and keyword arguments for ``model(*args, **kwargs)``."""

    args: tuple = ()
    kwargs: Dict[str, Any] = field(default_factory=dict)


def _normalize_input(example_input: Any) -> Tuple[tuple, dict]:
    if isinstance(example_input, Inputs):
        return tuple(example_input.args), dict(example_input.kwargs)
    if example_input is None:
        return (), {}
    if isinstance(example_input, dict):
        return (), dict(example_input)
    if isinstance(example_input, (list, tuple)):
        return tuple(example_input), {}
    return (example_input,), {}


def _bind_names(model: Any, args: tuple, kwargs: dict) -> Dict[str, Any]:
    try:
        bound = inspect.signature(model.forward).bind_partial(*args, **kwargs)
        return dict(bound.arguments)
    except Exception:
        named = {f"input{i}": a for i, a in enumerate(args)}
        named.update(kwargs)
        return named


def _container_paths(model: Any) -> Dict[str, Any]:
    torch = _torch()
    skip = (torch.nn.ModuleList, torch.nn.ModuleDict, torch.nn.ParameterList, torch.nn.ParameterDict)
    return {p: m for p, m in model.named_modules() if p and not isinstance(m, skip)}


# Strategy 1: torch.fx ---------------------------------------------------------


def _ir_fx(model: Any, args: tuple, kwargs: dict) -> Tuple[List[_Node], Optional[int]]:
    torch = _torch()
    import torch.fx as fx

    class Tracer(fx.Tracer):
        def is_leaf_module(self, m: Any, qualname: str) -> bool:  # noqa: D401
            return classify_module(m).role != "container"

    provided = _bind_names(model, args, kwargs)
    concrete: Dict[str, Any] = {}
    params = inspect.signature(model.forward).parameters
    for pname, p in params.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if pname not in provided and p.default is not inspect.Parameter.empty:
            concrete[pname] = p.default
    tracer = Tracer()
    graph = tracer.trace(model, concrete_args=concrete or None)
    gm = fx.GraphModule(tracer.root, graph)

    env: Dict[Any, Any] = {}
    if args or kwargs:
        values = []
        for node in graph.nodes:
            if node.op != "placeholder":
                continue
            key = str(node.target).lstrip("*")
            base = key.rsplit("_", 1)[0] if key not in provided and key not in concrete else key
            if key in provided:
                values.append(provided[key])
            elif base in concrete:
                values.append(concrete[base])
            elif key in concrete:
                values.append(concrete[key])
            elif node.args:
                values.append(node.args[0])
            else:
                values.append(None)

        class Recorder(fx.Interpreter):
            def run_node(self, n: Any) -> Any:
                out = super().run_node(n)
                env[n] = out
                return out

        with torch.no_grad():
            Recorder(gm).run(*values)

    nodes: List[_Node] = []
    index: Dict[Any, int] = {}
    calls: Dict[str, int] = {}

    def srcs_of(n: Any) -> List[int]:
        found: List[int] = []
        fx.node.map_arg((n.args, n.kwargs), lambda a: found.append(index[a]) if a in index else None)
        return found

    def chain_of(n: Any, exclude: Optional[str] = None) -> Tuple[str, ...]:
        stack = n.meta.get("nn_module_stack") or {}
        paths = []
        for k, v in stack.items():
            path = v[0] if isinstance(v, tuple) and isinstance(v[0], str) else k
            if path and path != exclude:
                paths.append(path)
        return tuple(paths)

    def first_tensor(v: Any) -> Any:
        ts = _flatten(v, [])
        return ts[0] if ts else None

    for n in graph.nodes:
        nid = len(nodes)
        val = env.get(n)
        if n.op == "placeholder":
            if str(n.target).rstrip("_1234567890") in concrete and str(n.target) not in provided:
                index[n] = nid
                nodes.append(_Node(nid, "func", "concrete"))
                continue
            node = _Node(nid, "input", str(n.target).lstrip("*"), value=first_tensor(val))
        elif n.op == "get_attr":
            obj = gm
            for part in str(n.target).split("."):
                obj = getattr(obj, part)
            node = _Node(nid, "param", str(n.target), value=obj)
        elif n.op == "call_module":
            path = str(n.target)
            calls[path] = calls.get(path, -1) + 1
            sub = model.get_submodule(path)
            node = _Node(nid, "module", path, srcs_of(n), chain_of(n, exclude=path), module=sub, call=calls[path], value=first_tensor(val))
            outs = _flatten(val, []) if val is not None else []
            if len(outs) > 1:
                node.aux["weights"] = outs[1]
        elif n.op in ("call_function", "call_method"):
            target = n.target
            name = target if isinstance(target, str) else getattr(target, "__name__", str(target))
            name = _canon(name)
            if name == "getattr" and len(n.args) > 1 and n.args[1] in _NON_TENSOR_ATTRS:
                name = "size"
            kw = {}
            if "is_causal" in n.kwargs:
                kw["is_causal"] = n.kwargs["is_causal"]
            node = _Node(nid, "func", name, srcs_of(n), chain_of(n), value=first_tensor(val), kw=kw)
        elif n.op == "output":
            node = _Node(nid, "output", "output", srcs_of(n))
        else:  # pragma: no cover
            continue
        index[n] = nid
        nodes.append(node)
    return nodes, None


# Strategy 2: hooks + TorchFunctionMode ---------------------------------------


class _Recorder:
    def __init__(self, model: Any) -> None:
        self.model = model
        self.nodes: List[_Node] = []
        self.ids: Dict[int, int] = {}  # id(tensor) -> nid
        self.alive: List[Any] = []  # keep tensors alive so ids are not reused
        self.params = {id(p): n for n, p in list(model.named_parameters(remove_duplicate=False)) + list(model.named_buffers(remove_duplicate=False))}
        self.param_nodes: Dict[int, int] = {}
        self.paths = {id(m): p for p, m in model.named_modules(remove_duplicate=False)}
        self.roles: Dict[int, Role] = {}
        self.chain: List[str] = []
        self.flags: List[Any] = []
        self.leaf_depth = 0
        self.paused = False
        self.calls: Dict[str, int] = {}
        self.functional = True

    def role(self, m: Any) -> Role:
        r = self.roles.get(id(m))
        if r is None:
            r = self.roles[id(m)] = classify_module(m)
        return r

    def src_of(self, t: Any) -> Optional[int]:
        nid = self.ids.get(id(t))
        if nid is not None:
            return nid
        name = self.params.get(id(t))
        if name is not None:
            if id(t) not in self.param_nodes:
                pid = len(self.nodes)
                self.nodes.append(_Node(pid, "param", name, value=t))
                self.param_nodes[id(t)] = pid
            return self.param_nodes[id(t)]
        return None

    def register(self, out: Any, nid: int) -> List[Any]:
        ts = _flatten(out, [])
        for t in ts:
            self.ids[id(t)] = nid
            self.alive.append(t)
        return ts

    def add_input(self, name: str, value: Any) -> None:
        for t in _flatten(value, []):
            nid = len(self.nodes)
            self.nodes.append(_Node(nid, "input", name, value=t))
            self.register(t, nid)

    # module hooks
    def pre(self, module: Any, args: tuple, kwargs: Optional[dict] = None) -> None:
        if self.paused:
            self.flags.append(None)
            return
        self.paused = True
        try:
            if self.leaf_depth:
                self.flags.append(None)
                return
            path = self.paths.get(id(module), type(module).__name__)
            if self.role(module).role == "container":
                self.flags.append("c")
                if path:
                    self.chain.append(path)
                return
            ts = _flatten((args, kwargs or {}), [])
            srcs = [s for s in (self.src_of(t) for t in ts) if s is not None]
            kw = {}
            if kwargs and isinstance(kwargs.get("is_causal"), bool):
                kw["is_causal"] = kwargs["is_causal"]
            self.flags.append(("leaf", path, srcs, kw))
            self.leaf_depth += 1
        finally:
            self.paused = False

    def post(self, module: Any, args: tuple, *rest: Any) -> None:
        output = rest[-1]
        flag = self.flags.pop() if self.flags else None
        if flag is None:
            return
        if flag == "c":
            path = self.paths.get(id(module), "")
            if path and self.chain and self.chain[-1] == path:
                self.chain.pop()
            return
        self.leaf_depth -= 1
        self.paused = True
        try:
            _, path, srcs, kw = flag
            self.calls[path] = self.calls.get(path, -1) + 1
            nid = len(self.nodes)
            node = _Node(nid, "module", path, srcs, tuple(self.chain), module=module, call=self.calls[path], kw=kw)
            ts = self.register(output, nid)
            node.value = ts[0] if ts else None
            if len(ts) > 1:
                node.aux["weights"] = ts[1]
            self.nodes.append(node)
        finally:
            self.paused = False

    # functional ops
    def on_func(self, func: Any, args: tuple, kwargs: dict, out: Any) -> None:
        ts_out = _flatten(out, [])
        if not ts_out:
            return
        name = _canon(getattr(func, "__name__", str(func)))
        ins = _flatten((args, kwargs), [])
        srcs = []
        for t in ins:
            s = self.src_of(t)
            if s is not None and s not in srcs:
                srcs.append(s)
        kw = {}
        if isinstance(kwargs.get("is_causal"), bool):
            kw["is_causal"] = kwargs["is_causal"]
        nid = len(self.nodes)
        node = _Node(nid, "func", name, srcs, tuple(self.chain), kw=kw)
        self.register(out, nid)
        node.value = ts_out[0]
        self.nodes.append(node)


def _ir_hooks(model: Any, args: tuple, kwargs: dict) -> Tuple[List[_Node], Any]:
    torch = _torch()
    rec = _Recorder(model)
    for name, value in _bind_names(model, args, kwargs).items():
        rec.add_input(name, value)

    handles = []
    try:
        for m in model.modules():
            try:
                handles.append(m.register_forward_pre_hook(rec.pre, with_kwargs=True))
                handles.append(m.register_forward_hook(rec.post, with_kwargs=True))
            except TypeError:  # torch < 2.0
                handles.append(m.register_forward_pre_hook(lambda mod, a: rec.pre(mod, a)))
                handles.append(m.register_forward_hook(rec.post))
        mode_cls = None
        try:
            from torch.overrides import TorchFunctionMode

            class Mode(TorchFunctionMode):
                def __torch_function__(self, func, types, a=(), kw=None):  # type: ignore[override]
                    kw = kw or {}
                    out = func(*a, **kw)
                    if not rec.paused and not rec.leaf_depth:
                        rec.paused = True
                        try:
                            rec.on_func(func, a, kw, out)
                        finally:
                            rec.paused = False
                    return out

            mode_cls = Mode
        except ImportError:  # pragma: no cover
            rec.functional = False
        with torch.no_grad():
            if mode_cls is not None:
                with mode_cls():
                    out = model(*args, **kwargs)
            else:  # pragma: no cover
                out = model(*args, **kwargs)
    finally:
        for h in handles:
            h.remove()
    srcs = []
    rec.paused = True
    for t in _flatten(out, []):
        s = rec.ids.get(id(t))
        if s is not None and s not in srcs:
            srcs.append(s)
    rec.nodes.append(_Node(len(rec.nodes), "output", "output", srcs))
    return rec.nodes, out


# Strategy 3: module tree ------------------------------------------------------


def _ir_tree(model: Any) -> List[_Node]:
    containers = _container_paths(model)
    nodes: List[_Node] = [_Node(0, "input", "input")]
    prev = 0
    skip_under: List[str] = []
    for path, m in model.named_modules():
        if any(path.startswith(s + ".") for s in skip_under):
            continue
        role = classify_module(m)
        if role.role in ("container", "pass"):
            continue
        skip_under.append(path)
        parts = path.split(".")
        chain = tuple(p for p in (".".join(parts[:i]) for i in range(1, len(parts))) if p in containers and classify_module(containers[p]).role == "container")
        nid = len(nodes)
        nodes.append(_Node(nid, "module", path, [prev], chain, module=m))
        prev = nid
    nodes.append(_Node(len(nodes), "output", "output", [prev]))
    return nodes


# ── Mapping the graph onto spec blocks ──────────────────────────────────────

_PREFIX = {
    "tokens": "t",
    "input": "i",
    "embedding": "e",
    "posenc": "p",
    "attention": "a",
    "mlp": "m",
    "norm": "n",
    "add": "r",
    "linear": "l",
    "activation": "f",
    "unembed": "u",
    "softmax": "s",
    "loss": "x",
    "group": "g",
}


@dataclass
class Analysis:
    """Everything ``from_module`` learned. ``spec`` is the public result."""

    spec: ModelSpec
    strategy: str
    values: Dict[str, Any] = field(default_factory=dict)  # block id -> output tensor
    aux: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # block id -> port -> tensor
    warnings: List[str] = field(default_factory=list)
    output: Any = None  # the model's return value (hooks strategy only)


class _Mapper:
    def __init__(self, model: Any, nodes: List[_Node], has_input: bool) -> None:
        self.model = model
        self.nodes = nodes
        self.has_input = has_input
        self.blocks: List[Block] = []
        self.groups: List[Block] = []
        self.group_ids: Dict[str, str] = {}
        self.counters: Dict[str, int] = {}
        self.bindings: Dict[str, str] = {}
        self.values: Dict[str, Any] = {}
        self.aux: Dict[str, Dict[str, Any]] = {}
        self.src: Dict[int, List[str]] = {}
        self.pos: Dict[int, bool] = {}
        self.porigin: Dict[int, Tuple[str, Any]] = {}
        self.emb_weights: Dict[int, str] = {}
        self.pending_pos: Dict[str, bool] = {}
        self.fcount: Dict[Tuple[str, str], int] = {}
        self.output: Optional[str] = None
        self.containers = _container_paths(model)

    def new_id(self, kind: str) -> str:
        p = _PREFIX.get(kind, "b")
        n = self.counters.get(p, 0)
        self.counters[p] = n + 1
        return f"{p}{n}"

    def group_for(self, chain: Tuple[str, ...]) -> Optional[str]:
        parent: Optional[str] = None
        for path in chain:
            if path not in self.group_ids:
                gid = self.new_id("group")
                sub = self.containers.get(path)
                cls = type(sub).__name__ if sub is not None else "Module"
                self.groups.append(Block(gid, "group", BlockParams(), [], label=f"{path} ({cls})", group=parent))
                self.group_ids[path] = gid
            parent = self.group_ids[path]
        return parent

    def add_block(self, node: _Node, kind: str, params: BlockParams, inputs: List[str], label: str, binding: str) -> str:
        bid = self.new_id(kind)
        self.blocks.append(Block(bid, kind, params, _dedup(inputs), label=label, group=self.group_for(node.chain)))
        self.bindings[bid] = binding
        if node.value is not None:
            self.values[bid] = node.value
        if node.aux:
            self.aux[bid] = dict(node.aux)
        self.src[node.nid] = [bid]
        return bid

    def func_binding(self, node: _Node, kind: str) -> str:
        container = node.chain[-1] if node.chain else ""
        k = (container, kind)
        n = self.fcount.get(k, 0)
        self.fcount[k] = n + 1
        return f"f:{container}:{kind}#{n}"

    def passthrough(self, node: _Node) -> None:
        out: List[str] = []
        for s in node.srcs:
            out.extend(self.src.get(s, []))
        self.src[node.nid] = _dedup(out)
        has_blocks = bool(self.src[node.nid])
        self.pos[node.nid] = not has_blocks and any(self.pos.get(s) for s in node.srcs)
        origins = [self.porigin[s] for s in node.srcs if s in self.porigin]
        if not has_blocks and len(origins) == 1:
            self.porigin[node.nid] = origins[0]

    def block(self, bid: str) -> Block:
        for b in self.blocks:
            if b.id == bid:
                return b
        raise KeyError(bid)

    def run(self) -> None:
        for node in self.nodes:
            getattr(self, "on_" + node.op)(node)

    # handlers
    def on_input(self, node: _Node) -> None:
        torch = _torch()
        v = node.value
        if v is not None:
            is_int = not (v.is_floating_point() or v.is_complex())
        else:
            is_int = any(isinstance(m, torch.nn.Embedding) for m in self.model.modules())
        if is_int:
            params = BlockParams()
            self.add_block(node, "tokens", params, [], node.name, f"i:{node.name}")
        else:
            params = BlockParams(in_=int(v.shape[-1]) if v is not None and v.dim() > 0 else None)
            self.add_block(node, "input", params, [], node.name, f"i:{node.name}")

    def on_param(self, node: _Node) -> None:
        self.src[node.nid] = []
        self.porigin[node.nid] = (node.name, node.value)

    def on_output(self, node: _Node) -> None:
        for s in node.srcs:
            blocks = self.src.get(s, [])
            if blocks:
                self.output = blocks[-1]
                return

    def on_module(self, node: _Node) -> None:
        role = classify_module(node.module)
        if role.role == "pass":
            self.passthrough(node)
            return
        inputs: List[str] = []
        for s in node.srcs:
            inputs.extend(self.src.get(s, []))
        binding = f"m:{node.name}#{node.call}"
        label = node.name + (f" (call {node.call + 1})" if node.call else "")
        if role.role in ("leafgroup", "container"):
            self.add_block(node, "group", BlockParams(), inputs, type(node.module).__name__, binding)
            return
        kind, params = role.kind, BlockParams.from_dict(role.params.to_dict())
        if kind == "embedding" and node.srcs and all(self.pos.get(s) for s in node.srcs if not self.src.get(s)) and not inputs and any(self.pos.get(s) for s in node.srcs):
            bid = self.add_block(node, "posenc", BlockParams(variant="learned", maxLen=params.vocab, d=params.d), [], label, binding)
            self.pending_pos[bid] = True
            return
        if kind == "linear":
            w = getattr(node.module, "weight", None)
            tied = self.emb_weights.get(id(w)) if w is not None else None
            if tied:
                kind, params = "unembed", BlockParams(vocab=params.out, d=params.in_, tiedTo=tied)
        if kind == "attention" and params.causal is None and isinstance(node.kw.get("is_causal"), bool):
            params.causal = node.kw["is_causal"]
        bid = self.add_block(node, kind, params, inputs, label, binding)
        if kind == "embedding":
            self.emb_weights[id(node.module.weight)] = bid

    def on_func(self, node: _Node) -> None:
        name = node.name
        if name in _NON_TENSOR_OPS or name.startswith("assert") or name == "concrete":
            self.src[node.nid] = []
            return
        if name == "arange":
            self.src[node.nid] = []
            self.pos[node.nid] = True
            return
        operands = [self.src.get(s, []) for s in node.srcs]
        inputs = _dedup([b for o in operands for b in o])
        if not inputs:
            self.passthrough(node)
            return
        if name == "add":
            distinct = _dedup_lists([o for o in operands if o])
            pending = [o for o in distinct if len(o) == 1 and self.pending_pos.get(o[0])]
            if len(distinct) >= 2 and pending:
                pid = pending[0][0]
                others = _dedup([b for o in distinct if o is not pending[0] for b in o])
                blk = self.block(pid)
                blk.inputs = others
                self.blocks.remove(blk)
                self.blocks.append(blk)  # keep topological order
                del self.pending_pos[pid]
                if pid in self.values:
                    self.aux.setdefault(pid, {})["table"] = self.values[pid]
                if node.value is not None:
                    self.values[pid] = node.value
                self.src[node.nid] = [pid]
                return
            if len(distinct) >= 2:
                self.add_block(node, "add", BlockParams(), inputs, "residual add", self.func_binding(node, "add"))
                return
            origin = next((self.porigin[s] for s in node.srcs if s in self.porigin), None)
            if origin is not None:
                pname, obj = origin
                last = pname.split(".")[-1].lower()
                if "pos" in last or last in ("pe", "wpe"):
                    torch = _torch()
                    variant = "learned" if isinstance(obj, torch.nn.Parameter) else "sinusoidal"
                    shape = list(getattr(obj, "shape", []))
                    params = BlockParams(variant=variant, d=int(shape[-1]) if shape else None, maxLen=int(shape[-2]) if len(shape) >= 2 else None)
                    self.add_block(node, "posenc", params, inputs, pname, self.func_binding(node, "posenc"))
                    return
            self.passthrough(node)
            return
        if name in _ACTIVATIONS:
            self.add_block(node, "activation", BlockParams(variant=_ACTIVATIONS[name]), inputs, name, self.func_binding(node, "activation"))
            return
        if name in ("softmax", "log_softmax"):
            self.add_block(node, "softmax", BlockParams(), inputs, name, self.func_binding(node, "softmax"))
            return
        if name in ("cross_entropy", "nll_loss", "mse_loss", "binary_cross_entropy_with_logits"):
            self.add_block(node, "loss", BlockParams(), inputs, name, self.func_binding(node, "loss"))
            return
        if name in ("layer_norm", "rms_norm"):
            d = int(node.value.shape[-1]) if node.value is not None and node.value.dim() else None
            variant = "layernorm" if name == "layer_norm" else "rmsnorm"
            self.add_block(node, "norm", BlockParams(variant=variant, d=d), inputs, name, self.func_binding(node, "norm"))
            return
        if name in ("linear", "embedding"):
            w = next((self.porigin[s][1] for s in node.srcs if s in self.porigin), None)
            shape = list(getattr(w, "shape", []))
            if name == "embedding":
                params = BlockParams(vocab=shape[0] if shape else None, d=shape[1] if len(shape) > 1 else None)
                bid = self.add_block(node, "embedding", params, inputs, name, self.func_binding(node, "embedding"))
                if w is not None:
                    self.emb_weights[id(w)] = bid
                return
            tied = self.emb_weights.get(id(w)) if w is not None else None
            if tied and len(shape) == 2:
                params = BlockParams(vocab=shape[0], d=shape[1], tiedTo=tied)
                self.add_block(node, "unembed", params, inputs, "unembed (tied)", self.func_binding(node, "unembed"))
            else:
                params = BlockParams(in_=shape[1] if len(shape) == 2 else None, out=shape[0] if shape else None)
                self.add_block(node, "linear", params, inputs, name, self.func_binding(node, "linear"))
            return
        if name == "scaled_dot_product_attention":
            q = None
            heads = None
            if node.value is not None and node.value.dim() == 4:
                heads = int(node.value.shape[1])
                q = int(node.value.shape[1] * node.value.shape[3])
            params = BlockParams(heads=heads, d=q, causal=node.kw.get("is_causal"))
            self.add_block(node, "attention", params, inputs, "sdpa", self.func_binding(node, "attention"))
            return
        self.passthrough(node)

    def finish(self) -> Tuple[List[Block], str]:
        # Drop inputs nobody reads (e.g. a `targets` argument that was None).
        consumed = {i for b in self.blocks for i in b.inputs}
        keep = [b for b in self.blocks if b.kind not in ("tokens", "input") or b.id in consumed or b.id == self.output]
        dropped = {b.id for b in self.blocks} - {b.id for b in keep}
        for bid in dropped:
            self.bindings.pop(bid, None)
            self.values.pop(bid, None)
        self.blocks = keep
        vocabs = {b.params.vocab for b in self.blocks if b.kind == "embedding" and b.params.vocab}
        for b in self.blocks:
            if b.kind == "tokens":
                v = next((c.params.vocab for c in self.blocks if c.kind == "embedding" and b.id in c.inputs), None)
                b.params.vocab = v
        output = self.output or (self.blocks[-1].id if self.blocks else "")
        out_block = next((b for b in self.blocks if b.id == output), None)
        if out_block is not None and out_block.kind == "linear" and out_block.params.out in vocabs:
            out_block.kind = "unembed"
            out_block.params = BlockParams(vocab=out_block.params.out, d=out_block.params.in_, bias=out_block.params.bias)
        used_groups = {b.group for b in self.blocks if b.group}
        changed = True
        while changed:
            changed = False
            for g in self.groups:
                if g.id in used_groups and g.group and g.group not in used_groups:
                    used_groups.add(g.group)
                    changed = True
        groups = [g for g in self.groups if g.id in used_groups]
        return groups + self.blocks, output


def _dedup(xs: Sequence[str]) -> List[str]:
    seen: Dict[str, None] = {}
    for x in xs:
        seen.setdefault(x, None)
    return list(seen)


def _dedup_lists(xs: Sequence[List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    for x in xs:
        if x not in out:
            out.append(x)
    return out


def _origin_source(model: Any) -> str:
    cls = type(model)
    return f"{cls.__module__}:{cls.__qualname__}"


class _EvalNoGrad:
    def __init__(self, model: Any) -> None:
        self.model = model

    def __enter__(self) -> None:
        self.training = self.model.training
        self.model.eval()

    def __exit__(self, *exc: Any) -> None:
        self.model.train(self.training)


def analyze(model: Any, example_input: Any = None, *, strategy: str = "auto", name: Optional[str] = None) -> Analysis:
    """Like ``from_module`` but also returns the activations it saw."""
    torch = _torch()
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"expected an nn.Module, got {type(model).__name__}")
    if strategy not in ("auto", "fx", "hooks", "tree"):
        raise ValueError("strategy must be auto, fx, hooks or tree")
    args, kwargs = _normalize_input(example_input)
    has_input = example_input is not None
    notes: List[str] = []
    order = ["fx", "hooks", "tree"] if strategy == "auto" else [strategy]
    nodes: Optional[List[_Node]] = None
    used = ""
    out = None
    for s in order:
        try:
            with _EvalNoGrad(model):
                if s == "fx":
                    nodes, _ = _ir_fx(model, args, kwargs)
                elif s == "hooks":
                    if not has_input:
                        notes.append("hooks: skipped, needs example_input")
                        continue
                    nodes, out = _ir_hooks(model, args, kwargs)
                else:
                    nodes = _ir_tree(model)
                    notes.append("tree: edges follow module registration order and may not match forward()")
            used = s
            break
        except Exception as e:  # never fail: fall through to the next strategy
            notes.append(f"{s}: {type(e).__name__}: {str(e).splitlines()[0] if str(e) else ''}".rstrip(": "))
            nodes = None
    if nodes is None:  # tree cannot really fail, but stay total
        nodes, used = [_Node(0, "input", "input"), _Node(1, "output", "output", [0])], "tree"
    mapper = _Mapper(model, nodes, has_input)
    mapper.run()
    blocks, output = mapper.finish()
    if not blocks:
        blocks = [Block("g0", "group", BlockParams(), [], label=type(model).__name__)]
        output = "g0"
    spec = ModelSpec(
        name=name or type(model).__name__,
        blocks=blocks,
        output=output,
        origin=Origin("torch", _origin_source(model)),
    )
    spec.bindings = dict(mapper.bindings)
    for n in notes:
        if used == "tree" or not n.startswith(("fx:", "hooks:")) or n.startswith("tree"):
            warnings.warn(f"papertoanything.from_module: {n}", stacklevel=3)
    return Analysis(spec, used, dict(mapper.values), dict(mapper.aux), notes, out)


def from_module(model: Any, example_input: Any = None, *, strategy: str = "auto", name: Optional[str] = None) -> ModelSpec:
    """Map an ``nn.Module`` to a ModelSpec. Never fails on an unknown model.

    ``example_input`` is a tensor, a tuple of positional arguments, or a dict
    of keyword arguments. With it, shapes are known and the ``hooks``
    fallback is available; without it, only ``fx`` and ``tree`` are tried.
    The model is put in eval mode for the pass and restored afterwards, and
    no gradients are recorded, so the model's state is unchanged.
    """
    return analyze(model, example_input, strategy=strategy, name=name).spec


# ── Capture ──────────────────────────────────────────────────────────────────


def capture(
    model: Any,
    example_input: Any,
    max_elems: int = DEFAULT_MAX_ELEMS,
    *,
    spec: Optional[ModelSpec] = None,
    step: int = 0,
    loss: Optional[float] = None,
    max_frame_elems: int = DEFAULT_MAX_FRAME_ELEMS,
) -> TraceFrame:
    """Run one forward pass and return a TraceFrame of every block's output.

    Keys are ``"<blockId>:out"`` (plus ``":weights"`` when an attention
    module returned its pattern, ``":table"`` for a learned positional
    table). Block ids match ``spec`` (default: ``from_module`` on the same
    model and input). Tensors over ``max_elems`` values are clipped to their
    leading corner and marked ``clipped`` with the ``dataShape`` actually
    sent; once ``max_frame_elems`` values have been sent, remaining tensors
    carry their shape and no data (``clipped`` true, ``dataShape`` zeros).
    Blocks with no tensor are listed in ``uncaptured``.
    """
    an = analyze(model, example_input, strategy="hooks")
    target = spec if spec is not None else an.spec
    values: Dict[str, Any] = {}
    aux: Dict[str, Dict[str, Any]] = {}
    if target is an.spec:
        values, aux = an.values, an.aux
    else:
        by_binding = {v: k for k, v in an.spec.bindings.items()}
        if target.bindings:
            for bid, key in target.bindings.items():
                src = by_binding.get(key)
                if src is None:
                    continue
                if an.spec.block(src).kind != target.block(bid).kind:
                    continue
                if src in an.values:
                    values[bid] = an.values[src]
                if src in an.aux:
                    aux[bid] = an.aux[src]
        else:
            ours = [(b.id, b.kind) for b in an.spec.blocks]
            theirs = [(b.id, b.kind) for b in target.blocks]
            if ours != theirs:
                raise ValueError("spec does not match this model; build it with from_module(model, example_input)")
            values, aux = an.values, an.aux

    tensors: List[FrameTensor] = []
    budget = max_frame_elems
    captured = set()

    def put(key: str, t: Any) -> None:
        nonlocal budget
        torch = _torch()
        if not isinstance(t, torch.Tensor) or t.is_complex():
            return
        if budget <= 0 and t.dim() > 0:
            shape = [int(s) for s in t.shape]
            tensors.append(FrameTensor(key, shape, "", clipped=True, dataShape=[0] * len(shape)))
            return
        ft = encode_tensor(key, t, max_elems=max(1, min(max_elems, budget)))
        budget -= math.prod(ft.dataShape) if ft.dataShape is not None else t.numel()
        tensors.append(ft)

    for b in target.blocks:
        if b.id in values:
            put(f"{b.id}:out", values[b.id])
            captured.add(b.id)
        for port, t in aux.get(b.id, {}).items():
            put(f"{b.id}:{port}", t)

    if loss is None:
        loss_block = next((b.id for b in target.blocks if b.kind == "loss" and b.id in values), None)
        if loss_block is not None:
            loss = scalar(values[loss_block])
        else:
            torch = _torch()
            for t in _flatten(an.output, []):
                if isinstance(t, torch.Tensor) and t.dim() == 0 and t.is_floating_point():
                    loss = scalar(t)
                    break
    uncaptured = [b.id for b in target.blocks if b.id not in captured and not (b.kind == "group" and not b.inputs)]
    return TraceFrame(step=step, tensors=tensors, loss=loss, uncaptured=uncaptured or None)


# ── Watching a training run ─────────────────────────────────────────────────


class TraceWatch:
    """Push a TraceFrame to a sink every ``every`` training steps.

    Steps are counted by a forward hook on the model (each forward call in
    training mode is one step) until you call ``step()`` yourself, after
    which only your calls count. Each frame is a fresh ``capture`` on the
    fixed ``example_input`` in eval mode without gradients, so watching does
    not change training. Use as a context manager, or call ``close()``.
    """

    def __init__(self, model: Any, example_input: Any, push: Callable[[TraceFrame], None], *, every: int = 10, spec: Optional[ModelSpec] = None, max_elems: int = DEFAULT_MAX_ELEMS) -> None:
        if every < 1:
            raise ValueError("every must be at least 1")
        self.model = model
        self.example_input = example_input
        self.push = push
        self.every = every
        self.spec = spec
        self.max_elems = max_elems
        self.count = 0
        self.manual = False
        self.last_loss: Optional[float] = None
        self._busy = False
        self._handle = model.register_forward_hook(self._on_forward)

    def _on_forward(self, module: Any, args: Any, output: Any) -> None:
        if self._busy or self.manual or not module.training:
            return
        torch = _torch()
        for t in _flatten(output, []):
            if isinstance(t, torch.Tensor) and t.dim() == 0 and t.is_floating_point():
                self.last_loss = scalar(t)
                break
        self._tick()

    def step(self, loss: Any = None) -> None:
        """Count one training step; optionally report its loss."""
        self.manual = True
        if loss is not None:
            self.last_loss = scalar(loss)
        self._tick()

    def _tick(self) -> None:
        self.count += 1
        if self.count % self.every == 0:
            self.snapshot()

    def snapshot(self) -> TraceFrame:
        """Capture and push a frame now."""
        self._busy = True
        try:
            frame = capture(self.model, self.example_input, self.max_elems, spec=self.spec, step=self.count, loss=self.last_loss)
        finally:
            self._busy = False
        self.push(frame)
        return frame

    def close(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def __enter__(self) -> "TraceWatch":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def watch_trace(model: Any, example_input: Any, push: Callable[[TraceFrame], None], every: int = 10, spec: Optional[ModelSpec] = None) -> TraceWatch:
    return TraceWatch(model, example_input, push, every=every, spec=spec)
