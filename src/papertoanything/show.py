"""``ptoa.show``, ``ptoa.save`` and ``ptoa.load``: the three bridge transports."""

from __future__ import annotations

import json
import os
import webbrowser
from typing import Any, Mapping, Optional, Tuple, Union

from .link import to_url
from .spec import ModelSpec, TraceFrame, dumps

__all__ = ["show", "save", "load", "as_spec"]

#: Fragments longer than this may be cut by some OS URL handlers.
LONG_URL = 32000


def _is_module(obj: Any) -> bool:
    try:
        import torch
    except ImportError:
        return False
    return isinstance(obj, torch.nn.Module)


def as_spec(obj: Any, example_input: Any = None) -> ModelSpec:
    """ModelSpec from a ModelSpec, a dict, a JSON string/path, or an nn.Module."""
    if isinstance(obj, ModelSpec):
        return obj
    if isinstance(obj, Mapping):
        return ModelSpec.from_dict(obj)
    if isinstance(obj, (str, os.PathLike)):
        text = str(obj)
        if os.path.isfile(text):
            with open(text, encoding="utf-8") as fh:
                data = json.load(fh)
            if "spec" in data and "blocks" not in data:
                data = data["spec"]
            return ModelSpec.from_dict(data)
        return ModelSpec.from_json(text)
    if _is_module(obj):
        from .torch import from_module

        return from_module(obj, example_input)
    raise TypeError(f"cannot make a spec from {type(obj).__name__}")


def _open(url: str, open_browser: bool) -> None:
    if open_browser and not os.environ.get("PTA_NO_BROWSER"):
        try:
            webbrowser.open(url)
        except Exception:
            pass


def show(model_or_spec: Any, example_input: Any = None, mode: str = "auto", *, open_browser: bool = True, path: Optional[str] = None, lab_url: Optional[str] = None) -> Any:
    """Show a model or spec in the Lab.

    mode="link"   print and open ``https://lab.papertoanything.com/#s=...``.
                  The spec travels only in the URL fragment, which browsers
                  never send to a server. Returns the URL.
    mode="local"  start a bridge on 127.0.0.1 (random port, random token),
                  push a TraceFrame when ``example_input`` is given, and open
                  the hosted Lab on it.
                  Returns the running ``LocalServer`` (``.push``, ``.watch``,
                  ``.close``).
    mode="file"   write a ``.pta`` file (``path``, default ``<name>.pta``)
                  to drop on the Lab. Returns the path.
    mode="auto"   same as "link".
    """
    from .server import LocalServer

    if mode == "auto":
        mode = "link"
    spec = as_spec(model_or_spec, example_input)
    if mode == "link":
        url = to_url(spec)
        print(url)
        if len(url) > LONG_URL:
            print(f"papertoanything: this link is {len(url)} characters; if it does not open, use mode='file'.")
        _open(url, open_browser)
        return url
    if mode == "local":
        server = LocalServer(spec=spec, lab_url=lab_url).start()
        if example_input is not None and _is_module(model_or_spec):
            from .torch import capture

            server.push(capture(model_or_spec, example_input, spec=spec))
        print(f"papertoanything: serving {spec.name} to the Lab at {server.url}")
        _open(server.url, open_browser)
        return server
    if mode == "file":
        trace = None
        if example_input is not None and _is_module(model_or_spec):
            from .torch import capture

            trace = capture(model_or_spec, example_input, spec=spec)
        out = save(spec, path or f"{spec.name}.pta", trace=trace)
        print(f"papertoanything: wrote {out}; drop it on the Lab")
        return out
    raise ValueError("mode must be auto, link, local or file")


def save(model_or_spec: Any, path: str, example_input: Any = None, trace: Optional[TraceFrame] = None) -> str:
    """Write a ``.pta`` file::

        {"protocol":"pta-file","v":1,"producer":{...},"spec":{...},"trace":{TraceFrame}?}

    Tensors in ``trace`` are base64 little-endian float32, as on /events.
    """
    from . import __version__

    spec = as_spec(model_or_spec, example_input)
    if trace is None and example_input is not None and _is_module(model_or_spec):
        from .torch import capture

        trace = capture(model_or_spec, example_input, spec=spec)
    doc = {"protocol": "pta-file", "v": 1, "producer": {"name": "papertoanything", "version": __version__}, "spec": spec.to_dict()}
    if trace is not None:
        doc["trace"] = trace.to_dict()
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dumps(doc))
    return path


def load(path: str) -> Tuple[ModelSpec, Optional[TraceFrame]]:
    """Read a ``.pta`` file written by ``save`` (or a bare spec JSON)."""
    with open(path, encoding="utf-8") as fh:
        first = fh.readline()
        rest = fh.read()
    try:
        doc = json.loads(first + rest)
    except json.JSONDecodeError:
        head = json.loads(first)
        if head.get("protocol") == "pta-run":
            return ModelSpec.from_dict(head["spec"]), None
        raise
    if "blocks" in doc:
        return ModelSpec.from_dict(doc), None
    trace = doc.get("trace")
    return ModelSpec.from_dict(doc["spec"]), None if trace is None else TraceFrame.from_dict(trace)
