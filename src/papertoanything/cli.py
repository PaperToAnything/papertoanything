"""``pta``: the command-line half of papertoanything."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from typing import Any, List, Optional

from . import __version__

DESCRIPTION = """\
See a PyTorch model in the Paper To Anything Lab. Nothing is uploaded:
links carry the model in the URL fragment (browsers never send it to a
server), `serve` runs on 127.0.0.1 only, and `.pta` files stay on disk.
"""

EPILOG = """\
examples:
  pta inspect model.py:GPT --input-shape 1,64 --vocab 50304
  pta inspect model.py:GPT --link            design view link for any size
  pta link spec.json                         spec file -> lab link
  pta decode 'https://lab.papertoanything.com/#s=...'
  pta save model.py:TinyNet -o tiny.pta --input-shape 1,2 --dtype float
  pta serve --spec spec.json                 local bridge on 127.0.0.1

SOURCE is a spec JSON file, a .pta file, or path/to/file.py:Name (or
package.module:Name) where Name is an nn.Module class (built with no
arguments), a function returning a module, or a module instance.
"""


def _parse_shape(text: str) -> List[int]:
    try:
        shape = [int(s) for s in text.replace("x", ",").split(",") if s.strip()]
    except ValueError:
        raise argparse.ArgumentTypeError(f"bad shape {text!r}; use e.g. 1,64") from None
    if not shape or any(s < 0 for s in shape):
        raise argparse.ArgumentTypeError(f"bad shape {text!r}")
    return shape


def _split_ref(ref: str) -> Optional[tuple]:
    """'a/b/model.py:GPT' or 'pkg.mod:GPT' -> (module part, attr). Windows drive
    letters ('C:\\x\\m.py:GPT') are handled by splitting on the last colon."""
    if ":" not in ref:
        return None
    left, _, attr = ref.rpartition(":")
    if not attr.isidentifier() or not left:
        return None
    return left, attr


def _load_object(ref: str) -> Any:
    left, attr = _split_ref(ref)  # type: ignore[misc]
    if left.endswith(".py") or os.path.sep in left or "/" in left:
        path = os.path.abspath(left)
        if not os.path.isfile(path):
            raise SystemExit(f"pta: no such file: {left}")
        name = os.path.splitext(os.path.basename(path))[0]
        sys.path.insert(0, os.path.dirname(path))
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise SystemExit(f"pta: cannot import {left}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(left)
    if not hasattr(mod, attr):
        raise SystemExit(f"pta: {left} has no attribute {attr}")
    return getattr(mod, attr)


def _instantiate(obj: Any, ref: str) -> Any:
    try:
        import torch
    except ImportError:
        raise SystemExit("pta: inspecting a model needs PyTorch: pip install 'papertoanything[torch]'") from None
    if isinstance(obj, torch.nn.Module):
        return obj
    if callable(obj):
        try:
            model = obj()
        except TypeError as e:
            raise SystemExit(
                f"pta: could not build {ref} with no arguments ({e}).\n"
                f"     Add a function that builds it, e.g. `def make(): return {getattr(obj, '__name__', 'Model')}(...)`, and pass file.py:make."
            ) from None
        if isinstance(model, torch.nn.Module):
            return model
        raise SystemExit(f"pta: {ref} returned {type(model).__name__}, not an nn.Module")
    raise SystemExit(f"pta: {ref} is not a module, class or function")


def _example_input(args: argparse.Namespace, model: Any) -> Any:
    if not getattr(args, "input_shape", None):
        for attr in ("example_input", "example_input_array"):
            v = getattr(model, attr, None)
            if v is not None:
                return v
        return None
    import torch

    shape = args.input_shape
    if args.dtype == "long":
        vocab = args.vocab
        if vocab is None:
            emb = next((m for m in model.modules() if isinstance(m, torch.nn.Embedding)), None)
            vocab = emb.num_embeddings if emb is not None else 2
        g = torch.Generator().manual_seed(0)
        return torch.randint(0, vocab, shape, generator=g)
    g = torch.Generator().manual_seed(0)
    return torch.randn(*shape, generator=g)


def _source(args: argparse.Namespace) -> tuple:
    """(spec, model or None, example_input or None)."""
    from .show import as_spec, load

    src = args.source
    if _split_ref(src) and not os.path.isfile(src):
        model = _instantiate(_load_object(src), src)
        x = _example_input(args, model)
        from .torch import from_module

        spec = from_module(model, x)
        if os.path.isfile(_split_ref(src)[0]):
            spec.origin.source = f"{os.path.basename(_split_ref(src)[0])}:{_split_ref(src)[1]}"
        return spec, model, x
    if src.endswith(".pta"):
        return load(src)[0], None, None
    return as_spec(src), None, None


def _add_input_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--input-shape", type=_parse_shape, metavar="1,64", help="build an example input of this shape (enables shapes, the hooks fallback and traces)")
    p.add_argument("--dtype", choices=("long", "float"), default="long", help="example input dtype: long token ids (default) or float features")
    p.add_argument("--vocab", type=int, help="upper bound for random token ids (default: the first nn.Embedding's size)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pta", description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", action="version", version=f"papertoanything {__version__}")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    s = sub.add_parser("link", help="print (and open) a Lab link for a spec or model", description="Encode a spec into https://lab.papertoanything.com/#s=... (deflate-raw + base64url in the fragment).")
    s.add_argument("source", help="spec.json, model.pta, or file.py:Name")
    s.add_argument("--no-open", action="store_true", help="print only, do not open a browser")
    _add_input_args(s)

    s = sub.add_parser("decode", help="decode a Lab link back to spec JSON")
    s.add_argument("url", help="a full link, '#s=...', or the bare payload")

    s = sub.add_parser("inspect", help="map a model to a spec and print its blocks")
    s.add_argument("source", help="file.py:Name or package.module:Name (or a spec/.pta file)")
    s.add_argument("--json", action="store_true", help="print the spec as JSON instead of a table")
    s.add_argument("--link", action="store_true", help="also print the Lab link")
    _add_input_args(s)

    s = sub.add_parser("save", help="write a .pta file to drop on the Lab")
    s.add_argument("source", help="spec.json or file.py:Name")
    s.add_argument("-o", "--output", help="output path (default <name>.pta)")
    _add_input_args(s)

    s = sub.add_parser("serve", help="serve the bridge on 127.0.0.1 and open the Lab until Ctrl+C")
    s.add_argument("source", nargs="?", help="optional spec.json, .pta, or file.py:Name to show")
    s.add_argument("--lab-url", help="open this Lab instead of https://lab.papertoanything.com (default: $PTA_LAB_URL)")
    s.add_argument("--port", type=int, default=0, help="port on 127.0.0.1 (default: a free one)")
    s.add_argument("--no-open", action="store_true", help="do not open a browser")
    _add_input_args(s)
    return p


def _table(spec: Any) -> str:
    rows = [("id", "kind", "label", "params", "inputs", "group")]
    for b in spec.blocks:
        params = " ".join(f"{k}={v}" for k, v in b.params.to_dict().items())
        rows.append((b.id, b.kind, b.label or "", params, ",".join(b.inputs), b.group or ""))
    widths = [min(max(len(r[i]) for r in rows), 40) for i in range(len(rows[0]))]
    lines = ["  ".join(c[: widths[i]].ljust(widths[i]) for i, c in enumerate(r)).rstrip() for r in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    lines.append("")
    lines.append(f"{spec.name}: {len(spec.blocks)} blocks, output {spec.output}, origin {spec.origin.kind}:{spec.origin.source or ''}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.cmd:
        parser.print_help()
        return 0
    from .link import from_url, to_url

    if args.cmd == "decode":
        print(from_url(args.url).to_json(indent=2))
        return 0
    if args.cmd == "link":
        spec, _, _ = _source(args)
        url = to_url(spec)
        print(url)
        if not args.no_open and not os.environ.get("PTA_NO_BROWSER"):
            import webbrowser

            webbrowser.open(url)
        return 0
    if args.cmd == "inspect":
        spec, _, _ = _source(args)
        print(spec.to_json(indent=2) if args.json else _table(spec))
        if args.link:
            print(to_url(spec))
        return 0
    if args.cmd == "save":
        from .show import save

        spec, model, x = _source(args)
        trace = None
        if model is not None and x is not None:
            from .torch import capture

            trace = capture(model, x, spec=spec)
        out = save(spec, args.output or f"{spec.name}.pta", trace=trace)
        print(f"wrote {out}" + (" (spec + trace)" if trace is not None else " (spec only; pass --input-shape to include a trace)"))
        return 0
    if args.cmd == "serve":
        from .server import LocalServer

        spec = model = x = None
        if args.source:
            spec, model, x = _source(args)
        server = LocalServer(spec=spec, lab_url=args.lab_url, port=args.port).start()
        if model is not None and x is not None:
            from .torch import capture

            server.push(capture(model, x, spec=spec))
        print(f"serving on {server.url}")
        print("bridge on " + server.origin + " (127.0.0.1 only); the Lab reads it from your browser")
        print("press Ctrl+C to stop")
        if not args.no_open and not os.environ.get("PTA_NO_BROWSER"):
            import webbrowser

            webbrowser.open(server.url)
        server.serve_forever()
        return 0
    parser.error(f"unknown command {args.cmd}")
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
