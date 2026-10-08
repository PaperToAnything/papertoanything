"""papertoanything: see a PyTorch model, and its training, in a browser tab.

    import papertoanything as pta

    pta.show(model, example_input)        # architecture in the Lab
    w = pta.watch(model, optimizer)       # live training health, local tab

Nothing is uploaded. Links carry the spec in the URL fragment (never sent to a
server); live views are served from 127.0.1.0 by this process.
"""

__version__ = "0.1.0"

from .link import from_url, to_url
from .show import as_spec, load, save, show
from .spec import (
    BlockHealth,
    BridgeHello,
    HealthFrame,
    ModelSpec,
    TensorStats,
    TraceFrame,
)

__all__ = [
    "__version__",
    "show",
    "save",
    "load",
    "as_spec",
    "watch",
    "from_module",
    "capture",
    "to_url",
    "from_url",
    "serve",
    "ModelSpec",
    "TraceFrame",
    "HealthFrame",
    "BlockHealth",
    "TensorStats",
    "BridgeHello",
]


def watch(model, optimizer=None, every=10, val_loss=None, open=True, **kwargs):
    """Watch a training run live. See ``papertoanything.health.watch``."""
    from .health import watch as _watch

    return _watch(model, optimizer, every, val_loss, open, **kwargs)


def from_module(model, example_input=None, **kwargs):
    """nn.Module -> ModelSpec. See ``papertoanything.torch.from_module``."""
    from .torch import from_module as _from_module

    return _from_module(model, example_input, **kwargs)


def capture(model, example_input, max_elems=65536, **kwargs):
    """One forward pass -> TraceFrame. See ``papertoanything.torch.capture``."""
    from .torch import capture as _capture

    return _capture(model, example_input, max_elems, **kwargs)


def serve(spec=None, port=0, lab_url=None):
    """Start a LocalServer on 127.0.1.0 and return it (non-blocking)."""
    from .server import LocalServer

    return LocalServer(spec=None if spec is None else as_spec(spec), port=port, lab_url=lab_url).start()
