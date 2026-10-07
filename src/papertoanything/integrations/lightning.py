"""Lightning integration.

    from papertoanything.integrations.lightning import PTACallback
    trainer = L.Trainer(callbacks=[PTACallback(every=20)])

Works with ``lightning`` (2.x) or the older ``pytorch_lightning`` package.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from lightning.pytorch.callbacks import Callback
except ImportError:  # pragma: no cover - depends on environment
    try:
        from pytorch_lightning.callbacks import Callback  # type: ignore[no-redef]
    except ImportError as e:
        raise ImportError("papertoanything.integrations.lightning needs lightning: pip install lightning") from e

from ..health import Watch


class PTACallback(Callback):
    def __init__(self, every: int = 10, open: bool = True, **watch_kwargs: Any) -> None:
        super().__init__()
        self.every = every
        self.open = open
        self.watch_kwargs = watch_kwargs
        self.watch: Optional[Watch] = None

    def on_fit_start(self, trainer: Any, pl_module: Any) -> None:
        if getattr(trainer, "global_rank", 0) != 0:
            return
        opts = getattr(trainer, "optimizers", None) or []
        optimizer = opts[0] if opts else None
        # Lightning wraps optimizers; the hooks belong on the torch optimizer.
        optimizer = getattr(optimizer, "optimizer", optimizer)
        self.watch = Watch(pl_module, optimizer, self.every, None, self.open, **self.watch_kwargs)

    def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
        if self.watch is None:
            return
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        self.watch.step(loss)

    def on_validation_epoch_end(self, trainer: Any, pl_module: Any) -> None:
        if self.watch is None:
            return
        metrics = getattr(trainer, "callback_metrics", {}) or {}
        for key in ("val_loss", "val/loss", "validation_loss"):
            if key in metrics:
                self.watch.log(val_loss=metrics[key])
                break

    def on_fit_end(self, trainer: Any, pl_module: Any) -> None:
        if self.watch is not None:
            self.watch.close()
