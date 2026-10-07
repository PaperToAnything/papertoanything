"""Hugging Face ``transformers.Trainer`` integration.

    from papertoanything.integrations.hf import PTACallback
    trainer = Trainer(..., callbacks=[PTACallback(every=20)])

Steps are counted by the Trainer's optimizer step hooks; the logged training
loss and ``eval_loss`` are attached to frames.
"""

from __future__ import annotations

from typing import Any, Optional

try:
    from transformers import TrainerCallback
except ImportError as e:  # pragma: no cover - depends on environment
    raise ImportError("papertoanything.integrations.hf needs transformers: pip install transformers") from e

from ..health import Watch


class PTACallback(TrainerCallback):
    def __init__(self, every: int = 10, open: bool = True, **watch_kwargs: Any) -> None:
        self.every = every
        self.open = open
        self.watch_kwargs = watch_kwargs
        self.watch: Optional[Watch] = None

    def on_train_begin(self, args: Any, state: Any, control: Any, model: Any = None, optimizer: Any = None, **kwargs: Any) -> None:
        if model is None or self.watch is not None:
            return
        if not getattr(state, "is_world_process_zero", True):
            return
        self.watch = Watch(model, optimizer, self.every, None, self.open, **self.watch_kwargs)

    def on_step_begin(self, args: Any, state: Any, control: Any, optimizer: Any = None, **kwargs: Any) -> None:
        # The Trainer may create the optimizer after on_train_begin.
        w = self.watch
        if w is not None and w.optimizer is None and optimizer is not None:
            w.optimizer = optimizer
            w._handles.append(optimizer.register_step_pre_hook(w._opt_pre))
            w._handles.append(optimizer.register_step_post_hook(w._opt_post))

    def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict] = None, **kwargs: Any) -> None:
        if self.watch is None or not logs:
            return
        self.watch.log(loss=logs.get("loss"), val_loss=logs.get("eval_loss"))

    def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[dict] = None, **kwargs: Any) -> None:
        if self.watch is not None and metrics:
            self.watch.log(val_loss=metrics.get("eval_loss"))

    def on_train_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if self.watch is not None:
            self.watch.close()
