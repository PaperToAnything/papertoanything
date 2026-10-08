# papertoanything

See a PyTorch model, and watch it train, in the [Paper To Anything Lab](https://lab.papertoanything.com). Nothing is uploaded.

This is open-source research work, developed and maintained by one person. It is alpha software and still in development; the spec format is version 1 and may change before 1.0.

`papertoanything` (`import papertoanything as ptoa`, command `ptoa`) turns an `nn.Module` into a model spec the Lab can draw, and streams training health from your Python process into a browser tab on your own machine.

## Install

```bash
pip install papertoanything
```

Python 3.10 or newer. The package has no required dependencies. Anything that reads a model needs PyTorch 2.0 or newer (`pip install "papertoanything[torch]"`). The `hf` extra adds `transformers` for the Hugging Face callback.

## Three ways to get a model into the Lab

None of them uploads anything.

| | Command | What reaches the Lab | Network |
|---|---|---|---|
| Link | `ptoa.show(model, x, mode="link")` | The architecture, in the URL fragment | Only to load the Lab page |
| Live bridge | `ptoa.watch(model, optimizer)` or `ptoa.show(..., mode="local")` | Architecture, forward-pass tensors and training health, read from your own process by your own browser | Only to load the Lab page |
| File | `ptoa.save(model, "m.pta", example_input=x)` | A `.pta` file you drop on the Lab | None |

## Quick start

```python
import papertoanything as ptoa

ptoa.show(model, example_input)                  # prints and opens a link
ptoa.show(model, example_input, mode="file")     # writes <name>.pta
```

Watch training:

```python
with ptoa.watch(model, optimizer, every=10) as w:
    for x, y in loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        w.step(loss)
```

`watch` prints the Lab URL and opens it. The page shows loss, gradient norm, and per-block activation and gradient statistics, update ratios, and attention-head statistics where they can be observed. Each frame is also written to a JSONL run file in `./pta-runs/` that you can replay in the Lab.

Command line:

```bash
ptoa inspect model.py:GPT --input-shape 1,64        # table of blocks
ptoa save model.py:GPT -o gpt.pta --input-shape 1,64
ptoa serve spec.json                                # live bridge until Ctrl+C
```

Hugging Face `Trainer` and PyTorch Lightning are supported through `papertoanything.integrations.hf.PTACallback` and `papertoanything.integrations.lightning.PTACallback`.

## Privacy

- The package makes no network requests of its own: no telemetry, no update check.
- The bridge listens on 127.0.0.1 only, behind a random per-session token, and answers only the Lab's origin.
- Weights and training data are never put in a spec or a link.

## Limitations

- The Lab is a hosted web app and is not part of this package. The live bridge needs the Lab page to load; file mode does not.
- One process and one model per `watch`. Safari blocks the live bridge; use file mode there.
- Attention patterns are omitted for fused kernels that hide the probabilities.

## Links

- Homepage: https://papertoanything.com
- Repository: https://github.com/PaperToAnything/papertoanything
- Documentation: https://papertoanything.com/lab/import/
- Lab: https://lab.papertoanything.com

## Licence

Apache-2.0. Copyright 2026 Dhruva P Gowda. The Lab web app is a separate, proprietary product and is not covered by this licence.
