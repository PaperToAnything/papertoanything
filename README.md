# papertoanything

See a PyTorch model, and watch it train, in the [Paper To Anything Lab](https://lab.papertoanything.com). Nothing is uploaded.

`papertoanything` (`import papertoanything as pta`, command `pta`) turns an `nn.Module` into a model spec the Lab can draw, and streams training health from your Python process into a browser tab on your own machine.

```bash
pip install papertoanything
```

Python 3.10 or newer. The package has no required dependencies. Anything that reads a model needs PyTorch 2.0 or newer (`pip install "papertoanything[torch]"`). Extras: `torch`, and `hf` (adds `transformers` for the Hugging Face callback).

## What it does

There are three ways to get a model into the Lab. None of them uploads anything.

| | Command | What reaches the Lab | Network |
|---|---|---|---|
| **Link** | `pta.show(model, x, mode="link")` | The architecture, in the URL fragment | Only to load the Lab page |
| **Live bridge** | `pta.watch(model, optimizer)` or `pta.show(..., mode="local")` | Architecture, forward-pass tensors and training health, read from your own process by your own browser | Only to load the Lab page |
| **File** | `pta.save(model, "m.pta", example_input=x)` | A `.pta` file you drop on the Lab | None |

* **Link.** The spec (the blocks and how they connect; no weights, no data) is compressed into the part of the URL after `#`. Browsers never send that part to a server.
* **Live bridge.** `pta.show(..., mode="local")` and `pta.watch` start a small server in your Python process on `127.0.0.1`, on a random port, protected by a random token for the session. They then open `https://lab.papertoanything.com/?bridge=http://127.0.0.1:<port>&token=<token>`. The Lab page runs in your browser and reads the bridge directly; the data goes from your process to your browser tab and nowhere else.
* **File.** `pta save` writes a JSON file. Drop it on the Lab. This works in every browser and on every machine, with no server and no connection to your process.

## Quick starts

### Show an architecture

```python
import papertoanything as pta

pta.show(model, example_input)                  # prints and opens a link
pta.show(model, example_input, mode="local")    # live bridge, with one captured forward pass
pta.show(model, example_input, mode="file")     # writes <name>.pta
```

### Watch training

```python
import papertoanything as pta

with pta.watch(model, optimizer, every=10) as w:
    for x, y in loader:
        loss = loss_fn(model(x), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        w.step(loss)
```

`watch` prints the Lab URL and opens it. The page shows loss, validation loss, gradient norm, and one row per block (attention, MLP, norm, embedding and so on): activation mean, spread and largest value, the share of exact zeros, the gradient reaching the block, the size of its weights, how much the last step changed them, and the entropy of each attention head. Each frame is also written to a JSONL run file in `./pta-runs/`, which you can drop on the Lab to replay the run.

* With `optimizer`, steps are counted by the optimizer's own step hooks and `w.step(loss)` only attaches the loss. Without it, call `w.step(loss)` once per step, after `optimizer.step()`.
* If the model returns a scalar loss (nanoGPT returns `(logits, loss)`), the loss is picked up automatically.
* `val_loss` is a number or a function; a function is called every `val_every` steps.
* `open=False` skips opening a browser; `w.url` has the URL.
* In a notebook, `w = pta.watch(...)` returns at once and the server keeps running in a background thread; call `w.close()` when done.

A tiny GPT that trains on the CPU in under a minute:

```bash
python examples/watch_tiny_gpt.py                 # healthy
python examples/watch_tiny_gpt.py --break relu    # dead ReLUs: "mlp hidden zero %" goes to 100
python examples/watch_tiny_gpt.py --break lr      # learning rate too high: loss diverges
```

### Command line

```bash
pta --version
pta inspect model.py:GPT --input-shape 1,64        # table of blocks
pta inspect model.py:GPT --json > spec.json
pta link spec.json                                 # open the Lab with the architecture
pta decode "https://lab.papertoanything.com/#s=..."
pta save model.py:GPT -o gpt.pta --input-shape 1,64
pta serve spec.json                                # live bridge until Ctrl+C
```

`model.py:Name` may name a class that builds with no arguments, a function that returns a model, or a model instance.

### Hugging Face and Lightning

```python
from papertoanything.integrations.hf import PTACallback
trainer = Trainer(model=model, args=args, train_dataset=ds, callbacks=[PTACallback(every=20)])

from papertoanything.integrations.lightning import PTACallback
trainer = L.Trainer(callbacks=[PTACallback(every=20)])
```

## Browser notes for the live bridge

The hosted Lab is an `https://` page talking to `http://127.0.0.1`. Chrome, Edge and Firefox treat loopback as a secure context, so this works. Chrome may show a one-time prompt, "allow access to devices on your local network"; allow it for `lab.papertoanything.com`. Safari blocks `https` pages from reading `http://127.0.0.1`. If the bridge does not connect, or you work on a remote machine, in a locked-down browser or without a network, use file mode: `pta save` (or `pta.save(...)`), then drop the `.pta` file on the Lab. File mode works everywhere.

## What `watch` measures

Every `every` steps (default 10), one frame with:

* per block: activation statistics (mean, standard deviation, largest absolute value, fraction of exact zeros, NaN and Inf counts, a 32-bin histogram; the fraction of saturated values for tanh and sigmoid), the same statistics for the gradient arriving at the block, the norm of its weights and of their gradient, and the update ratio ‖Δθ‖ / ‖θ‖ for the last step;
* for MLP blocks, the statistics of the inner activation, where dead ReLUs show;
* for attention blocks, the entropy of each head's attention rows and the top-left 32 × 32 of each head's pattern for the first sequence in the batch, when the probabilities can be observed (the module returns them per head, computes them with `softmax`, or calls `scaled_dot_product_attention` without a custom mask). Otherwise they are left out, not estimated;
* loss, validation loss, learning rate, global gradient norm, time split into data, forward, backward and optimizer where measurable, and CUDA memory.

Between frames the hooks return immediately. On a frame step the statistics are reductions computed on the device where the tensor lives, and only the small results are copied to the CPU. The update ratio keeps a copy of the parameters on frame steps; turn it off with `update_ratio=False` for very large models. The bridge keeps the last 5000 frames for tabs that connect late. The package does not decide what is wrong; the Lab does that from the frames. It only warns when a NaN or Inf first appears.

## How a model becomes blocks

`pta.from_module(model, example_input)` tries `torch.fx` first. If the model has data-dependent control flow, it runs one forward pass with hooks instead and records what ran. Without an example input and without fx, it falls back to the module tree and warns that the connections are a guess.

It recognises `nn.Embedding` (and learned position tables), `nn.MultiheadAttention` and attention modules that carry a head count (nanoGPT's `CausalSelfAttention`, Hugging Face `GPT2Attention`, BERT self-attention), MLP modules, `nn.Linear` and Hugging Face `Conv1D`, LayerNorm and RMSNorm, common activations, softmax, losses, and residual additions. Tied output layers become `unembed` with `tiedTo`. Anything else becomes a `group` block labelled with its class name; it does not raise on a model it does not know. The model is put in eval mode for the pass, run without gradients, and restored.

## Privacy

* The package makes no network requests of its own: no telemetry, no update check.
* The bridge listens on `127.0.0.1` only, never on a network interface.
* Its data endpoints require the random per-session token, refuse requests whose `Host` is not that loopback address, and refuse any request whose `Origin` is not the bridge itself, `https://lab.papertoanything.com`, or `http://localhost:5180` (a Lab dev server). Only those origins receive CORS headers; there is no wildcard.
* In link mode the spec is in the URL fragment, which browsers do not send to any server. Anyone you send the link to can read the architecture in it.
* Run files and `.pta` files are written to your disk and nowhere else.
* Weights and training data are never put in a spec or a link. Forward-pass tensors and health statistics leave your process only over the loopback bridge, and only when you ask for them.

## Formats

* **Link:** `https://lab.papertoanything.com/#s=` + base64url (no padding) of raw DEFLATE (RFC 1951) of the UTF-8 compact JSON spec. `tests/vectors/link.json` is the shared test vector with the TypeScript engine.
* **Bridge:** `GET /spec` (bridge hello), `GET /trace` (latest forward pass), `GET /health?since=N` (frames after step N), `GET /events` (server-sent events: `hello`, `spec`, `frame`, `health`, `bye`). All take `?token=`.
* **`.pta` file:** one JSON object `{"protocol":"pta-file","v":1,"spec":...,"trace":...}`. **Run file:** JSON lines, a header `{"protocol":"pta-run","v":1,"spec":...}` then one `{"event":"health","data":{...}}` per frame.
* Tensors are base64 of little-endian float32. A tensor too large to send whole is cut to its leading corner and marked `"clipped": true` with the `dataShape` that was sent.

## Supported frameworks

* PyTorch 2.0 or newer: `from_module`, `capture`, `show`, `watch`.
* Hugging Face `transformers` `Trainer` and PyTorch Lightning, through callbacks.
* Anything else: build a spec yourself (`pta.ModelSpec`) and use link, bridge or file mode with it.

## Limitations

* The Lab is a hosted web app and is not part of this package. The live bridge needs the Lab page to load, so it needs a network connection for that; file mode does not.
* One process and one model per `watch`. For distributed training call `watch` on rank 0 only (the callbacks do this).
* `torch.compile`d models: watch the original module, not the compiled wrapper.
* Attention patterns are omitted for fused kernels that hide the probabilities and for modules that only return head-averaged weights.
* Functional blocks (residual adds, activations written as `F.gelu`) have activations in `capture` but no statistics in `watch`, which hooks modules.
* A link for a very large architecture can be too long for some systems to open; use a file instead.
* The Hugging Face and Lightning callbacks are written against current versions; the Hugging Face callback is tested only for import.
* Alpha software: the spec format is version 1 and may change before 1.0.

## Development

```bash
git clone https://github.com/PaperToAnything/pta
cd pta
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test]"
python -m pytest -q
python -m build            # sdist and wheel in dist/
```

Documentation: https://papertoanything.com/lab/import/

## Citation

```bibtex
@software{dhruvapgowda2026pta,
  author  = {Dhruva P Gowda},
  title   = {papertoanything: see a PyTorch model and its training in the Paper To Anything Lab},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/PaperToAnything/pta},
  license = {Apache-2.0}
}
```

## Licence

Apache-2.0. See [LICENSE](LICENSE). Copyright 2026 Dhruva P Gowda.

The Lab web app at lab.papertoanything.com is a separate, proprietary product and is not covered by this licence.
