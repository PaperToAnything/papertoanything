# papertoanything

See a PyTorch model, and watch it train, in a browser tab. Nothing is uploaded.

```bash
pip install "papertoanything[torch]"      # or plain `papertoanything` for links and files only
```

```python
import papertoanything as pta

w = pta.watch(model, optimizer)    # opens a live page on 127.0.0.1
for x, y in loader:
    loss = loss_fn(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    w.step(loss)
```

The page shows loss, validation loss, gradient norm, and a row per block
(attention, MLP, norm, embedding, ...): activation mean, spread and largest
value, the share of exact zeros, the gradient that reaches the block, the
size of its weights, how much the last step changed them, and the entropy of
each attention head. Dead ReLUs, divergence and collapsed attention show up
as numbers you can see change, without adding a single `print`.

Version 0.0.1. Python 3.10 or newer. No required dependencies: links, files
and the server use only the standard library. PyTorch (2.0 or newer) is
needed for anything that reads a model.

## What works in 0.0.1

| | |
|---|---|
| `pta.watch(model, optimizer)` | Live training health on a local page, plus a replay file in `./pta-runs/`. |
| `pta.show(model, x, mode="link")` | Lab link with the architecture in the URL fragment. |
| `pta.show(model, x, mode="local")` | Local server with the spec and one captured forward pass. |
| `pta.show(model, x, mode="file")` / `pta.save` | A `.pta` file to drop on the Lab. |
| `pta.from_module(model, x)` | `nn.Module` to the Lab's model spec. |
| `pta.capture(model, x)` | Every block's output from one forward pass. |
| `pta` CLI | `link`, `decode`, `inspect`, `save`, `serve`, `--version`. |
| Callbacks | Hugging Face `Trainer` and Lightning (optional imports). |

**The Lab is not bundled in 0.0.1.** The local server shows the package's
own viewer: the plain-numbers page described above, with charts and a table.
The Lab at lab.papertoanything.com draws the model and turns these numbers
into diagnoses; once its build ships inside the package, the same local
server will serve it with no change to your code. Until then, links open the
hosted Lab in design view, and live training uses the built-in viewer.

## Three ways to get a model into a tab, none of which upload it

**1. Link (design view, any size).** The spec (the list of blocks and how
they connect, no weights and no data) is compressed into the part of the URL
after `#`. Browsers never send that part to any server, so the hosted Lab
page loads and reads the model from the address bar; Paper To Anything's
servers never see it.

```python
pta.show(model, example_input, mode="link")     # prints and opens the link
```

**2. Local (live).** The package starts a small web server inside your
Python process, bound to `127.0.0.1` only, on a random free port, with a
random token for the session. Your browser talks to your own process. It
serves the viewer, the spec, and a stream of frames as training runs. This
is how live activations and training health reach the tab: they never leave
the machine.

```python
server = pta.show(model, example_input, mode="local")   # returns immediately
server.push(pta.capture(model, other_input, spec=server.spec))
server.close()
```

**3. File.** `pta.save(model, "model.pta", example_input=x)` writes the spec
and one captured forward pass as JSON. Drop it on the Lab, or keep it.

`mode="auto"` (the default) uses the local server when a Lab build is
available and a link otherwise.

## In a notebook

```python
import papertoanything as pta

w = pta.watch(model, optimizer, every=20)   # prints the URL and opens a tab
# ... run your training cells; the server runs in a background thread ...
w.close()
```

Cells return immediately; the tab keeps updating while later cells train. A
tab opened late fetches the history it missed. Pass `open=False` to skip
opening a browser, and use `w.url` to open it yourself.

## In a script

```python
import papertoanything as pta

with pta.watch(model, optimizer, every=10, val_loss=evaluate) as w:
    for step, (x, y) in enumerate(loader):
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        w.step(loss)
```

* With `optimizer`, steps are counted by the optimizer's own step hooks;
  `w.step(loss)` just attaches the loss. Without it, call `w.step(loss)` once
  per step, after `optimizer.step()`.
* If the model returns a scalar loss (nanoGPT returns `(logits, loss)`), the
  loss is picked up automatically.
* `val_loss` is a number or a function; a function is called every
  `val_every` steps (default ten frames).
* The model is mapped to blocks on the first training forward pass, using
  that batch. Pass `example_input=` to map it up front.

Try it on a tiny GPT that trains on the CPU in under a minute:

```bash
python examples/watch_tiny_gpt.py                 # healthy
python examples/watch_tiny_gpt.py --break relu    # dead ReLUs: "mlp hidden zero %" goes to 100
python examples/watch_tiny_gpt.py --break lr      # learning rate too high: loss explodes
```

### Hugging Face and Lightning

```python
from papertoanything.integrations.hf import PTACallback
trainer = Trainer(model=model, args=args, train_dataset=ds, callbacks=[PTACallback(every=20)])

from papertoanything.integrations.lightning import PTACallback
trainer = L.Trainer(callbacks=[PTACallback(every=20)])
```

## What `watch` measures

Every `every` steps (default 10), one frame with:

* per block: activation statistics (mean, standard deviation, largest
  absolute value, fraction of exact zeros, NaN and Inf counts, a 32-bin
  histogram; the fraction of saturated values for tanh and sigmoid), the
  same statistics for the gradient arriving at the block's output, the norm
  of its weights and of their gradient, and the update ratio
  ‖Δθ‖ / ‖θ‖ for the last step;
* for MLP blocks, the statistics of the inner activation, where dead ReLUs
  are visible;
* for attention blocks, the entropy of each head's attention rows and the
  top-left 32 × 32 of each head's pattern for the first sequence in the
  batch, but only when the probabilities can be observed: the module returns
  them per head, or computes them with `softmax`, or calls
  `scaled_dot_product_attention` without a custom mask (then the pattern is
  recomputed from the same queries and keys). Otherwise they are left out,
  never estimated;
* loss, validation loss, learning rate, global gradient norm, time split
  into data, forward, backward and optimizer where measurable, and CUDA
  memory.

Between frames the hooks return immediately. On a frame step the statistics
are reductions computed where the tensor lives, then copied once. The update
ratio keeps a copy of the parameters on frame steps; turn it off with
`update_ratio=False` for very large models. The package does not decide what
is wrong; the Lab does that from the frames. It only prints a warning when a
NaN or Inf first appears.

## `from_module`: how a model becomes blocks

`pta.from_module(model, example_input)` tries `torch.fx` first. If the model
has data-dependent control flow (an `assert` on the sequence length is
enough), it runs one forward pass with hooks instead and records what ran.
Without an example input and without fx, it falls back to the module tree
and warns that the connections are a guess.

It recognises `nn.Embedding` (a learned position table when indexed by
`arange`), `nn.MultiheadAttention` and attention modules that carry a head
count (nanoGPT's `CausalSelfAttention`, Hugging Face `GPT2Attention`, BERT
self-attention), MLP modules, `nn.Linear` and Hugging Face `Conv1D`,
LayerNorm and RMSNorm, ReLU, GELU, tanh, sigmoid and SiLU, softmax, losses,
and the residual `x + f(x)` additions between them. Tied output layers become
`unembed` with `tiedTo`. Anything else becomes a `group` block labelled with
its class name. It does not raise on a model it does not know. The model is
put in eval mode for the pass, run without gradients, and restored.

## Command line

```bash
pta --version
pta inspect model.py:GPT --input-shape 1,64        # table of blocks
pta inspect model.py:GPT --json > spec.json
pta link spec.json                                 # open the Lab with it
pta decode "https://lab.papertoanything.com/#s=..."
pta save model.py:GPT -o gpt.pta --input-shape 1,64
pta serve spec.json                                # local viewer until Ctrl+C
```

`model.py:Name` may name a class that builds with no arguments, a function
that returns a model, or a model instance.

## Privacy

* The local server listens on `127.0.0.1` only, never on a network
  interface. Its data endpoints require the per-session token in the URL it
  opens, refuse requests whose `Host` is not that loopback address, and
  refuse requests from any other origin. It sends no CORS headers, so other
  websites in the same browser cannot read it.
* Link mode puts the spec in the URL fragment. Browsers do not send the
  fragment to servers. Anyone you send the link to can read the architecture
  in it, so share links the way you would share the spec.
* Run files and `.pta` files are written to your disk and nowhere else.
* The package makes no network requests of its own: no telemetry, no update
  checks, no fonts or scripts from a CDN in the viewer.

## Formats

* **Link:** `https://lab.papertoanything.com/#s=` + base64url (no padding) of
  raw DEFLATE (RFC 1951) of the UTF-8 compact JSON spec. `tests/vectors/link.json`
  is the shared test vector with the TypeScript engine.
* **Local server:** `GET /spec` (bridge hello), `GET /trace` (latest forward
  pass), `GET /health?since=N` (training frames after step N),
  `GET /events` (server-sent events: `hello`, `spec`, `frame`, `health`,
  `bye`). All take `?token=`.
* **`.pta` file:** one JSON object `{"protocol":"pta-file","v":1,"spec":...,"trace":...}`.
  **Run file:** JSON lines, a header `{"protocol":"pta-run","v":1,"spec":...}`
  then one `{"event":"health","data":{...}}` per frame.
* Tensors are base64 of little-endian float32. A tensor too large to send
  whole is cut to its leading corner and marked `"clipped": true` with the
  `dataShape` that was sent; `shape` is always the real shape.

## Limitations

* The Lab's build is not bundled yet; locally you get the built-in viewer.
* One process, one model per `watch`. Distributed training: call `watch`
  on rank 0 only (the callbacks do this).
* `torch.compile`d models: watch the original module, not the compiled
  wrapper.
* Attention patterns are omitted for fused kernels that hide the
  probabilities and for modules that only return head-averaged weights.
* Functional blocks (residual adds, activations written as `F.gelu`) have
  activations in `capture` but no statistics in `watch`, which hooks modules.
* A link for a very large architecture can be too long for some systems to
  open; use a file instead.
* Hugging Face and Lightning callbacks are written against current
  versions; the Hugging Face callback is tested only for import here.

## Developing

```bash
cd python
python -m pytest            # or: python -m unittest discover -s tests -t .
python -m build             # sdist and wheel in dist/
```

Licensed under Apache-2.0.
