"""Watch a tiny GPT train, live, in a browser tab.

    pip install "papertoanything[torch]"
    python watch_tiny_gpt.py                 # healthy run
    python watch_tiny_gpt.py --break relu    # dead ReLUs in the MLPs
    python watch_tiny_gpt.py --break lr      # learning rate far too high: divergence

A nanoGPT-style character model (2 layers, 2 heads, d=32) learns a
synthetic sequence on the CPU in about 200 steps. `pta.watch` starts a bridge on
127.0.0.1 and opens the hosted Lab on it, which shows loss, gradient norm and
per-block statistics as it trains. Nothing is uploaded.

--break relu  swaps GELU for ReLU and starts the MLPs' first layer with a
              bias of -4, so almost every hidden unit outputs exactly zero
              and never recovers (watch "zero %" on the mlp blocks).
--break lr    uses lr=3.0 with no warmup, so updates are huge relative to
              the weights (watch "update ratio" and the loss).
"""

import argparse
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import papertoanything as pta


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size):
        super().__init__()
        self.c_attn = nn.Linear(n_embd, 3 * n_embd)
        self.c_proj = nn.Linear(n_embd, n_embd)
        self.n_head, self.n_embd = n_head, n_embd
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, n_embd, relu=False):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.act = nn.ReLU() if relu else nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)

    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, relu=False):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, relu)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x + self.mlp(self.ln_2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab, block_size=32, n_layer=2, n_head=2, n_embd=32, relu=False):
        super().__init__()
        self.block_size = block_size
        self.wte = nn.Embedding(vocab, n_embd)
        self.wpe = nn.Embedding(block_size, n_embd)
        self.h = nn.ModuleList([Block(n_embd, n_head, block_size, relu) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab, bias=False)
        self.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        b, t = idx.size()
        assert t <= self.block_size  # control flow: pta falls back from fx to hooks
        x = self.wte(idx) + self.wpe(torch.arange(t, device=idx.device))
        for block in self.h:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def make_text(n=20000, seed=0):
    """A synthetic language: words from a tiny grammar, so there is structure to learn."""
    g = torch.Generator().manual_seed(seed)
    subjects = ["the cat", "a dog", "my friend", "the model"]
    verbs = ["sees", "likes", "trains", "copies"]
    objects = ["the ball.", "a token.", "the next word.", "itself."]
    out = []
    while sum(len(s) for s in out) < n:
        i, j, k = (int(torch.randint(0, 4, (1,), generator=g)) for _ in range(3))
        out.append(f"{subjects[i]} {verbs[j]} {objects[k]} ")
    return "".join(out)[:n]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--break", dest="breakage", choices=("relu", "lr"), help="train badly on purpose")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--every", type=int, default=5, help="emit a health frame every N steps")
    ap.add_argument("--no-open", action="store_true", help="do not open a browser tab")
    ap.add_argument("--no-wait", action="store_true", help="exit when training ends instead of keeping the page alive")
    ap.add_argument("--delay", type=float, default=0.02, help="seconds to sleep per step so the run is watchable")
    args = ap.parse_args()

    torch.manual_seed(0)
    text = make_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    block = 32

    model = TinyGPT(len(chars), block_size=block, relu=args.breakage == "relu")
    if args.breakage == "relu":
        for blk in model.h:
            nn.init.constant_(blk.mlp.c_fc.bias, -4.0)
    lr = 3.0 if args.breakage == "lr" else 3e-3
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    def batch(bs=16):
        ix = torch.randint(0, len(data) - block - 1, (bs,))
        x = torch.stack([data[i : i + block] for i in ix])
        y = torch.stack([data[i + 1 : i + block + 1] for i in ix])
        return x, y

    def val_loss():
        model.eval()
        with torch.no_grad():
            x, y = batch(64)
            _, loss = model(x, y)
        model.train()
        return loss.item()

    w = pta.watch(model, opt, every=args.every, val_loss=val_loss, open=not args.no_open, val_every=args.every * 4)
    with w:
        for step in range(args.steps):
            x, y = batch()
            _, loss = model(x, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            w.step(loss)
            if step % 50 == 0:
                print(f"step {step:4d}  loss {loss.item():.3f}")
            if args.delay:
                time.sleep(args.delay)
        print(f"done: {len(w.frames)} health frames; run file {w.run_file.path if w.run_file else '(none)'}")
        if not args.no_wait and w.server is not None:
            print(f"bridge still at {w.url}; press Ctrl+C to exit")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()
