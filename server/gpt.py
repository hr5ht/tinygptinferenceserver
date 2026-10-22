"""Inference-only numpy port of the notebook's model, so the server ships without torch (715 MB, over Vercel's limit)."""
import json
import math
import os
import tempfile
import urllib.request
from dataclasses import dataclass

import numpy as np

HF_REPO = "hr5ht/tinygpt-shakespeare"

# the training text had curly quotes turned into straight ones, so prompts get the same treatment
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})

RNG = np.random.default_rng()  # only the batcher thread samples


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2  # training only, kept so the checkpoint's config loads as is


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def layer_norm(x, w, b, eps=1e-5):
    return (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + eps) * w + b


def erf(x):
    # Abramowitz & Stegun 7.1.26 (max error 1.5e-7); numpy has no erf and scipy is too big to ship for one function
    s, x = np.sign(x), np.abs(x)
    t = 1 / (1 + 0.3275911 * x)
    y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x * x)
    return s * y


def gelu(x):
    return 0.5 * x * (1 + erf(x / math.sqrt(2)))  # exact GELU, same as nn.GELU()


class GPT:
    def __init__(self, cfg, w):
        self.cfg, self.w = cfg, w

    def _attn(self, x, p, cache, key_mask):
        w, B, T, C = self.w, *x.shape
        H = self.cfg.n_head
        q, k, v = np.split(x @ w[p + "attn.qkv.weight"].T, 3, axis=-1)
        # (B, T, C) -> (B, n_head, T, head_size)
        q, k, v = (t.reshape(B, T, H, C // H).transpose(0, 2, 1, 3) for t in (q, k, v))
        if cache is not None:
            # keys/values of earlier positions go in front of the new ones
            k = np.concatenate([cache[0], k], axis=2)
            v = np.concatenate([cache[1], v], axis=2)
        S = k.shape[2]  # past + current positions
        att = q @ k.transpose(0, 1, 3, 2) / math.sqrt(C // H)
        if key_mask is not None:
            # padding slots are never attended to; -1e9 instead of -inf keeps rows that are all padding finite
            att = np.where(key_mask[:, None, None, :], att, -1e9)
        # the new queries sit at positions S-T..S-1, each sees keys up to itself
        att = np.where(np.tril(np.ones((T, S), dtype=bool), k=S - T), att, -np.inf)
        y = (softmax(att) @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return y @ w[p + "attn.proj.weight"].T, (k, v)

    def __call__(self, idx, caches=None, pos=None, key_mask=None):
        """idx: (B, T) token ids. pos: (B, T) position ids when rows are padded differently;
        key_mask: (B, past + T), False on padding. Returns logits (B, T, vocab) and the new caches."""
        w, (B, T) = self.w, idx.shape
        past = 0 if caches is None else caches[0][0].shape[2]
        assert past + T <= self.cfg.block_size, "sequence longer than block_size"
        if pos is None:
            pos = np.arange(past, past + T)
        x = w["tok_emb.weight"][idx] + w["pos_emb.weight"][pos]
        new_caches = []
        for i in range(self.cfg.n_layer):
            p = f"blocks.{i}."
            a, c = self._attn(layer_norm(x, w[p + "ln1.weight"], w[p + "ln1.bias"]), p, None if caches is None else caches[i], key_mask)
            x = x + a
            h = gelu(layer_norm(x, w[p + "ln2.weight"], w[p + "ln2.bias"]) @ w[p + "mlp.0.weight"].T + w[p + "mlp.0.bias"])
            x = x + h @ w[p + "mlp.2.weight"].T + w[p + "mlp.2.bias"]
            new_caches.append(c)
        logits = layer_norm(x, w["ln_f.weight"], w["ln_f.bias"]) @ w["tok_emb.weight"].T  # tied embeddings
        return logits, new_caches


class Tokenizer:
    def __init__(self, chars):
        self.chars = chars
        self.stoi = {c: i for i, c in enumerate(chars)}

    def unknown(self, s):
        """Characters in s that the model has never seen (the API rejects these)."""
        return sorted(set(s.translate(QUOTES)) - self.stoi.keys())

    def encode(self, s):
        return [self.stoi[c] for c in s.translate(QUOTES)]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)


def load(repo=HF_REPO):
    """Download shakespeare.npz (made by convert.py) from the Hugging Face Hub once, then rebuild model + tokenizer.
    Cached in the temp dir, the only writable place on Vercel."""
    path = os.path.join(tempfile.gettempdir(), repo.replace("/", "--") + ".npz")
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://huggingface.co/{repo}/resolve/main/shakespeare.npz", path + ".part")
        os.replace(path + ".part", path)
    with np.load(path) as f:
        w = {k: f[k] for k in f.files}
    cfg = GPTConfig(**json.loads(str(w.pop("config"))))
    return GPT(cfg, w), Tokenizer(w.pop("chars").tolist())


def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """logits: (vocab,) for the last position -> next token id. temperature=0 is greedy."""
    if temperature == 0:
        return int(logits.argmax())
    logits = logits.astype(np.float64) / temperature  # float64 so the probabilities sum to 1 for rng.choice
    if top_k is not None:
        kth = np.sort(logits)[-min(top_k, logits.size)]
        logits = np.where(logits < kth, -np.inf, logits)
    probs = softmax(logits)
    if top_p is not None:
        order = np.argsort(-probs)
        sorted_probs = probs[order]
        # drop a token once the tokens before it already cover top_p (the top one always stays)
        probs[order[np.cumsum(sorted_probs) - sorted_probs >= top_p]] = 0
        probs /= probs.sum()
    return int(RNG.choice(probs.size, p=probs))


def generate_batch(model, prompts, max_new_tokens, sampling, stop=lambda i: False):
    """Generate for several prompts at once. Yields (i, token, done) for every live row each step.

    Prompts are left padded to the same length. Learned positions must start at 0 for every prompt, so
    each row gets its own position ids, and padding slots are masked out of attention. Row i finishes
    after max_new_tokens[i] tokens or once stop(i) is true, and is dropped from the batch.
    Each len(prompts[i]) + max_new_tokens[i] must fit in block_size (the API enforces this).
    """
    lens = np.array([len(p) for p in prompts])
    pad = lens.max() - lens
    idx = np.array([[0] * n + p for n, p in zip(pad, prompts)])
    slots = np.arange(lens.max())
    valid = slots[None, :] >= pad[:, None]
    pos = np.maximum(slots[None, :] - pad[:, None], 0)
    logits, caches = model(idx, pos=pos, key_mask=valid)

    rows = list(range(len(prompts)))  # original index of each live row
    produced = [0] * len(prompts)
    next_pos = lens.copy()
    while True:
        nxt = np.array([[sample_next(logits[j, -1], **sampling[i])] for j, i in enumerate(rows)])
        keep = []
        for j, i in enumerate(rows):
            produced[i] += 1
            done = produced[i] >= max_new_tokens[i] or stop(i)
            yield i, int(nxt[j, 0]), done
            if not done:
                keep.append(j)
        if not keep:
            return
        if len(keep) < len(rows):
            nxt, valid, next_pos = nxt[keep], valid[keep], next_pos[keep]
            caches = [(k[keep], v[keep]) for k, v in caches]
            rows = [rows[j] for j in keep]
            # drop slots that are padding for every remaining row, so the cache stays within block_size
            trim = int((~valid).sum(axis=1).min())
            valid = valid[:, trim:]
            caches = [(k[:, :, trim:], v[:, :, trim:]) for k, v in caches]
        valid = np.concatenate([valid, np.ones((len(rows), 1), dtype=bool)], axis=1)
        logits, caches = model(nxt, caches=caches, pos=next_pos[:, None], key_mask=valid)
        next_pos += 1
