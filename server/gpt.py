import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from torch.nn import functional as F

HF_REPO = "hr5ht/tinygpt-shakespeare"

# the training text had curly quotes turned into straight ones, so prompts get the same treatment
QUOTES = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"'})


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool)))

    def forward(self, x, cache=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_head, T, head_size)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        if cache is not None:
            # keys/values of earlier positions go in front of the new ones
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        S = k.size(2)  # past + current positions
        att = (q @ k.transpose(-2, -1)) / math.sqrt(k.size(-1))
        # the new queries sit at positions S-T..S-1, each sees keys up to itself
        att = att.masked_fill(~self.mask[S - T:S, :S], float("-inf"))
        att = self.attn_drop(F.softmax(att, dim=-1))
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y)), (k, v)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, cache=None):
        a, cache = self.attn(self.ln1(x), cache)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, cache


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight  # tied embeddings
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None, caches=None):
        B, T = idx.shape
        past = 0 if caches is None else caches[0][0].size(2)
        assert past + T <= self.cfg.block_size, "sequence longer than block_size"
        pos = torch.arange(past, past + T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        new_caches = []
        for i, block in enumerate(self.blocks):
            x, c = block(x, None if caches is None else caches[i])
            new_caches.append(c)
        logits = self.head(self.ln_f(x))
        loss = None if targets is None else F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, new_caches


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


def load(repo=HF_REPO, device="cpu"):
    """Download the checkpoint from the Hugging Face Hub (cached after the first call) and rebuild model + tokenizer."""
    ckpt = torch.load(hf_hub_download(repo, "shakespeare.pt"), map_location=device)
    model = GPT(GPTConfig(**ckpt["config"])).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model, Tokenizer(ckpt["chars"])


def sample_next(logits, temperature=1.0, top_k=None, top_p=None):
    """logits: (B, vocab) for the last position -> (B, 1) next token ids. temperature=0 is greedy."""
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        # drop a token once the tokens before it already cover top_p (the top one always stays)
        sorted_logits[probs.cumsum(dim=-1) - probs >= top_p] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)


@torch.no_grad()
def generate(model, idx, max_new_tokens, **sampling):
    """Re-runs the whole window every step. Kept as the reference the cached path is tested against."""
    for _ in range(max_new_tokens):
        logits, _, _ = model(idx[:, -model.cfg.block_size:])
        idx = torch.cat([idx, sample_next(logits[:, -1], **sampling)], dim=1)
    return idx


@torch.no_grad()
def generate_cached(model, idx, max_new_tokens, **sampling):
    """Runs the prompt once, then feeds one token at a time and reuses the KV cache."""
    bs = model.cfg.block_size
    logits, _, caches = model(idx[:, -bs:])
    for _ in range(max_new_tokens):
        nxt = sample_next(logits[:, -1], **sampling)
        idx = torch.cat([idx, nxt], dim=1)
        if caches[0][0].size(2) == bs:
            # learned absolute positions can't slide, so a full cache is rebuilt from the last half window
            # (output differs from `generate` past block_size); rotary embeddings would allow a rolling cache
            logits, _, caches = model(idx[:, -(bs // 2):])
        else:
            logits, _, caches = model(nxt, caches=caches)
    return idx
