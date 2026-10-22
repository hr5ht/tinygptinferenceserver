"""One-off: turn the training checkpoint (a torch pickle) into shakespeare.npz, so the server runs without torch.

    .venv/bin/python convert.py   (needs torch, from requirements-dev.txt)
    hf upload hr5ht/tinygpt-shakespeare shakespeare.npz
"""
import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download

ckpt = torch.load(hf_hub_download("hr5ht/tinygpt-shakespeare", "shakespeare.pt"), map_location="cpu")
# the causal mask buffers are rebuilt on the fly, and head.weight is tok_emb.weight (tied)
w = {k: v.numpy() for k, v in ckpt["model"].items() if not k.endswith("attn.mask") and k != "head.weight"}
np.savez("shakespeare.npz", **w, config=np.array(json.dumps(ckpt["config"])), chars=np.array(list(ckpt["chars"])))
