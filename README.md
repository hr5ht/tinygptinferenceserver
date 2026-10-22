# tinygpt: a character-level GPT trained on Shakespeare

A 10.77M-parameter GPT written from scratch in PyTorch (no Hugging Face model classes), trained on the complete works of Shakespeare. Everything (data prep, model, KV cache, sampling, training, benchmarks) is in [`model.ipynb`](model.ipynb).

- Trained checkpoint: [hr5ht/tinygpt-shakespeare](https://huggingface.co/hr5ht/tinygpt-shakespeare)
- Run it yourself: [open in Colab](https://colab.research.google.com/github/hr5ht/tinygptinferenceserver/blob/main/model.ipynb) → Runtime → T4 GPU → Run all

## Data

[The Complete Works of William Shakespeare](https://www.gutenberg.org/ebooks/100) from Project Gutenberg: 5.36M characters after stripping the Gutenberg header/footer and normalising curly quotes. Character-level tokenizer, 97 tokens. 90/10 train/val split.

## Model

| | |
|---|---|
| Parameters | 10.77M |
| Layers / heads / embedding | 6 / 6 / 384 |
| Context | 256 characters |
| Architecture | pre-norm transformer, learned position embeddings, tied input/output embeddings, dropout 0.2 |
| Generation | temperature, top-k, top-p sampling, with and without a KV cache |

## Training

4,000 steps, batch 64, AdamW (lr 1e-3, warmup 200 steps then cosine decay to 1e-4), gradient clipping at 1.0, fp16 autocast. **14.6 minutes on a Colab T4.**

| step | train loss | val loss |
|---|---|---|
| 0 | 4.636 | 4.639 |
| 500 | 1.699 | 1.762 |
| 1000 | 1.380 | 1.534 |
| 2000 | 1.216 | 1.412 |
| 3000 | 1.139 | 1.364 |
| 3999 | 1.107 | **1.339** |

Full log: [`losses.csv`](losses.csv). Step 0 loss matches ln(97) ≈ 4.57, the loss of a uniform guess over the vocabulary.

## Sample

Prompt `ROMEO:`, temperature 0.8, top-k 50:

```
ROMEO:
Good morrow, good sir, no more. And all the suns i'
the mean of his life.

ANTONIO.
He's a good friend and sickness to his hand.

CLOTEN.
Why, I think thou hast won thee, but a word and bright him that he saw a way
may be undone me, that any person belongs to the question of an ox of a
cabit? Have I mine own patience, and the rest spirit?
```

It learned the play format (speaker name, line break, dialogue), real character names from different plays, and Shakespearean phrasing ("Good morrow", "thou hast", "i'").

## KV cache benchmark

300 tokens generated from `ROMEO.\n`, batch size 1.

| hardware | no cache | kv cache | speedup |
|---|---|---|---|
| T4 GPU | 225 tok/s | 235 tok/s | 1.0x |
| Apple M1 GPU (MPS) | 60 tok/s | 95 tok/s | 1.6x |
| Apple M1 CPU | 104 tok/s | **417 tok/s** | **4.0x** |

On a GPU, a model this small is limited by kernel launch overhead (~100 small kernels per token), not compute, so processing 1 token instead of up to 256 per step barely changes the time. On CPU, compute scales with the number of tokens processed, so the cache gives a 4x speedup, and the M1 CPU with the cache outruns the T4. That matters because the model is served on CPU.
