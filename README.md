# tinygpt: a character-level GPT trained on Shakespeare

A 10.77M-parameter GPT written from scratch in PyTorch (no Hugging Face model classes), trained on the complete works of Shakespeare. Everything (data prep, model, KV cache, sampling, training, benchmarks) is in [`model/model.ipynb`](model/model.ipynb).

- Trained checkpoint: [hr5ht/tinygpt-shakespeare](https://huggingface.co/hr5ht/tinygpt-shakespeare)
- Run it yourself: [open in Colab](https://colab.research.google.com/github/hr5ht/tinygptinferenceserver/blob/main/model/model.ipynb) → Runtime → T4 GPU → Run all

## Repository layout

```
model/
  model.ipynb   data prep, model, KV cache, training, benchmarks (saved T4 outputs)
  losses.csv    train/val loss every 250 steps
server/                inference server (runs on Vercel)
  gpt.py               numpy port of the model (no torch), KV cache, batched generation; weights from the Hub
  convert.py           one-off: torch checkpoint -> shakespeare.npz
  batcher.py           request queue + worker thread that batches requests
  app.py               FastAPI: /health, /v1/generate (JSON or streaming), /metrics
  load/                locust load test, batching on vs off, results as csv
```

The checkpoint (`shakespeare.pt`) and the raw text are not in the repo: the checkpoint lives on the Hugging Face Hub and the notebook downloads the text itself.

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

Full log: [`model/losses.csv`](model/losses.csv). Step 0 loss matches ln(97) ≈ 4.57, the loss of a uniform guess over the vocabulary.

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

## Inference server

[`server/`](server/) serves the model over HTTP with FastAPI on Vercel. torch is 715 MB unzipped on Linux, over Vercel's 500 MB function limit, so `gpt.py` reimplements inference in numpy (57 MB). [`convert.py`](server/convert.py) turns the torch checkpoint into `shakespeare.npz` on the Hub. The numpy logits match torch to within 1e-5, and greedy outputs are identical.

- `POST /v1/generate` takes `{"prompt", "max_tokens", "temperature", "top_k", "top_p", "stream"}` and returns `{"text", "tokens"}`, or server-sent events with one token each when `stream` is true.
- **Dynamic batching:** requests wait in a bounded queue. A worker thread takes the first request, waits up to 20 ms for up to 8 more, and generates them as one batch: left-padded prompts, per-row positions and an attention mask over the padding, with the KV cache shared across the batch. A row leaves the batch as soon as it finishes or its client disconnects.
- **Back-pressure:**
  - a full queue returns `503` with `Retry-After`;
  - generation past `TIMEOUT_S` returns `504`;
  - on shutdown the server finishes the requests already queued.
- **Auth:** every endpoint except `/health` needs `Authorization: Bearer $MODEL_TOKEN`. The URL is public, and only the gateway knows the token.
- **Metrics:** `/metrics` exposes Prometheus metrics: latency, tokens/sec, batch size and queue depth.

Run it locally:

```
cd server
python3.12 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
MODEL_TOKEN=dev .venv/bin/uvicorn app:app
curl -X POST localhost:8000/v1/generate -H 'Authorization: Bearer dev' -H 'content-type: application/json' \
  -d '{"prompt": "ROMEO.\n", "max_tokens": 100}'
```

### Load test: batching off vs on

[`server/load/run.sh`](server/load/run.sh) runs [locust](https://locust.io) against the server with 16 concurrent users for 30 s, each sending its next request as soon as the last one returns (100 tokens per request). It runs once with `MAX_BATCH=1` (batching off) and once with `MAX_BATCH=8`, on an Apple M1 CPU. It was measured on the earlier torch build. The numpy port generates at the same speed or faster: 462 tok/s single and 1568 tok/s with a batch of 8. Raw results: [`server/load/results/`](server/load/results/).

| | batching off | batching on (≤ 8) |
|---|---|---|
| requests/s | 4.8 | **14.8** (3.1x) |
| tokens/s | ~480 | **~1480** |
| median latency (non-streaming) | 4.0 s | **1.3 s** |
| failures | 0 | 0 |

One request at a time spends each step's matrix multiplies on a single row. A batch of 8 does 8 rows for little more than the cost of one, so throughput triples and requests stop waiting behind each other.

### Deploy (Vercel)

`server/` deploys as-is. Vercel finds the FastAPI `app` in `app.py`, installs `requirements.txt` and runs it as one function, with no Docker and no config file. `.vercelignore` keeps tests, the load test and local weights out of the bundle.

```
cd server
.venv/bin/python convert.py && hf upload hr5ht/tinygpt-shakespeare shakespeare.npz   # once
vercel link
vercel env add MODEL_TOKEN production
vercel deploy --prod
```

Each new instance downloads the 43 MB of weights from the Hub on cold start, which takes a few seconds. Metrics are per instance.
