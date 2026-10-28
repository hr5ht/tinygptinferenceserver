# tinygpt: a character-level GPT trained on Shakespeare

A 10.77M-parameter GPT written from scratch in PyTorch (no Hugging Face model classes), trained on the complete works of Shakespeare. Everything (data prep, model, KV cache, sampling, training, benchmarks) is in [`model/model.ipynb`](model/model.ipynb).

- Trained checkpoint: [hr5ht/tinygpt-shakespeare](https://huggingface.co/hr5ht/tinygpt-shakespeare)
- Run it yourself: [open in Colab](https://colab.research.google.com/github/hr5ht/tinygptinferenceserver/blob/main/model/model.ipynb) → Runtime → T4 GPU → Run all
- **Try it: [tinygpt-gateway.vercel.app](https://tinygpt-gateway.vercel.app)**: play a scene with the model. API docs: [/docs](https://tinygpt-gateway.vercel.app/docs).

## Architecture

```
browser (chat page) ─┐
                     ├─HTTPS─▶ gateway (Vercel) ──Bearer MODEL_TOKEN──▶ server (Vercel) ──▶ numpy GPT + KV cache
curl / your code ────┘            │                                        │ on cold start
                                  └── Upstash Redis                        ▼
                                      rate limits, cache          Hugging Face Hub: weights
```

- **model:** trained once in a notebook on a T4. The weights live on the Hugging Face Hub.
- **server:** runs the model: KV cache, dynamic batching, streaming, back-pressure. Only the gateway can call it.
- **gateway:** the public front door:
  - API keys and per-caller rate limits;
  - a cache for repeated requests;
  - clean errors;
  - the chat page at `/`, written in plain HTML, CSS and JavaScript.

## Repository layout

```
model/
  model.ipynb   data prep, model, KV cache, training, benchmarks (saved T4 outputs)
  losses.csv    train/val loss every 250 steps
  v3.ipynb      v3: 27.8M params from scratch (TinyStories, Shakespeare, dialogue), sized for Vercel's free plan
server/                inference server (runs on Vercel)
  gpt.py               numpy port of the model (no torch), KV cache, batched generation; weights from the Hub
  convert.py           one-off: torch checkpoint -> shakespeare.npz
  batcher.py           request queue + worker thread that batches requests
  app.py               FastAPI: /health, /v1/generate (JSON or streaming), /metrics
  load/                locust load test, batching on vs off, results as csv
gateway/               public API (runs on Vercel)
  app.py               FastAPI: validation, API keys, rate limits, cache, CORS, proxy to the server (JSON or streaming)
  store.py             Upstash Redis over REST: token bucket (Lua), cache
  static/              the chat page (plain HTML/CSS/JS): index.html, style.css, app.js
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

## v3: a bigger model for the chat

Played as a chat, v1 and v2 keep the play format but barely answer what they are told. On real exchanges from the plays, Shakespeare's actual reply is only 2–3% more likely after the line it answered than after a random line. Fine-tuning v2 on dialogue didn't change that: in trial runs its answers got no better and that gap stayed flat, because those answers were already in its training text. The chat page also fits the scene into the context, which leaves v2 only the last 2–3 exchanges.

[`model/v3.ipynb`](model/v3.ipynb) ([open in Colab](https://colab.research.google.com/github/hr5ht/tinygptinferenceserver/blob/main/model/v3.ipynb) → Runtime → T4 GPU → Run all) trains v3 from scratch:
- **Size, for Vercel's free plan** (1 vCPU, 2 GB): 8 layers × 512, 27.8M params (111 MB), 1024-token context. With the server's numpy code on one thread, a chat turn at full context takes ~3.4 s (v2: 0.7 s), and 8 at once take ~43 s, inside the server's 60 s timeout. Bigger costs more than it adds: 10 × 512 is ~10% slower, 12 × 640 ~1.8×.
- **Data:** ~200M tokens over 6000 steps. Most of the run is TinyStories for plain English, with 5% Shakespeare. The last 15%, while the learning rate winds down, is half Shakespeare, and half of that is dialogue in the chat page's format with the loss on the answers only. It uses v2's tokenizer, so the server needs no new one.
- **Free Colab:** with Google Drive mounted, it caches the prepared data and saves a checkpoint every 500 steps; after a disconnect, Run all resumes. The first eval line prints the measured speed and the time left.
- **Checks:** v3 against v2 on held-out scenes (answers as served, answers with the same 256 tokens, and line gain), the two playing the same scenes, and the export (`v3.npz`) loaded with the server's own `gpt.py`. The run stops before the upload unless v3 answers better than v2.

The last cell uploads `v3.npz` to `<your account>/tinygpt-shakespeare-v3` on the Hub. To serve it, the server needs a `MODELS` entry and a per-model context check instead of the fixed `BLOCK_SIZE = 256`. The same goes for the gateway's validation and `app.js` (a prompt budget per model), plus a `v3` item in the model menu.

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

## Gateway

[`gateway/`](gateway/) is what the public talks to. It serves the chat page at `/` and exposes the same `POST /v1/generate` as the server, with CORS open so browsers on other sites can call it too. It never exposes the server's `MODEL_TOKEN`.

- **Callers:**
  - without a key, the caller is identified by IP (10 requests/min); the chat page works this way.
  - with `Authorization: Bearer <api key>`, the key must be in `API_KEYS` and gets 60 requests/min. A wrong key is a `401`, not a fallback to anonymous.
- **Rate limits:** each caller has a token bucket in Upstash Redis. Refill and take run in one Lua script, so they're atomic across serverless instances. A caller over the limit gets `429` with `Retry-After`. Redis is reached over HTTPS, so there are no connections to pool in serverless functions.
- **Cache:** greedy requests (`temperature: 0`) always produce the same output, so they're cached for a day (`X-Cache: HIT`). Sampled requests are never cached.
- **Errors:**
  - the gateway rejects bad input itself (`422`), without calling the server;
  - the server's `422` (unknown character), `503` (busy) and `504` pass through;
  - any other upstream failure becomes `502`.
- **Streaming:** server-sent events are relayed byte for byte. When the client disconnects, the upstream connection closes too, so the server stops generating.
- **Redis unavailable:** if it isn't configured, or it's down, limits and cache are off and the API keeps working.

```
curl -X POST https://tinygpt-gateway.vercel.app/v1/generate -H 'content-type: application/json' \
  -d '{"prompt": "ROMEO.\n", "max_tokens": 200}'
```

Run locally against the deployed server, with no Redis, so no limits or cache:

```
cd gateway
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
SERVER_URL=https://tinygpt-server.vercel.app MODEL_TOKEN=... .venv/bin/uvicorn app:app --port 8001
```

Deploy: a Vercel project with Root Directory `gateway`. Project settings:

| variable | value |
|---|---|
| `SERVER_URL` | the server's URL |
| `MODEL_TOKEN` | same value as the server's |
| `API_KEYS` | comma-separated keys you hand out |
| Upstash | add Upstash Redis from Vercel's Storage tab; it sets `KV_REST_API_URL` and `KV_REST_API_TOKEN` |

## Chat page

[`gateway/static/`](gateway/static/) is the chat at [tinygpt-gateway.vercel.app](https://tinygpt-gateway.vercel.app). It's plain HTML, CSS and JavaScript with no build step, served by the gateway itself. The model only continues text, so a chat is played as a scene:
- you speak as one character (e.g. ROMEO) and the model replies as another (JULIET);
- the conversation is written out in the format the model was trained on: `ROMEO.\n<speech>\n\nJULIET.\n`;
- the model continues the script.

How it fits the model's limits:
- **Context:** older speeches are dropped from the front of the script, so prompt plus reply always fit in the 256-character context.
- **Ending a reply:** a reply ends at the first blank line, where the model would start the next speaker. The app stops reading and hangs up there. The gateway closes the upstream connection and the server stops generating, so a reply costs only the characters it shows.

Seven scenes are included, from *Romeo and Juliet* to *King Lear*. Every character name occurs at least 30 times in the training text, so the model has learned each one's speech format.

The layout follows the usual LLM chat app: scenes in a collapsible sidebar (a drawer on phones), each keeping its own conversation; the model picker and sampling settings in the header; suggested lines and the composer at the bottom. It ships a dark theme, with every color a CSS variable so a light theme only overrides that block.

Locally it's served at `http://localhost:8001` by the gateway command above.
