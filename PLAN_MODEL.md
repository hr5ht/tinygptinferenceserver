# tiny-llm-serve — part 1: model + inference server

Small GPT written from scratch in PyTorch (char-level, ~10M params, trained on the complete works of Shakespeare, Project Gutenberg #100) + the inference server that runs next to it: streaming, KV cache, dynamic batching, load tested. The model lives in `model/model.ipynb` and is trained on Colab (T4). Deployed as a Docker Space on Hugging Face (free CPU tier), and the part 2 gateway calls it at `https://<user>-<space>.hf.space`.

Stack: Python 3.12, PyTorch, Jupyter/Colab, FastAPI, Locust, pytest, Hugging Face Spaces

Not doing: Hugging Face model classes, Redis, user API keys/rate limiting (part 2 owns those), frontend. The only Docker is the minimal Dockerfile that Spaces needs to run FastAPI.

Free Spaces notes: CPU only (2 vCPU), sleeps after ~48h without traffic (part 2's daily cron keeps it awake), and the Space is public, so `MODEL_TOKEN` is the only thing keeping strangers from bypassing the gateway.

## Contract with part 2

Everything except `/health` requires `Authorization: Bearer $MODEL_TOKEN` (shared secret, only the gateway knows it).

- `GET /health` -> `{"status": "ok", "model": "<checkpoint name>"}`
- `POST /v1/generate` body `{prompt, max_tokens, temperature, top_k, top_p, stream}`
  - `stream=false` -> `{"text": "...", "tokens": n}`
  - `stream=true` -> SSE, one `data: {"token": "..."}` per token, then `data: [DONE]`
- `503` + `Retry-After` when the queue is full, `401` on bad token, `422` on bad params
- `GET /metrics` -> Prometheus text

---

## Week 1 - model and training (all in `model/model.ipynb`, checks are assert cells)

### Day 1
- [x] chore: init model folder with gitignore and Shakespeare data
- [x] feat(data): Gutenberg download + cleaning, char-level tokenizer, train/val split, batch sampling

### Day 2
- [x] feat(model): multi-head causal self-attention, pre-norm transformer block
- [x] feat(model): GPT with GPTConfig, position embeddings, weight init, tied embeddings, causal mask check

### Day 3
- [x] feat(train): AdamW loop, warmup + cosine lr, grad clipping, eval-mode train/val loss
- [x] feat(train): checkpoint with config + vocab, loss log to csv, Colab download

### Day 4
- [x] feat(generate): sampling with temperature, top-k, top-p
- [x] feat(model): kv cache with position offset, cached generation, tokens/sec benchmark, cached == uncached check

### Day 5
(Colab, T4: Run all)
- [ ] chore: run the small config first to check loss goes down, then the full config
- [ ] chore(release): upload shakespeare.pt as a github release asset

### Day 6
- [ ] docs: loss curve plot and sample output in the notebook, both runs (small vs full) in a table

### Day 7
(buffer — fix whatever the first training run turns up)

## Week 2 - serving, deploy

### Day 8
- [ ] refactor: copy GPT, sampling and generation code from the notebook into `server/model.py` (the server can't import a notebook)

### Day 9
- [ ] feat(api): fastapi app with /health, env settings, bearer token, model loaded at startup
- [ ] feat(api): POST /v1/generate with validation, run in threadpool, SSE streaming

### Day 10
(hardest day — batching with padding; spill into Day 11 if needed)
- [ ] feat(serve): request queue and background worker with clean shutdown
- [ ] feat(serve): dynamic batching (max_batch / max_wait_ms) with left padding, masked kv cache, per-request stop

### Day 11
- [ ] feat(serve): cancel on client disconnect, request timeout, fail pending futures on worker error
- [ ] feat(serve): 503 + Retry-After when queue is full, graceful drain on shutdown, consistent errors
- [ ] test(serve): api with tiny random-weight model, 401 without token, batched equals single-request, 503 when full

### Day 12
- [ ] feat(metrics): latency histogram, tokens/sec, queue depth and batch size gauges on /metrics
- [ ] perf(serve): preallocated cache buffers, fix max wait ignored when queue full
- [ ] feat(load): locust script with fixed prompts, batching toggle via env, results to csv (run on localhost)

### Day 13
- [ ] chore(deploy): Dockerfile (CPU torch, port 7860) and Space metadata, MODEL_TOKEN as Space secret
- [ ] feat(deploy): fetch checkpoint from release url and load before accepting requests, torch threads = vCPUs
- [ ] chore(load): one locust run against the Space

### Day 14
- [ ] ci: github actions running tests, pinned dependency versions
- [ ] docs: README with API contract, training curves, local vs Space load test numbers
