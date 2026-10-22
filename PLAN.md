# tiny-llm-serve plan

Small GPT written from scratch in PyTorch (char-level, ~10M params, trained on a TinyStories subset) + a FastAPI backend that serves it: streaming, KV cache, dynamic batching, Redis rate limiting and caching, load tested.

Stack: Python, PyTorch, FastAPI, Redis, Locust, pytest

Not doing: Hugging Face model classes, Docker, auth beyond API keys, frontend.

---

## Week 1 - model

### Day 1
- [ ] chore: init repo with gitignore and requirements
- [ ] chore: package layout

### Day 2
- [ ] feat(data): script to download TinyStories subset
- [ ] feat(tokenizer): char-level tokenizer with encode/decode

### Day 3
- [ ] feat(data): train/val split and random batch sampling

### Day 4
- [ ] feat(model): bigram baseline
- [ ] feat(train): minimal training loop with AdamW
- [ ] feat(train): estimate train/val loss on fixed batches
- [ ] fix(train): switch to eval mode during loss estimation

### Day 5
- [ ] feat(model): single attention head with causal mask

### Day 6
- [ ] feat(model): multi-head attention with output projection
- [ ] feat(model): feed-forward layer and transformer block
- [ ] feat(model): pre-norm residual connections

### Day 7
- [ ] feat(model): GPT with token and position embeddings
- [ ] feat(model): GPTConfig dataclass
- [ ] feat(model): weight init and tied embeddings
- [ ] refactor(model): move attention into its own module

## Week 2 - training, generation, KV cache

### Day 8
- [ ] feat(train): warmup + cosine lr schedule
- [ ] feat(train): gradient clipping
- [ ] feat(train): config through CLI args

### Day 9
- [ ] feat(train): save and load checkpoints with config

### Day 10
(train on colab T4)
- [ ] chore: colab notebook for gpu training
- [ ] feat(train): log losses to csv
- [ ] fix(train): load checkpoint on cpu when cuda isn't available
- [ ] chore(train): seed everything for reproducible runs

### Day 11
- [ ] feat(generate): sampling with temperature
- [ ] feat(generate): top-k and top-p filtering

### Day 12
- [ ] feat(model): kv cache in attention layers
- [ ] fix(model): position offset when using cache

### Day 13
- [ ] feat(generate): cached generation path
- [ ] fix(model): stop cache from growing past block_size

### Day 14
- [ ] feat(bench): tokens/sec with and without cache
- [ ] refactor(generate): share sampling code between both paths

## Week 3 - serving

### Day 15
- [ ] feat(api): fastapi app with /health
- [ ] feat(api): settings from env
- [ ] feat(api): load model once at startup

### Day 16
- [ ] feat(api): POST /generate
- [ ] feat(api): validate max_tokens and temperature
- [ ] fix(api): run generation in threadpool so the event loop isn't blocked

### Day 17
- [ ] feat(api): stream tokens over server-sent events

### Day 18
- [ ] feat(serve): request queue and background worker
- [ ] refactor(api): route /generate through the queue
- [ ] fix(serve): stop worker cleanly on shutdown

### Day 19
- [ ] feat(serve): batch requests up to max_batch or max_wait_ms
- [ ] feat(model): left padding and attention mask for batched prompts
- [ ] feat(generate): batched generation with per-request stop
- [ ] fix(model): mask padded positions in kv cache
- [ ] chore(serve): make batch size and wait time configurable

### Day 20
- [ ] fix(generate): finished sequences kept generating inside a batch

### Day 21
- [ ] feat(api): cancel generation when client disconnects
- [ ] feat(api): request timeout
- [ ] fix(serve): futures left pending after worker error

## Week 4 - redis, load testing, deploy, tests

### Day 22
- [ ] feat(auth): api key header check
- [ ] feat(ratelimit): redis token bucket per api key
- [ ] fix(ratelimit): make refill atomic with lua script

### Day 23
- [ ] feat(cache): cache greedy completions in redis
- [ ] feat(metrics): latency histogram and tokens/sec
- [ ] feat(metrics): queue depth and batch size gauges
- [ ] feat(api): /metrics endpoint

### Day 24
- [ ] feat(load): locust script for generate and stream
- [ ] chore(load): fixed prompt set for load tests

### Day 25
- [ ] feat(bench): toggle batching through env for comparison runs
- [ ] perf(serve): reuse preallocated cache buffers
- [ ] fix(serve): max wait ignored when queue already full
- [ ] chore(bench): save load test results to csv

### Day 26
- [ ] feat(serve): return 503 when queue is full
- [ ] feat(api): graceful shutdown drains the queue
- [ ] refactor: consistent error responses

### Day 27
- [ ] chore(deploy): render config and start command
- [ ] feat(model): fetch checkpoint from release url on startup
- [ ] fix(config): redis url with tls
- [ ] fix(api): load model before accepting requests on cold start

### Day 28
- [ ] chore: pytest config and fixtures
- [ ] test(tokenizer): encode/decode round trip
- [ ] test(data): batch shapes and target offset by one
- [ ] test(model): future tokens don't change earlier outputs
- [ ] test(generate): cached and uncached greedy outputs match
- [ ] test(generate): top-k never samples outside k

### Day 29
- [ ] test(api): generate and stream with a tiny random-weight model
- [ ] test(serve): batched output equals single-request output
- [ ] test(ratelimit): 429 once bucket is empty
- [ ] test(cache): sampled requests skip the cache

### Day 30
- [ ] test: smoke test script against deployed url
- [ ] ci: run tests on push with github actions
- [ ] chore: pin dependency versions
