# tiny-llm-serve — part 2: API gateway on Vercel

Public API for the part 1 model server. Runs on Vercel, owns everything user-facing: API keys, Redis rate limiting, response caching, streaming passthrough, error handling. Holds no model and never imports torch — it forwards to the model server on Hugging Face Spaces.

Stack: Python, FastAPI, httpx, Upstash Redis (via Vercel marketplace), pytest

Not doing: PyTorch, batching/queueing (part 1 owns those), user signup, frontend, Prometheus metrics (serverless instances don't share counters — use Vercel's built-in logs/analytics).

## Env vars

- `MODEL_URL` — `https://<user>-<space>.hf.space`, the part 1 server
- `MODEL_TOKEN` — shared secret sent as `Authorization: Bearer` to the model server
- `API_KEYS` — comma-separated client keys
- `REDIS_URL` — Upstash `rediss://` url

## Public API

- `GET /health` -> gateway status + whether the model server is reachable
- `POST /v1/generate` with `X-API-Key` header, same body as part 1 -> JSON or SSE
- Errors: `401` bad key, `429` rate limited, `502` model offline, `503` model busy (passed through with `Retry-After`), `504` upstream timeout

---

## Week 1

### Day 1
- [ ] chore: init repo with gitignore, requirements, vercel.json
- [ ] feat(api): fastapi app with /health
- [ ] feat(api): settings from env
- [ ] chore(deploy): first deploy to vercel, confirm /health works

### Day 2
- [ ] feat(proxy): POST /v1/generate forwards to model server with httpx
- [ ] feat(api): validate max_tokens and temperature before hitting upstream
- [ ] feat(proxy): send MODEL_TOKEN to upstream
- [ ] fix(proxy): 502 when model server is unreachable, 504 on upstream timeout
- [ ] fix(proxy): return 503 "model waking up" when the Space is asleep or rebuilding, instead of passing through its HTML page

### Day 3
- [ ] feat(proxy): stream SSE through from upstream
- [ ] fix(proxy): close upstream stream when client disconnects
- [ ] chore(deploy): verify streaming on vercel (check function max duration covers the longest generation)

### Day 4
- [ ] feat(auth): api key header check
- [ ] feat(ratelimit): redis token bucket per api key
- [ ] fix(ratelimit): make refill atomic with lua script
- [ ] fix(config): redis url with tls

### Day 5
- [ ] feat(cache): cache greedy completions in redis
- [ ] feat(health): report model reachability with a short timeout
- [ ] chore(deploy): vercel cron hits /health once a day so the Space never hits its 48h sleep
- [ ] refactor: consistent error responses, pass upstream 503 + Retry-After through

### Day 6
- [ ] chore: pytest config with a fake upstream (httpx.MockTransport)
- [ ] test(auth): 401 without a valid key
- [ ] test(ratelimit): 429 once bucket is empty
- [ ] test(cache): sampled requests skip the cache
- [ ] test(proxy): 502 when upstream is down, stream passes tokens through in order

### Day 7
- [ ] test: smoke test script against deployed url
- [ ] ci: run tests on push with github actions
- [ ] chore: pin dependency versions
- [ ] docs: README with architecture diagram (client -> vercel -> hf space) and how to run both parts
