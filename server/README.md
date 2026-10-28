# tinygpt inference server

Serves three GPTs written from scratch, with a KV cache, dynamic batching and token streaming: [v1](https://huggingface.co/hr5ht/tinygpt-shakespeare) (10.77M params, 97 characters, 256-token context), [v2](https://huggingface.co/hr5ht/tinygpt-shakespeare-v2) (12.3M, 4096 word pieces, 256) and [v3](https://huggingface.co/hr5ht/tinygpt-shakespeare-v3) (27.8M, the same word pieces, 1024). Inference is plain numpy (no torch), deployed on Vercel; the weights download from the Hub at startup. Source: [github.com/hr5ht/tinygptinferenceserver](https://github.com/hr5ht/tinygptinferenceserver).

Every endpoint except `/health` needs `Authorization: Bearer $MODEL_TOKEN`.

| endpoint | |
|---|---|
| `GET /health` | `{"status": "ok", "models": ["v1", "v2", "v3"]}` |
| `POST /v1/generate` | `{"prompt", "max_tokens"=100, "temperature"=0.8, "top_k"=50, "top_p"=null, "stream"=false, "model"="v1"}` |
| `GET /metrics` | Prometheus: latency, tokens/sec, batch size, queue depth |

`stream=false` returns `{"text": "...", "tokens": n}`. `stream=true` returns server-sent events: `data: {"token": "..."}` per token (a character for v1, a word piece for v2 and v3), then `data: [DONE]`.

Errors: `401` bad token, `422` invalid parameters, a character outside v1's 97-char vocabulary, or prompt + `max_tokens` over the model's context (256 tokens for v1 and v2, 1024 for v3); `503` + `Retry-After` when the queue is full; `504` past `TIMEOUT_S`.
