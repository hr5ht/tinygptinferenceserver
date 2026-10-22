# tinygpt inference server

Serves [hr5ht/tinygpt-shakespeare](https://huggingface.co/hr5ht/tinygpt-shakespeare), a 10.77M-param character-level GPT written from scratch, with a KV cache, dynamic batching and token streaming. Inference is plain numpy (no torch), deployed on Vercel; the weights download from the Hub at startup. Source: [github.com/hr5ht/tinygptinferenceserver](https://github.com/hr5ht/tinygptinferenceserver).

Every endpoint except `/health` needs `Authorization: Bearer $MODEL_TOKEN`.

| endpoint | |
|---|---|
| `GET /health` | `{"status": "ok", "model": "tinygpt-shakespeare"}` |
| `POST /v1/generate` | `{"prompt", "max_tokens"=100, "temperature"=0.8, "top_k"=50, "top_p"=null, "stream"=false}` |
| `GET /metrics` | Prometheus: latency, tokens/sec, batch size, queue depth |

`stream=false` returns `{"text": "...", "tokens": n}`. `stream=true` returns server-sent events: `data: {"token": "..."}` per character, then `data: [DONE]`.

Errors: `401` bad token, `422` invalid parameters, a character outside the model's 97-char vocabulary, or prompt + `max_tokens` over 256; `503` + `Retry-After` when the queue is full; `504` past `TIMEOUT_S`.
