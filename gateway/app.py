import hashlib
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from store import Redis

SERVER_URL = os.environ["SERVER_URL"].rstrip("/")  # the inference server
MODEL_TOKEN = os.environ["MODEL_TOKEN"]  # shared secret with the server; never leaves the gateway
API_KEYS = {k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()}
ANON_PER_MIN = int(os.environ.get("ANON_PER_MIN", 10))  # per IP, no key (the web page)
KEY_PER_MIN = int(os.environ.get("KEY_PER_MIN", 60))  # per API key
CACHE_TTL_S = 24 * 3600

V1_CONTEXT = 256  # v1's context in characters: prompt + generated text must fit
MAX_CONTEXT = 1024  # the longest context of any model (v3), in tokens

log = logging.getLogger("uvicorn.error")


@asynccontextmanager
async def lifespan(app):
    # read timeout above the server's own 60 s generation timeout, so the server's 504 arrives first
    timeout = httpx.Timeout(70, connect=10)
    async with httpx.AsyncClient(base_url=SERVER_URL, headers={"Authorization": f"Bearer {MODEL_TOKEN}"}, timeout=timeout) as client:
        app.state.client, app.state.redis = client, Redis.from_env()
        if app.state.redis is None:
            log.warning("no Upstash Redis configured: rate limits and cache are off")
        yield
        if app.state.redis:
            await app.state.redis.close()


app = FastAPI(title="tinygpt gateway", lifespan=lifespan)
# the chat page is same-origin; CORS lets browser clients on other sites use the API too (no cookies, so any origin is fine)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["Retry-After", "X-Cache"],
)


class GenerateRequest(BaseModel):
    # same limits as the server, so bad input never costs an upstream call. v1's tokens are characters,
    # so its context is checked here; word pieces (v2, v3) are counted by the server, which has the tokenizer
    prompt: str = Field(min_length=1, max_length=4000)
    max_tokens: int = Field(100, ge=1, le=MAX_CONTEXT - 1)
    temperature: float = Field(0.8, ge=0, le=2)
    top_k: int | None = Field(50, ge=1)
    top_p: float | None = Field(None, gt=0, le=1)
    stream: bool = False
    model: Literal["v1", "v2", "v3"] = "v1"

    @model_validator(mode="after")
    def fits_context(self):
        if self.model == "v1" and len(self.prompt) + self.max_tokens > V1_CONTEXT:
            raise ValueError(f"prompt length + max_tokens must be at most {V1_CONTEXT} for v1")
        return self


def caller(request, authorization):
    """Who to rate limit, and how hard: an API key if one is sent (it must be valid), else the client IP."""
    if authorization:
        key = authorization.removeprefix("Bearer ").strip()
        if key not in API_KEYS:
            raise HTTPException(401, "invalid API key", headers={"WWW-Authenticate": "Bearer"})
        return "key:" + hashlib.sha256(key.encode()).hexdigest()[:16], KEY_PER_MIN
    # Vercel sets x-forwarded-for itself, so clients can't spoof it there
    ip = request.headers.get("x-forwarded-for") or (request.client.host if request.client else "unknown")
    ip = ip.split(",")[0].strip()
    return "ip:" + ip, ANON_PER_MIN


def upstream_error(r):
    """The caller's mistakes (422) and back-pressure (503, 504) pass through; anything else is the gateway's problem."""
    if r.status_code in (422, 503, 504):
        try:
            detail = r.json()["detail"]
        except (ValueError, KeyError):
            detail = r.text[:200]
        headers = {"Retry-After": r.headers["retry-after"]} if "retry-after" in r.headers else None
        return HTTPException(r.status_code, detail, headers=headers)
    log.error("inference server returned %s: %s", r.status_code, r.text[:200])
    return HTTPException(502, "inference server error")


async def upstream(send):
    try:
        return await send
    except httpx.TimeoutException:
        raise HTTPException(504, "inference server timed out")
    except httpx.HTTPError:
        raise HTTPException(502, "inference server unreachable")


@app.get("/health")
async def health(request: Request):
    try:
        r = await request.app.state.client.get("/health", timeout=5)
        server = r.json()["status"] if r.status_code == 200 else f"http {r.status_code}"
    except (httpx.HTTPError, ValueError, KeyError):
        server = "unreachable"
    return {"status": "ok", "server": server, "rate_limits": request.app.state.redis is not None}


@app.post("/v1/generate")
async def generate(body: GenerateRequest, request: Request, authorization: str = Header("")):
    client, redis = request.app.state.client, request.app.state.redis
    who, per_min = caller(request, authorization)
    if redis and (wait := await redis.wait(who, per_min)):
        raise HTTPException(429, f"rate limit: {per_min} requests per minute", headers={"Retry-After": str(wait)})

    payload = body.model_dump()
    if body.stream:
        return await stream(client, payload)

    # greedy output depends only on model + prompt + length, so it's cached; sampled output is meant to differ
    cache_key = None
    if body.temperature == 0 and redis:
        cache_key = "cache:" + hashlib.sha256(json.dumps([body.model, body.prompt, body.max_tokens]).encode()).hexdigest()
        if hit := await redis.get(cache_key):
            return JSONResponse(json.loads(hit), headers={"X-Cache": "HIT"})

    r = await upstream(client.post("/v1/generate", json=payload))
    if r.status_code != 200:
        raise upstream_error(r)
    if cache_key:
        await redis.set(cache_key, r.text, CACHE_TTL_S)
    return JSONResponse(r.json(), headers={"X-Cache": "MISS"} if cache_key else None)


async def stream(client, payload):
    r = await upstream(client.send(client.build_request("POST", "/v1/generate", json=payload), stream=True))
    if r.status_code != 200:
        await r.aread()
        await r.aclose()
        raise upstream_error(r)

    async def relay():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        except httpx.HTTPError:
            yield b'data: {"error": "inference server connection lost"}\n\n'
        finally:
            # also runs when our client disconnects: closing upstream lets the server stop generating
            await r.aclose()

    return StreamingResponse(relay(), media_type="text/event-stream")


# the chat page (static/); mounted last so the API routes win
app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")
