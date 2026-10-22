import asyncio
import json
import os
import queue
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, Histogram, generate_latest
from pydantic import BaseModel, Field, model_validator

from batcher import Batcher, Job
from gpt import HF_REPO, load

MODEL_TOKEN = os.environ["MODEL_TOKEN"]  # required: the URL is public, only the gateway knows this
HF_REPO = os.environ.get("HF_REPO", HF_REPO)
MAX_BATCH = int(os.environ.get("MAX_BATCH", 8))
MAX_WAIT_MS = int(os.environ.get("MAX_WAIT_MS", 20))
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", 64))
TIMEOUT_S = float(os.environ.get("TIMEOUT_S", 60))

BLOCK_SIZE = 256  # the checkpoint's context length; prompt + generated text must fit in it

LATENCY = Histogram("generate_latency_seconds", "Request received to last token sent")


@asynccontextmanager
async def lifespan(app):
    # the model loads before uvicorn accepts any request
    model, app.state.tok = load(HF_REPO)
    assert model.cfg.block_size == BLOCK_SIZE
    app.state.batcher = Batcher(model, MAX_BATCH, MAX_WAIT_MS, MAX_QUEUE)
    app.state.batcher.start()
    yield
    app.state.batcher.stop()


app = FastAPI(title="tinygpt inference server", lifespan=lifespan)


def check_token(authorization: str = Header("")):
    if not secrets.compare_digest(authorization.encode(), f"Bearer {MODEL_TOKEN}".encode()):
        raise HTTPException(401, "invalid or missing bearer token", headers={"WWW-Authenticate": "Bearer"})


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    max_tokens: int = Field(100, ge=1, le=BLOCK_SIZE - 1)
    temperature: float = Field(0.8, ge=0, le=2)
    top_k: int | None = Field(50, ge=1)
    top_p: float | None = Field(None, gt=0, le=1)
    stream: bool = False

    @model_validator(mode="after")
    def fits_context(self):
        if len(self.prompt) + self.max_tokens > BLOCK_SIZE:
            raise ValueError(f"prompt length + max_tokens must be at most {BLOCK_SIZE}")
        return self


@app.get("/health")
def health():
    return {"status": "ok", "model": HF_REPO.split("/")[-1]}


@app.get("/metrics", dependencies=[Depends(check_token)])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/v1/generate", dependencies=[Depends(check_token)])
async def generate(body: GenerateRequest, request: Request):
    tok, batcher = request.app.state.tok, request.app.state.batcher
    if unknown := tok.unknown(body.prompt):
        raise HTTPException(422, f"characters the model has never seen: {''.join(unknown)!r}")

    loop = asyncio.get_running_loop()
    sampling = {"temperature": body.temperature, "top_k": body.top_k, "top_p": body.top_p}
    job = Job(tok.encode(body.prompt), body.max_tokens, sampling, loop)
    try:
        batcher.submit(job)
    except queue.Full:
        raise HTTPException(503, "server busy, retry shortly", headers={"Retry-After": "1"})

    start, deadline = time.perf_counter(), loop.time() + TIMEOUT_S
    if body.stream:
        return StreamingResponse(stream(job, tok, deadline, start), media_type="text/event-stream")

    # ponytail: a non-streaming client that disconnects isn't noticed; max_tokens + TIMEOUT_S bound the waste
    try:
        ids = [t async for t in job.tokens(deadline)]
    except TimeoutError:
        job.cancelled = True
        raise HTTPException(504, "generation timed out")
    LATENCY.observe(time.perf_counter() - start)
    return {"text": tok.decode(ids), "tokens": len(ids)}


def event(data):
    return f"data: {json.dumps(data) if not isinstance(data, str) else data}\n\n"


async def stream(job, tok, deadline, start):
    try:
        async for t in job.tokens(deadline):
            yield event({"token": tok.decode([t])})
        yield event("[DONE]")
        LATENCY.observe(time.perf_counter() - start)
    except TimeoutError:
        yield event({"error": "generation timed out"})
    except Exception as e:
        yield event({"error": str(e)})
    finally:
        job.cancelled = True  # stream over or client gone (Starlette cancels us): the worker drops the row

