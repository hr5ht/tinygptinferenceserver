"""Upstash Redis over its REST API: plain HTTPS, so it works from serverless functions without a connection pool."""
import logging
import os
import time

import httpx

log = logging.getLogger("uvicorn.error")

# token bucket, atomic inside Redis: refill by the time elapsed, then take one token if there is one.
# Returns 0 when the request may go ahead, else the seconds until a token is free.
BUCKET = """
local cap, rate, now = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3])
local b = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = math.min(cap, (tonumber(b[1]) or cap) + (now - (tonumber(b[2]) or now)) * rate)
local wait = 0
if tokens >= 1 then tokens = tokens - 1 else wait = math.ceil((1 - tokens) / rate) end
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], math.ceil(cap / rate))  -- an idle bucket is full again by then, so drop it
return wait
"""


class Redis:
    def __init__(self, url, token):
        self.http = httpx.AsyncClient(base_url=url, headers={"Authorization": f"Bearer {token}"}, timeout=5)

    @classmethod
    def from_env(cls):
        """None when Redis isn't configured (local dev): no rate limits, no cache."""
        # Vercel's Upstash integration names them KV_REST_API_*
        url = os.environ.get("UPSTASH_REDIS_REST_URL") or os.environ.get("KV_REST_API_URL")
        token = os.environ.get("UPSTASH_REDIS_REST_TOKEN") or os.environ.get("KV_REST_API_TOKEN")
        return cls(url, token) if url and token else None

    async def cmd(self, *args, default=None):
        # ponytail: Redis down means no limits and no cache (fail open), not a broken API
        try:
            r = await self.http.post("", json=[str(a) for a in args])
            r.raise_for_status()
            return r.json()["result"]
        except (httpx.HTTPError, KeyError):
            log.exception("redis %s failed", args[0])
            return default

    async def wait(self, who, per_min):
        """Seconds `who` must wait before another request, 0 if it may go now."""
        return await self.cmd("EVAL", BUCKET, 1, f"rl:{who}", per_min, per_min / 60, time.time(), default=0)

    async def get(self, key):
        return await self.cmd("GET", key)

    async def set(self, key, value, ttl):
        await self.cmd("SET", key, value, "EX", ttl)

    async def close(self):
        await self.http.aclose()
