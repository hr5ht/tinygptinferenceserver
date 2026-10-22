import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field

from prometheus_client import Counter, Gauge, Histogram

from gpt import generate_batch

log = logging.getLogger("uvicorn.error")

DONE = None  # sent after a job's last token
STOP = object()  # queued by Batcher.stop() behind the remaining jobs

TOKENS = Counter("generated_tokens", "Tokens generated")
BATCH_SIZE = Histogram("batch_size", "Requests generated together", buckets=(1, 2, 4, 8, 16, 32))
TOKENS_PER_SEC = Gauge("tokens_per_second", "Throughput of the most recent batch")
QUEUE_DEPTH = Gauge("queue_depth", "Requests waiting for a batch")


@dataclass
class Job:
    ids: list
    max_tokens: int
    sampling: dict
    loop: asyncio.AbstractEventLoop
    out: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancelled: bool = False  # set by the API when the client leaves or times out; the worker drops the row

    def emit(self, item):
        """Called from the worker thread: hand a token (or DONE, or an exception) to the request's event loop."""
        self.loop.call_soon_threadsafe(self.out.put_nowait, item)

    async def tokens(self, deadline):
        """Yield token ids as the worker produces them. Raises TimeoutError at the deadline (loop time)."""
        loop = asyncio.get_running_loop()
        while (item := await asyncio.wait_for(self.out.get(), deadline - loop.time())) is not DONE:
            if isinstance(item, Exception):
                raise item
            yield item


class Batcher:
    """Runs the model on its own thread so the event loop stays free.

    Requests wait in a bounded queue. The worker takes the first one, waits up to max_wait_ms for more
    (up to max_batch), then generates for all of them together. max_batch=1 turns batching off.
    """

    # ponytail: requests that arrive mid-batch wait for the whole batch to finish; continuous batching
    # (admitting new rows into a running batch) is the upgrade if long generations starve short ones

    def __init__(self, model, max_batch, max_wait_ms, max_queue):
        self.model, self.max_batch, self.max_wait = model, max_batch, max_wait_ms / 1000
        self.q = queue.Queue(max_queue)
        self.thread = threading.Thread(target=self._loop, name="batcher", daemon=True)
        QUEUE_DEPTH.set_function(self.q.qsize)

    def start(self):
        self.thread.start()

    def submit(self, job):
        self.q.put_nowait(job)  # raises queue.Full when the server is saturated -> 503

    def stop(self, timeout=30):
        """Finish the jobs already queued, then exit."""
        try:
            self.q.put(STOP, timeout=timeout)
        except queue.Full:
            pass
        self.thread.join(timeout)

    def _next_batch(self):
        batch, deadline = [], None
        while len(batch) < self.max_batch:
            try:
                job = self.q.get(timeout=None if not batch else max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                break
            if job is STOP:
                return batch, True
            if job.cancelled:  # client left while waiting in the queue
                continue
            if not batch:
                deadline = time.monotonic() + self.max_wait
            batch.append(job)
        return batch, False

    def _loop(self):
        stopping = False
        while not stopping:
            batch, stopping = self._next_batch()
            if batch:
                self._run(batch)

    def _run(self, batch):
        BATCH_SIZE.observe(len(batch))
        start, n = time.perf_counter(), 0
        try:
            for i, tok, done in generate_batch(
                self.model,
                [j.ids for j in batch],
                [j.max_tokens for j in batch],
                [j.sampling for j in batch],
                stop=lambda i: batch[i].cancelled,
            ):
                batch[i].emit(tok)
                if done:
                    batch[i].emit(DONE)
                n += 1
        except Exception as e:
            log.exception("batch failed")
            for j in batch:
                j.emit(e)  # every waiting request gets the error instead of hanging
            return
        TOKENS.inc(n)
        TOKENS_PER_SEC.set(n / (time.perf_counter() - start))
