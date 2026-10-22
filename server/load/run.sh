#!/usr/bin/env bash
# Load test the server on localhost with batching off (MAX_BATCH=1) and on (MAX_BATCH=8).
# Usage: load/run.sh   (USERS=16 DURATION=30s by default). Results: load/results/batch{1,8}_stats.csv
set -euo pipefail
cd "$(dirname "$0")/.."
export MODEL_TOKEN=${MODEL_TOKEN:-dev}
server=""
trap '[ -n "$server" ] && kill "$server" 2>/dev/null || true' EXIT  # never leave a server behind
for batch in 1 8; do
  MAX_BATCH=$batch .venv/bin/uvicorn app:app --port 8000 --log-level warning &
  server=$!
  until curl -sf localhost:8000/health >/dev/null; do sleep 0.5; done
  # locust exits 1 when a stream is cut off at the end of the run; the numbers are still valid
  .venv/bin/locust -f load/locustfile.py --headless -u "${USERS:-16}" -r "${USERS:-16}" -t "${DURATION:-30s}" \
    --host http://localhost:8000 --csv "load/results/batch$batch" --only-summary || true
  kill "$server"
  wait "$server" || true
  server=""
done
rm -f load/results/*_failures.csv load/results/*_exceptions.csv load/results/*_stats_history.csv
