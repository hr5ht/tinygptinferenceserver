import os
import random

from locust import HttpUser, constant, task

# fixed prompt set so runs are comparable
PROMPTS = ["ROMEO.\n", "JULIET.\n", "HAMLET.\nTo be, or not to be", "KING LEAR.\n", "MACBETH.\n", "First Citizen.\n"]
HEADERS = {"Authorization": f"Bearer {os.environ.get('MODEL_TOKEN', 'dev')}"}
BODY = {"max_tokens": 100, "temperature": 0.8, "top_k": 50}


class Client(HttpUser):
    wait_time = constant(0)  # closed loop: each user sends its next request as soon as the last one finishes

    @task(3)
    def generate(self):
        self.client.post("/v1/generate", json={**BODY, "prompt": random.choice(PROMPTS)}, headers=HEADERS)

    @task(1)
    def stream(self):
        body = {**BODY, "prompt": random.choice(PROMPTS), "stream": True}
        with self.client.post("/v1/generate", json=body, headers=HEADERS, stream=True, name="/v1/generate [stream, time to headers]") as r:
            for _ in r.iter_lines():
                pass
