"""
batch_llm.py — send MANY prompts to a local LLM at once.

Why batching matters: the walk-forward analysis makes
   #dates x (1 + n_rounds) x #stocks
LLM calls. Doing them one-at-a-time is hopelessly slow. A vLLM server processes
many requests concurrently (continuous batching on the GPU), so the trick on
the client side is simply to *fire all of a round's prompts at once* instead of
looping. Within one round every stock agent is independent, so this is safe.

Two interchangeable clients, both exposing the same method:

    generate_json_batch(prompts) -> list[dict]      # one parsed dict per prompt

  HttpBatchClient        talks to an OpenAI-compatible endpoint (vLLM, Ollama)
                         and sends the prompts concurrently with a thread pool.
                         This is the production path on a GPU node.

  SequentialBatchClient  wraps FAgent's in-process LLMClient and just loops.
                         No speed-up, but it needs no server — used to test the
                         batched pipeline on a CPU login node with the 0.5B model.

Both return `{}` for any prompt whose response can't be parsed; the caller
(StockAgent.message_from_data) then falls back to the rule-based answer.
"""

import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger("batch_llm")


def _extract_json(text: str) -> dict:
    """
    Pull the first JSON object out of a model response.

    On failure we DON'T return a bare {} — we return the raw text and the parse
    error under reserved "_raw"/"_error" keys, so the caller can log exactly WHY
    an agent fell back to the rule (the whole point of watching fallbacks). The
    dict still has no "score" key, so every existing fallback check keeps working.
    """
    try:
        cleaned = text
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1]
        return json.loads(cleaned.strip())
    except Exception as e:
        return {"_raw": text, "_error": f"json-parse: {e}"}


class HttpBatchClient:
    """Concurrent client for an OpenAI-compatible server (vLLM / Ollama)."""

    def __init__(self, base_url, model, system, schema,
                 max_workers=32, temperature=0.3, max_tokens=256):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.system = system
        self.schema = schema
        self.max_workers = max_workers
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Token/latency accounting. vLLM returns an exact `usage` block per call,
        # so a real run reports measured tokens rather than an estimate. Appended
        # from worker threads, hence the lock.
        self._acct_lock = threading.Lock()
        self._acct = []                    # (prompt_tok, completion_tok, seconds)

    def pop_usage(self):
        """
        Drain and return this batch's per-call accounting.

        Returns (prompt_tokens, completion_tokens, latencies, exact). `exact` is
        False if any call came back without a usage block, so the caller can
        label the number honestly.
        """
        with self._acct_lock:
            acct, self._acct = self._acct, []
        pt = sum(a[0] for a in acct)
        ct = sum(a[1] for a in acct)
        lat = [a[2] for a in acct]
        exact = bool(acct) and all(a[0] is not None for a in acct)
        return (pt or 0), (ct or 0), lat, exact

    def _format(self, prompt: str) -> str:
        """Append the JSON schema instruction, mirroring FAgent's generate_json."""
        return (f"{prompt}\n\nReturn ONLY a valid JSON object with this schema:\n"
                f"```json\n{json.dumps(self.schema, indent=2)}\n```")

    def _one_call(self, prompt: str) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": self.system},
                         {"role": "user", "content": self._format(prompt)}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        t0 = time.time()
        try:
            r = requests.post(f"{self.base_url}/chat/completions",
                              json=payload, timeout=120)
            r.raise_for_status()
            body = r.json()
            usage = body.get("usage") or {}
            self._account(usage.get("prompt_tokens"),
                          usage.get("completion_tokens"), time.time() - t0)
            return _extract_json(body["choices"][0]["message"]["content"])
        except Exception as e:
            # A failed call still consumed wall time; record it so throughput
            # numbers reflect what the job actually experienced.
            self._account(0, 0, time.time() - t0)
            logger.warning(f"batch call failed: {e}")
            return {"_raw": "", "_error": f"http: {e}"}

    def _account(self, prompt_tokens, completion_tokens, seconds):
        with self._acct_lock:
            self._acct.append((prompt_tokens, completion_tokens or 0, seconds))

    def generate_json_batch(self, prompts):
        """Send all prompts concurrently; preserve input order in the output."""
        if not prompts:
            return []
        workers = min(self.max_workers, len(prompts))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._one_call, prompts))


class SequentialBatchClient:
    """Loop over prompts using an in-process LLMClient. For CPU testing only."""

    def __init__(self, llm_client, system, schema):
        self.llm = llm_client
        self.system = system
        self.schema = schema
        self._acct = []

    def pop_usage(self):
        """
        Same interface as HttpBatchClient, but this client's LLMClient returns
        only parsed JSON — no usage block — so token counts are left to the
        caller (exact=False) and only latency is real.
        """
        acct, self._acct = self._acct, []
        return 0, 0, list(acct), False

    def generate_json_batch(self, prompts):
        out = []
        for p in prompts:
            t0 = time.time()
            out.append(self.llm.generate_json(p, self.schema,
                                              system_message=self.system))
            self._acct.append(time.time() - t0)
        return out
