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
from concurrent.futures import ThreadPoolExecutor

import requests

logger = logging.getLogger("batch_llm")


def _extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response; {} on failure."""
    try:
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1]
        return json.loads(text.strip())
    except Exception:
        return {}


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
        try:
            r = requests.post(f"{self.base_url}/chat/completions",
                              json=payload, timeout=120)
            r.raise_for_status()
            return _extract_json(r.json()["choices"][0]["message"]["content"])
        except Exception as e:
            logger.warning(f"batch call failed: {e}")
            return {}

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

    def generate_json_batch(self, prompts):
        return [self.llm.generate_json(p, self.schema, system_message=self.system)
                for p in prompts]
