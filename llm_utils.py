"""
LLM utilities for FINCON system — open-source / zero-cost backends.

No commercial API is required. Three backends are supported, selected via the
LLM_BACKEND environment variable (or the `backend` constructor argument):

  - "none" (default): no LLM at all. Components that accept an LLM client
    (e.g. the over-episode CVRF conceptualizer) fall back to their
    deterministic rule-based implementations. Zero cost.

  - "openai-compatible": any local OpenAI-compatible server — Ollama
    (default, http://localhost:11434/v1), vLLM (http://localhost:8000/v1),
    llama.cpp server, LM Studio. Run open-weight models such as
    qwen2.5:14b, llama3.1:8b, or deepseek-r1:14b locally at zero token cost.
        Example:  ollama serve &&  ollama pull qwen2.5:14b
                  export LLM_BACKEND=openai-compatible LLM_MODEL=qwen2.5:14b

  - "transformers": direct in-process HuggingFace inference on this machine's
    GPUs/CPUs (requires `transformers` and enough VRAM for the chosen model).
        Example:  export LLM_BACKEND=transformers \
                         LLM_MODEL=Qwen/Qwen2.5-7B-Instruct

Embeddings use sentence-transformers (open-source, local) when installed and
otherwise fall back to a deterministic hashing embedding, so memory retrieval
keeps working offline.

Note: FinCon does NOT fine-tune the LLM. All model weights stay frozen; the
system improves through verbal reinforcement — prompt/belief updates applied
via textual gradient descent (see risk_control/over_episode.py).
"""

import hashlib
import json
import logging
import math
import os
import time

logger = logging.getLogger("llm_utils")

DEFAULT_BASE_URLS = {
    "ollama": "http://localhost:11434/v1",
    "vllm": "http://localhost:8000/v1",
}


class LLMClient:
    """
    Client for open-source LLM backends with a consistent interface.

    Zero-cost by construction: it never calls a commercial API.
    """

    def __init__(self, model_name=None, api_key=None, temperature=0.3,
                 max_tokens=1024, backend=None, base_url=None):
        """
        Initialize LLM client.

        Args:
            model_name (str, optional): Model to use. For "openai-compatible"
                this is the served model tag (e.g. "qwen2.5:14b" on Ollama);
                for "transformers" a HuggingFace id (e.g.
                "Qwen/Qwen2.5-7B-Instruct").
            api_key (str, optional): Only needed if a local server enforces
                one; defaults to "not-needed".
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens to generate.
            backend (str, optional): "none" | "openai-compatible" |
                "transformers". Defaults to $LLM_BACKEND or "none".
            base_url (str, optional): OpenAI-compatible server URL. Defaults
                to $LLM_BASE_URL or the local Ollama endpoint.
        """
        self.backend = backend or os.getenv("LLM_BACKEND", "none")
        self.model_name = model_name or os.getenv("LLM_MODEL", "qwen2.5:14b")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL")
                         or DEFAULT_BASE_URLS["ollama"]).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "not-needed")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._hf_pipeline = None
        self._embedder = None

        if self.backend not in ("none", "openai-compatible", "transformers"):
            raise ValueError(f"Unknown LLM backend: {self.backend}")

    @property
    def available(self):
        """Whether a real LLM backend is configured."""
        return self.backend != "none"

    def generate(self, prompt, system_message=None, temperature=None):
        """
        Generate text using the configured open-source backend.

        Args:
            prompt (str): Input prompt for generation
            system_message (str, optional): System message for chat models
            temperature (float, optional): Override default temperature

        Returns:
            str: Generated text
        """
        temp = temperature if temperature is not None else self.temperature

        if self.backend == "none":
            raise RuntimeError(
                "LLM_BACKEND=none: no LLM configured. Either rely on the "
                "rule-based fallbacks (pass llm_client=None to components) "
                "or start a free local server, e.g.:\n"
                "  ollama pull qwen2.5:14b && ollama serve\n"
                "  export LLM_BACKEND=openai-compatible LLM_MODEL=qwen2.5:14b")

        if self.backend == "transformers":
            return self._generate_transformers(prompt, system_message, temp)
        return self._generate_openai_compatible(prompt, system_message, temp)

    def _generate_openai_compatible(self, prompt, system_message, temperature,
                                    retries=3):
        import requests

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        last_error = None
        for attempt in range(retries):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload,
                    timeout=300,
                )
                response.raise_for_status()
                text = response.json()["choices"][0]["message"]["content"]
                logger.debug(f"LLM request took {time.time() - start_time:.2f}s")
                return text.strip()
            except Exception as e:
                last_error = e
                logger.warning(f"LLM request attempt {attempt + 1} failed: {e}")
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(
            f"Could not reach open-source LLM server at {self.base_url} "
            f"after {retries} attempts: {last_error}")

    def _generate_transformers(self, prompt, system_message, temperature):
        if self._hf_pipeline is None:
            from transformers import pipeline
            logger.info(f"Loading local HuggingFace model {self.model_name}")
            self._hf_pipeline = pipeline(
                "text-generation", model=self.model_name, device_map="auto")

        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        outputs = self._hf_pipeline(
            messages,
            max_new_tokens=self.max_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
        )
        return outputs[0]["generated_text"][-1]["content"].strip()

    def generate_with_format(self, prompt, output_format, system_message=None,
                             temperature=None):
        """Generate text constrained to a described output format."""
        formatted_prompt = f"""{prompt}

Please format your response as follows:
{output_format}"""
        return self.generate(formatted_prompt, system_message, temperature)

    def generate_json(self, prompt, json_schema, system_message=None,
                      temperature=None):
        """
        Generate JSON output using the LLM.

        Returns:
            dict: Parsed JSON object ({} on parse failure).
        """
        formatted_prompt = f"""{prompt}

Please provide your response as a valid JSON object with the following schema:
```json
{json.dumps(json_schema, indent=2)}
```

Your response should only contain the JSON object, without any additional text."""

        response = self.generate(formatted_prompt, system_message, temperature)
        try:
            if "```json" in response:
                json_text = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_text = response.split("```")[1].strip()
            else:
                json_text = response.strip()
            return json.loads(json_text)
        except Exception as e:
            logger.error(f"Error parsing JSON from LLM response: {str(e)}")
            logger.error(f"Raw response: {response}")
            return {}

    def get_embedding(self, text, model=None):
        """
        Get an embedding vector using open-source models only.

        Uses sentence-transformers (all-MiniLM-L6-v2, local and free) when
        installed; otherwise a deterministic feature-hashing embedding so that
        memory retrieval remains functional with no dependencies.

        Args:
            text (str): Text to embed
            model (str, optional): sentence-transformers model name

        Returns:
            list: Embedding vector
        """
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer(
                    model or os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
            return self._embedder.encode(text).tolist()
        except ImportError:
            return self._hashing_embedding(text)
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            return self._hashing_embedding(text)

    @staticmethod
    def _hashing_embedding(text, dim=384):
        """Deterministic bag-of-words feature-hashing embedding (L2-normed)."""
        vector = [0.0] * dim
        for token in text.lower().split():
            digest = hashlib.md5(token.encode()).digest()
            index = int.from_bytes(digest[:4], "little") % dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]
