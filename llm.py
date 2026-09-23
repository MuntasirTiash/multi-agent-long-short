"""
llm.py — bridge to the zero-cost LLMClient in llm_utils.py.

Phase 2 gives the agents a real brain: a local, open-source Qwen2.5 model.
The LLMClient in llm_utils.py (shared with the sibling FAgent project) supports
a rule-based "none" backend, a local Ollama/vLLM server, or in-process
HuggingFace transformers — never a paid API.

We load that file directly by path (with importlib): the copy bundled in this
repo by default, or ../FAgent/utils/llm_utils.py when LLM_UTILS_PATH points
there. Loading by path rather than via sys.path means FAgent's own `config`
module can never shadow this package's `config`.

Backend is chosen by the LLM_BACKEND environment variable, e.g.:
    export LLM_BACKEND=transformers LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct
    # or, against a local server you started yourself:
    export LLM_BACKEND=openai-compatible LLM_MODEL=qwen2.5:7b
If LLM_BACKEND is unset or "none", make_llm_client() returns None and the
agents fall back to their Phase-0 rule-based logic (still zero cost).
"""

import importlib.util
import os

_LLM_UTILS = os.path.abspath(os.getenv(
    "LLM_UTILS_PATH", os.path.join(os.path.dirname(__file__), "llm_utils.py")))

# Default to the smallest instruct model so it runs even on a CPU login node;
# use 7B/14B on a GPU compute node for real experiments.
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def _load_llm_module():
    spec = importlib.util.spec_from_file_location("fagent_llm_utils", _LLM_UTILS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_llm_client(model_name: str = None):
    """
    Return an LLMClient, or None if no real backend is configured.

    None triggers the rule-based fallback inside StockAgent, so callers never
    have to branch on availability themselves.
    """
    if os.getenv("LLM_BACKEND", "none") == "none":
        return None
    module = _load_llm_module()
    client = module.LLMClient(
        model_name=model_name or os.getenv("LLM_MODEL", DEFAULT_MODEL),
        temperature=0.3,
        max_tokens=256,   # we only need a small JSON object back
    )
    return client if client.available else None
