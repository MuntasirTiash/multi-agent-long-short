#!/bin/bash
#SBATCH --job-name=vllm_smoke
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=dtyu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=results/vllm_smoke_%j.log

# ===========================================================================
# Prove the vLLM path end to end BEFORE any long job depends on it.
#
# This exists because job 1115934 looked successful and produced
# llm_analysis_*.json, but its log says "vLLM not importable after install;
# analysis will fall back to rule" and every round read "0/30 answered by LLM" —
# a 100% rule-based run wearing an LLM filename. So the bar here is not "the
# server started", it is "a real StockAgent prompt came back as parseable JSON
# through the same HttpBatchClient the analysis uses".
#
# Checks, in order, each fatal so the failure is unambiguous:
#   1. which torch actually gets imported (the agents env has a CPU-only
#      torch 2.12.1; vllm_pkgs ships its own CUDA build, and PYTHONPATH must win)
#   2. torch.cuda.is_available()
#   3. import vllm
#   4. server answers /v1/models
#   5. HttpBatchClient returns usable JSON for 8 real agent prompts
#   6. measured throughput, to size the real jobs against the 0.57 calls/s the
#      sequential transformers path gives
#
# Submit:  sbatch vllm_smoke.sh
# ===========================================================================

set -uo pipefail

echo "==== [1/6] Environment ===================================================="
module purge 2>/dev/null || true
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agents
export PYTHONPATH=/project/dtyu/ms3235/vllm_pkgs:${PYTHONPATH:-}
# The agents env's compiled extensions (optree) need GLIBCXX_3.4.31, but the
# system /lib64/libstdc++.so.6 only provides up to 3.4.29 — without this,
# `import vllm` dies with an ImportError that looks like a vLLM problem but is
# really a libstdc++ problem. The conda env ships up to GLIBCXX_3.4.34.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/project/dtyu/ms3235/hf_cache
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
mkdir -p results
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true

python - <<'PY' || { echo "FATAL: torch/vllm import checks failed."; exit 1; }
import sys
print("python:", sys.executable)
import torch
print("torch:", torch.__version__, "| cuda build:", torch.version.cuda,
      "| available:", torch.cuda.is_available())
print("torch loaded from:", torch.__file__)
if torch.version.cuda is None:
    # PYTHONPATH did not win over the agents env's CPU-only torch.
    sys.exit("FATAL: imported a CPU-only torch — vLLM cannot use the GPU")
if not torch.cuda.is_available():
    sys.exit("FATAL: torch cannot see the GPU")
import vllm
print("vllm:", vllm.__version__, "from", vllm.__file__)
PY

MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}
PORT=$((8000 + RANDOM % 1000))

echo "==== [2/6] Starting vLLM ($MODEL, port $PORT) ============================="
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "$PORT" --max-model-len 8192 \
    --gpu-memory-utilization 0.90 --no-enable-log-requests \
    > results/vllm_smoke_server_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
cleanup() { [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "==== [3/6] Waiting for /v1/models ========================================"
UP=0
for i in $(seq 1 180); do
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1 && { UP=1; \
        echo "up after ~$((i*5))s"; break; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo "server died; tail of its log:"; \
        tail -40 results/vllm_smoke_server_${SLURM_JOB_ID}.log; exit 1; }
    sleep 5
done
[ "$UP" = "1" ] || { echo "FATAL: server never answered"; \
    tail -40 results/vllm_smoke_server_${SLURM_JOB_ID}.log; exit 1; }

echo "==== [4/6] Real agent prompts through HttpBatchClient ===================="
export LLM_BACKEND=openai-compatible LLM_MODEL="$MODEL"
export LLM_BASE_URL="http://localhost:${PORT}/v1"
python - <<'PY'
import os, sys, time
import config, data_loader as dl
from batch_llm import HttpBatchClient
from features import PriceHistory, compute_extended_features
from stock_agent import StockAgent

tickers = list(config.UNIVERSE)[:8]
ohlcv = dl.load_ohlcv(tickers, "2024-10-01", "2026-07-01")
feats = compute_extended_features(ohlcv, "2025-07-22", config.UNIVERSE,
                                  history=PriceHistory(ohlcv, config.UNIVERSE))
agents = {t: StockAgent(t, config.UNIVERSE[t]) for t in tickers}
prompts = [agents[t].initial_prompt(feats[t]) for t in tickers]

client = HttpBatchClient(os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"],
                         StockAgent.SYSTEM, StockAgent.SCHEMA)
t0 = time.time()
datas = client.generate_json_batch(prompts)
dt = time.time() - t0
pt, ct, lat, exact = client.pop_usage()

usable = sum(1 for d in datas if isinstance(d, dict) and "score" in d)
print(f"\n{usable}/{len(prompts)} answered by LLM, "
      f"{len(prompts)-usable} would fall back to rule")
print(f"batch of {len(prompts)} in {dt:.1f}s = {len(prompts)/dt:.2f} calls/s")
print(f"tokens: {pt} prompt + {ct} completion  (exact usage from server: {exact})")
for t, d in list(zip(tickers, datas))[:3]:
    print(f"  {t}: {d}")
if usable == 0:
    sys.exit("FATAL: every call fell back — this is the 1115934 failure mode")
PY
STATUS=$?

echo "==== [5/6] Throughput at 32-way concurrency =============================="
[ "$STATUS" = "0" ] && python - <<'PY'
import os, time
import config, data_loader as dl
from batch_llm import HttpBatchClient
from features import PriceHistory, compute_extended_features
from stock_agent import StockAgent

n = 96                        # enough to see continuous batching work
tickers = (list(config.UNIVERSE) * 2)[:n]
uni = list(config.UNIVERSE)
ohlcv = dl.load_ohlcv(uni, "2024-10-01", "2026-07-01")
feats = compute_extended_features(ohlcv, "2025-07-22", config.UNIVERSE,
                                  history=PriceHistory(ohlcv, config.UNIVERSE))
prompts = [StockAgent(t, config.UNIVERSE[t]).initial_prompt(feats[t])
           for t in tickers]
client = HttpBatchClient(os.environ["LLM_BASE_URL"], os.environ["LLM_MODEL"],
                         StockAgent.SYSTEM, StockAgent.SCHEMA, max_workers=32)
t0 = time.time()
datas = client.generate_json_batch(prompts)
dt = time.time() - t0
rate = n / dt
usable = sum(1 for d in datas if isinstance(d, dict) and "score" in d)
print(f"{n} prompts in {dt:.1f}s = {rate:.1f} calls/s ({usable}/{n} usable)")
per_date = 498 + 3 * 2 * 498 + 3
for label, dates in (("weekly 74", 74), ("daily 373", 373)):
    h = per_date * dates / rate / 3600
    print(f"  {label:<12} {per_date*dates:>10,} calls -> {h:6.1f} h "
          f"({h/24:.1f} days)  vs 0.57 calls/s sequential: "
          f"{per_date*dates/0.57/3600/24:.1f} days")
print(f"  dates that fit in 3 days: {int(3*24*3600*0.85*rate/per_date)}")
PY

echo "==== [6/6] Done (exit=$STATUS) ==========================================="
cleanup
exit $STATUS
