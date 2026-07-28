#!/bin/bash
#SBATCH --job-name=orch_vllm
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=dtyu            # PI group (verified: sacctmgr assoc user=$USER)
#SBATCH --gres=gpu:a100:1         # A100 explicitly: qos=standard blocks L40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00           # standard qos allows up to 3 days
#SBATCH --output=results/vllm_analysis_%j.log

# ===========================================================================
# Phase-2 batched walk-forward: one agent per Dow-30 stock reasoning with a
# local, open-source Qwen2.5-7B served by vLLM (zero API cost). Compares
# no-communication vs ALL communication topologies (full/sparse/sector) and
# grades each with the full harness (rank-IC + transaction costs + Monte-Carlo
# null significance).
#
# Submit from the agent_orchestration directory:
#     sbatch serve_vllm_gpu.sh
#
# Stages:
#   1. Environment + (one-time) vLLM install
#   2. Start the vLLM OpenAI-compatible server on the GPU
#   3. Wait until it answers
#   4. Run the batched analysis against it (verbose per-step logging)
#   5. Stop the server (also on any early exit, via trap)
# If the server never comes up, the analysis degrades gracefully: every LLM
# call falls back to the rule, so the job still finishes with a (clearly
# logged) rule-based table instead of hanging.
# ===========================================================================

set -uo pipefail

echo "==== [1/5] Environment ===================================================="
module purge 2>/dev/null || true                      # clear any inherited modules
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agents
echo "python -> $(which python)"                       # must be the conda 'agents' env
python -c "import sys; print('executable:', sys.executable)"

export HF_HOME=/project/dtyu/ms3235/hf_cache          # 0.5B & 7B already cached
export PYTHONUNBUFFERED=1                              # stream logs live
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}
PORT=${VLLM_PORT:-8000}
export TOPOLOGIES=${TOPOLOGIES:-full,sparse,sector}
mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "==== [1/5] Ensuring vLLM is installed ====================================="
python -c "import vllm" 2>/dev/null || pip install --quiet vllm || true

VLLM_PID=""
cleanup() { [ -n "$VLLM_PID" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

echo "==== [2/5] Starting vLLM server for $MODEL on port $PORT ================="
if python -c "import vllm" 2>/dev/null; then
    python -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --max-model-len 4096 \
        --gpu-memory-utilization 0.90 > results/vllm_server_${SLURM_JOB_ID}.log 2>&1 &
    VLLM_PID=$!
    echo "vLLM server pid=$VLLM_PID (log: results/vllm_server_${SLURM_JOB_ID}.log)"
else
    echo "WARNING: vLLM not importable after install; analysis will fall back to rule."
fi

echo "==== [3/5] Waiting for the server (up to ~8 min for model load) =========="
for i in $(seq 1 96); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "vLLM is up after ~$((i*5))s."
        break
    fi
    sleep 5
done

echo "==== [4/5] Running batched analysis (topologies=$TOPOLOGIES) ============="
export LLM_BACKEND=openai-compatible
export LLM_MODEL="$MODEL"
export LLM_BASE_URL="http://localhost:${PORT}/v1"
python run_llm_analysis.py

echo "==== [5/5] Done; stopping vLLM ==========================================="
cleanup
echo "Outputs: results/vllm_analysis_${SLURM_JOB_ID}.log and results/llm_analysis_*.json"
ls -lt results/ | head -8
