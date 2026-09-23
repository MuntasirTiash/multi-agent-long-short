#!/bin/bash
#SBATCH --job-name=orch_weekly
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=dtyu            # PI group (verified: sacctmgr assoc user=$USER)
#SBATCH --gres=gpu:a100:1         # A100 explicitly: qos=standard blocks L40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00         # qos=standard MaxWall is exactly 3 days
#SBATCH --output=results/weekly_vllm_%j.log

# ===========================================================================
# WEEKLY walk-forward topology comparison, 498 firms, Qwen2.5-7B via vLLM.
#   run_llm_analysis.py -> no-comm baseline vs every topology, with rank-IC,
#   transaction costs and a Monte-Carlo null
#
# Weekly rebalancing over the same window:
#   74 dates x (498 round-0 + 3 topologies x 2 rounds x 498) ~= 258k LLM calls
# Round 0 is computed ONCE per date and replayed for every topology, so the
# no-comm baseline and the treatments see identical analyst opinions.
#
# NOTE this script grades EQUAL-WEIGHT top-N/bottom-N books and never touches
# ManagerAgent, so manager settings cannot move these numbers (see README, "Reading results honestly").
#
# Submit from the agent_orchestration directory:  sbatch weekly_vllm_3day.sh
# Scale at submit time, e.g.:
#   sbatch --export=ALL,MAX_DATES=20 weekly_vllm_3day.sh
# ===========================================================================

set -uo pipefail

echo "==== [1/5] Environment ===================================================="
# No `module purge`: on this cluster it makes Miniforge3 fail to load
# (seen in vllm_smoke and in job 1116161) and it strips CUDA.
module load Miniforge3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agents

# vLLM is pre-installed to a dedicated prefix so the job never pip-installs on
# the clock, and never silently continues without a server (which would grade a
# rule-based run and label it LLM — the silent-fallback failure mode).
export PYTHONPATH=/project/dtyu/ms3235/vllm_pkgs:${PYTHONPATH:-}
# REQUIRED: the agents env's optree extension needs GLIBCXX_3.4.31 but system
# /lib64/libstdc++.so.6 stops at 3.4.29, so `import vllm` dies with an
# ImportError that looks like a vLLM bug. The conda env ships GLIBCXX_3.4.34.
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}
export HF_HOME=/project/dtyu/ms3235/hf_cache          # 7B already cached (15G)
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export TOKENIZERS_PARALLELISM=false

MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}
PORT=${VLLM_PORT:-$((8000 + RANDOM % 1000))}          # avoid clashing with the
                                                      # sibling weekly job
# `full` is excluded on purpose: at 498 firms it puts 497 peer opinions in one
# prompt (~3.2k tokens) and would dominate the token budget. sparse is 10% of
# the cross-section (50 peers), sector is 20-78, corr_topk is 10.
export TOPOLOGIES=${TOPOLOGIES:-sparse,sector,corr_topk}
mkdir -p results
echo "python -> $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import vllm; print('vllm', vllm.__version__)" || {
    echo "FATAL: vLLM not importable — install it first:"
    echo "  pip install --target=/project/dtyu/ms3235/vllm_pkgs vllm"; exit 1; }

echo "==== [2/5] Starting vLLM for $MODEL on port $PORT ========================"
# 8192 context: the sector topology's revision prompts reach ~2k tokens and the
# manager's shortlist prompt ~2.6k, so the 4096 default would truncate.
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --no-enable-log-requests \
    > results/weekly_vllm_server_${SLURM_JOB_ID}.log 2>&1 &
VLLM_PID=$!
cleanup() { [ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT
echo "vLLM pid=$VLLM_PID (server log: results/weekly_vllm_server_${SLURM_JOB_ID}.log)"

echo "==== [3/5] Waiting for the server (up to 15 min for model load) =========="
UP=0
for i in $(seq 1 180); do
    if curl -sf "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
        UP=1; echo "vLLM is up after ~$((i*5))s."; break
    fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then
        echo "FATAL: vLLM died during startup. Tail of its log:"
        tail -40 results/weekly_vllm_server_${SLURM_JOB_ID}.log; exit 1
    fi
    sleep 5
done
# Abort rather than run: a rule-based run mislabelled as LLM is worse than no run.
[ "$UP" = "1" ] || { echo "FATAL: server never answered; aborting."; exit 1; }

echo "==== [4/5] Weekly walk-forward (topologies=$TOPOLOGIES) =================="
export LLM_BACKEND=openai-compatible
export LLM_MODEL="$MODEL"
export LLM_BASE_URL="http://localhost:${PORT}/v1"
python run_llm_analysis.py
STATUS=$?

echo "==== [5/5] Done (exit=$STATUS); stopping vLLM ============================="
cleanup
echo "Outputs:"
ls -lt results/llm_analysis_*.json results/transcript_*.jsonl 2>/dev/null | head -4
exit $STATUS
