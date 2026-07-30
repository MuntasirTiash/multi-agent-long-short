#!/bin/bash
#SBATCH --job-name=daily_mgr
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=dtyu            # PI group (verified via sacctmgr)
#SBATCH --gres=gpu:a100:1         # A100 explicitly: qos=standard blocks L40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00           # standard qos allows up to 3 days
#SBATCH --output=results/daily_mgr_%j.log

# ===========================================================================
# DAILY manager-agent backtest with FULL observability, on a GPU using the
# IN-PROCESS transformers backend (Qwen2.5-7B). No vLLM, no server, no internet
# needed on the compute node — the model is already in $HF_HOME.
#
# One StockAgent per Dow-30 stock forms an independent view (round 0), revises
# it over full/sparse/sector topologies, and a ManagerAgent turns the 30 final
# opinions into a dollar-neutral long/short book graded on next-day returns.
# Each run writes a self-contained results/daily_<stamp>_.../ directory:
#   run.log (full narrative: rankings + next-day returns + per-round memory +
#   EVERY fallback with its raw model output + the manager's decision),
#   metrics.json, metrics.csv, daily_returns.csv, fallbacks.csv, transcript.jsonl.
#
# Submit from the agent_orchestration directory:
#     sbatch daily_backtest_gpu.sh
# Scale at submit time, e.g.:
#     sbatch --export=ALL,MAX_DATES=20 daily_backtest_gpu.sh
#     sbatch --export=ALL,TOPOLOGIES=sparse,MAX_DATES=40 daily_backtest_gpu.sh
#     sbatch --export=ALL,LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct daily_backtest_gpu.sh
#
# In-process transformers is SEQUENTIAL (no batching), so daily x 3 topologies
# is heavy — MAX_DATES is capped low by default; raise it once you've eyeballed
# a small run. The fallback count in the summary makes a silent rule-based run
# impossible to miss.
# ===========================================================================

set -uo pipefail

echo "==== [1/3] Environment ===================================================="
# Miniforge3 BASE python: torch 2.6.0+cu124 (sees the A100) plus transformers and
# requests. Do NOT `module purge` (strips CUDA) and do NOT `conda activate agents`
# (that env has CPU-only torch and would silently run the 7B on CPU).
module load Miniforge3
echo "python -> $(which python)"
python -c "import sys, torch; print('exe:', sys.executable, '| cuda:', torch.cuda.is_available())"
# Abort early with a clear message if the GPU isn't visible (don't burn hours on CPU).
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "FATAL: torch.cuda.is_available() is False — GPU not visible; aborting."; exit 1; }

export HF_HOME=/project/dtyu/ms3235/hf_cache          # 0.5B & 7B cached here
export PYTHONUNBUFFERED=1                              # stream logs live
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export LLM_BACKEND=transformers
export LLM_MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}
export TOPOLOGIES=${TOPOLOGIES:-full,sparse,sector}
export MAX_DATES=${MAX_DATES:-8}                       # bound so it finishes; raise later
export DEMO_N=${DEMO_N:-30}
mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "==== [2/3] Daily manager backtest (model=$LLM_MODEL, dates=$MAX_DATES, topo=$TOPOLOGIES)"
python run_daily_backtest.py

echo "==== [3/3] Done =========================================================="
echo "SLURM log: results/daily_mgr_${SLURM_JOB_ID}.log"
echo "Per-run outputs are under the RESULTS DIR printed above (results/daily_<stamp>_...)."
ls -dt results/daily_2* 2>/dev/null | head -3
