#!/bin/bash
#SBATCH --job-name=d7b_daily
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=dtyu            # PI group (verified via sacctmgr)
#SBATCH --gres=gpu:a100:1         # A100 explicitly: qos=standard blocks L40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00         # qos=standard MaxWall is exactly 3 days
#SBATCH --output=results/daily_7b_%j.log

# ===========================================================================
# DAILY manager-agent backtest — 498 firms, Qwen2.5-7B, in-process transformers.
#
# Uses the ONLY GPU path proven to work in this repo: the Miniforge BASE python
# (torch 2.6.0+cu124) with LLM_BACKEND=transformers, which loads the 7B via
# FAgent's LLMClient -> HF pipeline(device_map="auto"). Jobs 1116184/1116195 ran
# this way and answered 30/30 by LLM with zero fallbacks. NOT vLLM: that path is
# still being set up, and the one previous vLLM job (1115934) silently ran 100%
# rule-based because `pip install vllm || true` swallowed the failure.
#
# WHY MAX_DATES=32 and not the full 373:
#   measured throughput on that 7B transformers run = 0.57 LLM calls/s
#   (1,656 calls in 2,883 s, sequential — SequentialBatchClient does not batch)
#   cost per rebalance date at 498 firms, 3 topologies, 2 rounds:
#     498 round-0 + 3 x 2 x 498 revisions + 3 manager = 3,489 calls
#   3 days of wall clock therefore buys ~35 dates; 32 leaves ~17 h of headroom
#   for model load, price/feature setup and slower-than-measured generation.
#   The full 373-date run needs ~26 days at this rate — it needs vLLM's
#   continuous batching, not a longer wall clock.
#
# A timeout would lose metrics.json (it is written after the loop), so the date
# count is deliberately conservative. run.log and transcript.jsonl are written
# incrementally, so even a killed job leaves the per-agent record behind.
#
# Submit from the agent_orchestration directory:  sbatch daily_7b_3day.sh
#   sbatch --export=ALL,MAX_DATES=10 daily_7b_3day.sh        # shorter probe
#   sbatch --export=ALL,TOPOLOGIES=sparse daily_7b_3day.sh   # 1 topology, 3x dates
# ===========================================================================

set -uo pipefail

echo "==== [1/3] Environment ===================================================="
# Miniforge3 BASE python: torch 2.6.0+cu124 (sees the A100) plus transformers and
# requests. Do NOT `module purge` (strips CUDA) and do NOT `conda activate agents`
# (that env has CPU-only torch 2.12.1 and would silently run the 7B on CPU).
module load Miniforge3
echo "python -> $(which python)"
python -c "import sys, torch; print('exe:', sys.executable, '| cuda:', torch.cuda.is_available())"
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || {
    echo "FATAL: torch.cuda.is_available() is False — GPU not visible; aborting."; exit 1; }

export HF_HOME=/project/dtyu/ms3235/hf_cache          # 7B cached here (15G)
export PYTHONUNBUFFERED=1                              # stream logs live
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export TOKENIZERS_PARALLELISM=false
export LLM_BACKEND=transformers
export LLM_MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}
# `full` is left out: at 498 firms it puts 497 peer opinions in one prompt, which
# multiplies prefill cost for no extra dates. corr_topk exercises the new
# correlation topology; sparse is 10% of the cross-section (50 peers).
export TOPOLOGIES=${TOPOLOGIES:-sparse,sector,corr_topk}
export MAX_DATES=${MAX_DATES:-32}
# DEMO_N deliberately UNSET so the full 498-firm universe is used (the older
# daily_backtest_gpu.sh pinned DEMO_N=30 from the Dow-30 era).
mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "==== [2/3] Daily backtest (model=$LLM_MODEL, dates=$MAX_DATES, topo=$TOPOLOGIES)"
python run_daily_backtest.py
STATUS=$?

echo "==== [3/3] Done (exit=$STATUS) ==========================================="
echo "Newest results directory:"
ls -td results/daily_2* 2>/dev/null | head -1
exit $STATUS
