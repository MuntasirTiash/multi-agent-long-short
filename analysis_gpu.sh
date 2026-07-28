#!/bin/bash
#SBATCH --job-name=orch_llm
#SBATCH --partition=gpu
#SBATCH --qos=standard
#SBATCH --account=dtyu            # PI group (verified via sacctmgr)
#SBATCH --gres=gpu:a100:1         # A100 explicitly: qos=standard blocks L40
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00           # standard qos allows up to 3 days
#SBATCH --output=results/analysis_gpu_%j.log

# ===========================================================================
# Phase-2 batched walk-forward on a GPU, using the IN-PROCESS transformers
# backend (Qwen2.5-7B). No vLLM, no server, no internet needed on the compute
# node — the model is already in $HF_HOME. Compares no-communication vs ALL
# topologies (full/sparse/sector) and grades each with the full harness
# (rank-IC + transaction costs + Monte-Carlo null significance).
#
# Submit from the agent_orchestration directory:
#     sbatch analysis_gpu.sh
# Scale it at submit time, e.g.:
#     sbatch --export=ALL,MAX_DATES=74 analysis_gpu.sh          # full history
#     sbatch --export=ALL,LLM_MODEL=Qwen/Qwen2.5-0.5B-Instruct analysis_gpu.sh
# ===========================================================================

set -uo pipefail

echo "==== [1/3] Environment ===================================================="
# Use the Miniforge3 BASE python: it has torch 2.6.0+cu124 (sees the A100) plus
# transformers, pandas and requests. Do NOT `conda activate agents` here — that
# env has a CPU-only torch (torch.version.cuda is None) and would silently run
# the 7B model on CPU. (Verified on an A100 node, job 1116182.)
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
export MAX_DATES=${MAX_DATES:-12}                      # bound so it finishes; 74 = full
mkdir -p results
nvidia-smi --query-gpu=name,memory.total --format=csv || true

echo "==== [2/3] Batched walk-forward (model=$LLM_MODEL, dates=$MAX_DATES, topo=$TOPOLOGIES)"
python run_llm_analysis.py

echo "==== [3/3] Done =========================================================="
echo "Outputs:"
ls -lt results/ | head -8
