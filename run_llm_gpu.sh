#!/bin/bash
#SBATCH --job-name=orch_llm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=dtyu           # PI group account (adjust if sacctmgr differs)
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=orch_llm_%j.log

# Phase-2 LLM demo with the 7B model on a GPU node — one agent per Dow-30 stock,
# reasoning with local open-source Qwen2.5 (zero API cost).
# Submit from the agent_orchestration directory:  sbatch run_llm_gpu.sh

module load Miniforge3
# Batch shells are non-interactive: initialise conda's hook before activating.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate agents

export HF_HOME=/project/dtyu/ms3235/hf_cache          # 0.5B & 7B already cached
export LLM_BACKEND=transformers
export LLM_MODEL=${LLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}

# Full 30-stock universe (unset DEMO_N). transformers picks up the GPU via
# device_map="auto" inside FAgent's LLMClient.
python run_llm_demo.py
