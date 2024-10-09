#!/bin/bash
#SBATCH --job-name=sparsegpt-svamp
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1-12:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/svamp-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_svamp_sparsegpt.sh -m llama2-7b

