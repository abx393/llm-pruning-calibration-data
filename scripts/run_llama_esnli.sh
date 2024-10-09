#!/bin/bash
#SBATCH --job-name=eval-esnli
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/esnli-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_llama_esnli_remaining.sh -m llama7b

