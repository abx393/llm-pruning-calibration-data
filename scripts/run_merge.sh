#!/bin/bash
#SBATCH --job-name=merge2
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/merge-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_merge.sh -m llama

