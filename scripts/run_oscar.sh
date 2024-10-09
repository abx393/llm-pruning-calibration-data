#!/bin/bash
#SBATCH --job-name=wanda-osc
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-gpu=5
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/oscar-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_oscar.sh -m llama2-7b

