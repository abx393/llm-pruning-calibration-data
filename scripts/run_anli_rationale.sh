#!/bin/bash
#SBATCH --job-name=anli-rationale
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/anlirationale-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_anli_rationale.sh -m llama2
