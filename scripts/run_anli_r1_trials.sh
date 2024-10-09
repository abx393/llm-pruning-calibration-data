#!/bin/bash
#SBATCH --job-name=wanda-anlir1
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-12:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/anlir1-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_anli_r1_trials.sh -m llama2-7b

