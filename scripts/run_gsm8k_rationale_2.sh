#!/bin/bash
#SBATCH --job-name=wanda-gsm8k-rationale
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=10
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/gsm8k-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_gsm8k_rationale_2.sh -m llama2-7b

