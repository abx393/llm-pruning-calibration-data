#!/bin/bash
#SBATCH --job-name=wanda-gsm8k
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/gsm8k-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_llama_gsm8k.sh -m llama7b

