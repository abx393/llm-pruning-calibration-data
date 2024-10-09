#!/bin/bash
#SBATCH --job-name=wanda-pdwanda
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/pdwanda-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_pd_wanda_0_7.sh -m llama2-7b

