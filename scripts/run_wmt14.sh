#!/bin/bash
#SBATCH --job-name=wmt
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/wmt14-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_wmt14.sh -m llama2

