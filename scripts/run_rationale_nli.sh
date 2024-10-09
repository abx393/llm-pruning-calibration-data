#!/bin/bash
#SBATCH --job-name=wanda-nlicot
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=8:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/nlicotmatrix-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_rationale_nli.sh -m llama2

