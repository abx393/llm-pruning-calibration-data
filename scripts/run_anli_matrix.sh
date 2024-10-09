#!/bin/bash
#SBATCH --job-name=anli-matrix
#SBATCH --partition=gpu-a40
#SBATCH --nodes=2
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=8-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/anlimatrix-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_anli_matrix.sh -m llama2
