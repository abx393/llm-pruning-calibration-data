#!/bin/bash
#SBATCH --job-name=wanda-matrix
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/matrix-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_matrix.sh -m llama2-7b

