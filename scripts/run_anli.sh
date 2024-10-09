#!/bin/bash
#SBATCH --job-name=wanda-svamp
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/anli-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_anli.sh

