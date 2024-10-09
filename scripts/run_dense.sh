#!/bin/bash
#SBATCH --job-name=wanda-dense
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/dense-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_dense.sh -m llama2-7b

