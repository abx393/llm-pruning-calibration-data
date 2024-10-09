#!/bin/bash
#SBATCH --job-name=prune-esnli
#SBATCH --partition=gpu-a100
#SBATCH --account=cse
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=84G
#SBATCH --time=1-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/esnli-%j.out

source ~/.bashrc
conda activate prune_llm
bash run_experiments_llama_esnli_prune.sh -m llama7b

