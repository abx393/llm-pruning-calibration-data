#!/bin/bash
#SBATCH --job-name=soft-prompt
#SBATCH --partition=gpu-a40
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=3-00:00:00
#SBATCH --mail-user=abandari@uw.edu
#SBATCH --output=hyak_job_outputs/softprompt-%j.out

source ~/.bashrc
conda activate prune_llm
wandb login

model="huggyllama/llama-7b"
seed=0
python main.py --model ${model} --nsamples 1 --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration c4 --seqlen 256 --seed ${seed} --eval wikitext
python soft_prompt_prune.py --calibration c4

