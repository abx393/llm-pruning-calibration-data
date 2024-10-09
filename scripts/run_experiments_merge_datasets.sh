#!/bin/bash
while getopts ":m:s:" flag; do
	case $flag in
		m) # Handle -m flag
		if [ $OPTARG == llama ]; then
			model="huggyllama/llama-7b"
		elif [ $OPTARG == llama2 ]; then
			model="meta-llama/Llama-2-7b-chat-hf"
		else
			echo "model should be either 'llama' or 'llama2'"
			exit -1
		fi
		;;
	esac
done

seed=1
python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration svamp gsm8k mawps --eval all --input_format concat --seed ${seed} --save_model llm_pruned_weights/${seed}/pruned_arithmetic
python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration anli_r1 anli_r2 anli_r3 --eval all --input_format concat --seed ${seed} --save_model llm_pruned_weights/${seed}/pruned_nli
python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration commonsense_qa race winogrande --eval all --input_format concat --seed ${seed} --save_model llm_pruned_weights/${seed}/pruned_commonsense
