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

if [ ! -d llm_pruned_weights/${seed}/pruned_svamp ]; then
	echo "Directory doesn't exist"
	python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration svamp --eval svamp --input_format concat --seed ${seed} --skip_dense_eval --save_model llm_pruned_weights/${seed}/pruned_svamp
fi
if [ ! -d llm_pruned_weights/${seed}/pruned_gsm8k ]; then
	echo "Directory doesn't exist"
	python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration gsm8k --eval gsm8k --input_format concat --seed ${seed} --skip_dense_eval --save_model llm_pruned_weights/${seed}/pruned_gsm8k
fi
if [ ! -d llm_pruned_weights/${seed}/pruned_mawps ]; then
	echo "Directory doesn't exist"
	python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration mawps --eval mawps --input_format concat --seed ${seed} --skip_dense_eval --save_model llm_pruned_weights/${seed}/pruned_mawps
fi

if [ ! -d llm_pruned_weights/${seed}/pruned_commonsense_qa ]; then
	echo "Directory doesn't exist"
	python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration commonsense_qa --eval commonsense_qa --input_format concat --seed ${seed} --skip_dense_eval --save_model llm_pruned_weights/${seed}/pruned_commonsense_qa
fi
if [ ! -d llm_pruned_weights/${seed}/pruned_race ]; then
	echo "Directory doesn't exist"
	python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration race --eval race --input_format concat --seed ${seed} --skip_dense_eval --save_model llm_pruned_weights/${seed}/pruned_race
fi
if [ ! -d llm_pruned_weights/${seed}/pruned_winogrande ]; then
	echo "Directory doesn't exist"
	python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration winogrande --eval winogrande --input_format concat --seed ${seed} --skip_dense_eval --save_model llm_pruned_weights/${seed}/pruned_winogrande
fi


for eval_task in svamp gsm8k mawps anli_r1 anli_r2 anli_r3 commonsense_qa race winogrande; do
	python merge.py --model ${model} --input_models llm_pruned_weights/${seed}/pruned_commonsense_qa llm_pruned_weights/${seed}/pruned_race llm_pruned_weights/${seed}/pruned_winogrande --sparsity_ratio 0.5 --eval ${eval_task}
	#llm_pruned_weights/pruned_commonsense_qa llm_pruned_weights/pruned_race llm_pruned_weights/pruned_winogrande --sparsity_ratio 0.5 --eval ${eval_task}
done
