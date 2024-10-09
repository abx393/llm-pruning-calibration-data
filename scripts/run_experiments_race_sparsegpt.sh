#!/bin/bash

while getopts ":m:s:" flag; do
	case $flag in
		m) # Handle -m flag
		if [ $OPTARG == llama7b ]; then
			model_short="llama7b"
			model="huggyllama/llama-7b"
		elif [ $OPTARG == llama65b ]; then
			model_short="llama65b"
			model="huggyllama/llama-65b"
		elif [ $OPTARG == llama2-7b ]; then
			model_short="llama2"
			model="meta-llama/Llama-2-7b-chat-hf"
		elif [ $OPTARG == llama2-70b ]; then
			model_short="llama2-70b"
			model="meta-llama/Llama-2-70b-chat-hf"
		else
			echo "model should be 'llama7b', 'llama65b', 'llama2-7b', 'llama2-70b'"
			exit -1
		fi
		;;
	esac
done


for seed in 0; do
	mkdir -p out/${model_short}/${seed}
	rm out/${model_short}/${seed}/race_sparsegpt.csv
	touch out/${model_short}/${seed}/race_sparsegpt.csv
	echo "eval task,pruned race" >> out/${model_short}/${seed}/race_sparsegpt.csv

	# sparsegpt, pruning race
	for eval_task in gsm8k svamp mawps esnli anli_r1 anli_r2 anli_r3 commonsense_qa race winogrande wmt14; do
		echo -n ${eval_task} >> out/${model_short}/${seed}/race_sparsegpt.csv
		for calibration_task in race; do
			echo -n "," >> out/${model_short}/${seed}/race_sparsegpt.csv
			python main.py --model ${model} --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${model_short}/${seed}/race_sparsegpt.csv --calibration ${calibration_task} --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval
		done
		echo "" >> out/${model_short}/${seed}/race_sparsegpt.csv
	done
done
