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

for seed in 1; do
	mkdir out/${seed}
	rm out/${seed}/wmt14.csv
	touch out/${seed}/wmt14.csv
	echo "eval task,dense model score,pruned iwslt" >> out/${seed}/iwslt.csv

	# wanda, pruning matrix
	for eval_task in iwslt; do
		echo -n ${eval_task} >> out/${seed}/iwslt.csv
		echo -n , >> out/${seed}/iwslt.csv
		python main.py --model ${model} --prune_method none --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/iwslt.csv --calibration c4 --eval ${eval_task} --shot few --seed ${seed} --skip_dense_eval
		for calibration_task in iwslt; do
			echo -n "," >> out/${seed}/iwslt.csv
			python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/iwslt.csv --calibration ${calibration_task} --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval
		done
		echo "" >> out/${seed}/iwslt.csv
	done
done
