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


for seed in 0 10 100; do
	mkdir out/${seed}
	rm out/${seed}/anli_matrix.csv
	touch out/${seed}/anli_matrix.csv
	echo "eval task,dense model score,pruned c4,pruned gsm8k,pruned svamp,pruned mawps,pruned anli_r1, pruned anli_r2, pruned anli_r3,pruned commensense_qa,pruned race,pruned winogrande" >> out/${seed}/anli_matrix.csv

	# wanda, pruning matrix
	for eval_task in anli_r1 anli_r2 anli_r3; do
		echo -n ${eval_task} >> out/${seed}/anli_matrix.csv
		echo -n , >> out/${seed}/anli_matrix.csv
		python main.py --model ${model} --prune_method none --sparsity_ratio 0.5 --sparsity_type unstructured n--save out/unstructured/ --append_to_file out/${seed}/anli_matrix.csv --calibration c4 --eval ${eval_task} --shot few --seed ${seed} --skip_dense_eval
		for calibration_task in c4 gsm8k svamp mawps anli_r1 anli_r2 anli_r3 commonsense_qa race winogrande; do
			echo -n "," >> out/${seed}/anli_matrix.csv
			python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/anli_matrix.csv --calibration ${calibration_task} --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval
		done
		echo "" >> out/${seed}/anli_matrix.csv
	done

done
