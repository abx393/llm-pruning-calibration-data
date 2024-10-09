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


for seed in 2; do
	mkdir out/${seed}
	rm out/${seed}/cot_matrix.csv
	touch out/${seed}/cot_matrix.csv
	echo "eval task,dense model score,pruned c4,pruned gsm8k+cot,pruned gsm8k,pruned svamp+cot,pruned svamp" >> out/${seed}/cot_matrix.csv

	# wanda, pruning matrix
	for eval_task in gsm8k svamp mawps; do
		echo -n ${eval_task} >> out/${seed}/cot_matrix.csv
		echo -n , >> out/${seed}/cot_matrix.csv
		python main.py --model ${model} --prune_method none --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/cot_matrix.csv --calibration c4 --eval ${eval_task} --shot few --seed ${seed} --skip_dense_eval

		echo -n "," >> out/${seed}/cot_matrix.csv
		python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/cot_matrix.csv --calibration c4 --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval

		for calibration_task in gsm8k svamp; do
			echo -n "," >> out/${seed}/cot_matrix.csv
			python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/cot_matrix.csv --calibration ${calibration_task} --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval --rationale

			echo -n "," >> out/${seed}/cot_matrix.csv
			python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/cot_matrix.csv --calibration ${calibration_task} --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval
		done
		echo "" >> out/${seed}/cot_matrix.csv

		echo -n ${eval_task}+cot >> out/${seed}/cot_matrix.csv
		echo -n , >> out/${seed}/cot_matrix.csv
		python main.py --model ${model} --prune_method none --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/cot_matrix.csv --calibration c4 --eval ${eval_task} --shot few --seed ${seed} --skip_dense_eval --eval_rationale

		echo -n "," >> out/${seed}/cot_matrix.csv
		python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/cot_matrix.csv --calibration c4 --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval --eval_rationale

		for calibration_task in gsm8k svamp; do
			echo -n "," >> out/${seed}/cot_matrix.csv
			python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/cot_matrix.csv --calibration ${calibration_task} --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval --rationale --eval_rationale

			echo -n "," >> out/${seed}/cot_matrix.csv
			python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${seed}/cot_matrix.csv --calibration ${calibration_task} --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval --eval_rationale
		done
		echo "" >> out/${seed}/cot_matrix.csv
	done
done
