#!/bin/bash

while getopts ":m:s:" flag; do
	case $flag in
		m) # Handle -m flag
		if [ $OPTARG == llama ]; then
			model="huggyllama/llama-7b"
			model_short="llama"
		elif [ $OPTARG == llama2 ]; then
			model="meta-llama/Llama-2-7b-chat-hf"
			model_short="llama2"
		else
			echo "model should be either 'llama' or 'llama2'"
			exit -1
		fi
		;;
	esac
done


for seed in 0; do
	mkdir -p out/${model_short}/${seed}
	rm out/${model_short}/${seed}/nli_cot_matrix_esnli.csv
	touch out/${model_short}/${seed}/nli_cot_matrix_esnli.csv
	echo "eval task,dense model score,pruned c4,pruned esnli+cot,pruned esnli,pruned anli_r1+cot,pruned anli_r1" >> out/${model_short}/${seed}/nli_cot_matrix_esnli.csv

	# wanda, pruning matrix
	for eval_task in esnli ; do
		echo -n ${eval_task}+cot >> out/${model_short}/${seed}/nli_cot_matrix_esnli.csv
		echo -n , >> out/${model_short}/${seed}/nli_cot_matrix_esnli.csv
		python main.py --model ${model} --prune_method none --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${model_short}/${seed}/nli_cot_matrix_esnli.csv --calibration c4 --eval ${eval_task} --shot few --seed ${seed} --skip_dense_eval --eval_rationale

		echo -n "," >> out/${model_short}/${seed}/nli_cot_matrix_esnli.csv
		python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${model_short}/${seed}/nli_cot_matrix_esnli.csv --calibration c4 --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval --eval_rationale

		for calibration_task in esnli anli_r1; do
			echo -n "," >> out/${model_short}/${seed}/nli_cot_matrix_esnli.csv
			python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${model_short}/${seed}/nli_cot_matrix_esnli.csv --calibration ${calibration_task} --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval --rationale --eval_rationale

			echo -n "," >> out/${model_short}/${seed}/nli_cot_matrix_esnli.csv
			python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/unstructured/ --append_to_file out/${model_short}/${seed}/nli_cot_matrix_esnli.csv --calibration ${calibration_task} --eval ${eval_task} --shot few --input_format concat --padding_side left --seed ${seed} --skip_dense_eval --eval_rationale
		done
		echo "" >> out/${model_short}/${seed}/nli_cot_matrix_esnli.csv
	done
done
