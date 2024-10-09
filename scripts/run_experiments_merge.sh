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

best_w=0
max_acc=0
for w in 0.2 0.4 0.6 0.8; do
	python merge.py --model ${model} --input_models llm_pruned_weights/${seed}/pruned_svamp llm_pruned_weights/${seed}/pruned_gsm8k --weights $w --sparsity_ratio 0.5 --eval all --save_model llm_pruned_weights/${seed}/merged_svamp_gsm8k_$w --save out/merged_svamp_gsm8k_$w.txt --tune_dataset1 svamp --tune_dataset2 gsm8k
	acc=$(cat out/merged_svamp_gsm8k_$w.txt)
	if [ ${acc}>${max_acc} ]; then
		max_acc=${acc}
		best_w=$w
	fi
done
echo max_acc ${max_acc}
echo best_w ${best_w}

best_w_triple=0
max_acc_triple=0
for w in 0.2 0.4 0.6 0.8; do
	python merge.py --model ${model} --input_models llm_pruned_weights/${seed}/merged_svamp_gsm8k_${best_w} llm_pruned_weights/${seed}/pruned_mawps --weights $w --sparsity_ratio 0.5 --eval all --save_model llm_pruned_weights/${seed}/merged_svamp_gsm8k_mawps_$w --save out/merged_svamp_gsm8k_mawps_$w --tune_dataset1 gsm8k --tune_dataset2 mawps
	acc=$(cat out/merged_svamp_gsm8k_mawps_$w)
	if [ ${acc}>${max_acc_triple} ]; then
		max_acc_triple=${acc}
		best_w_triple=$w
	fi
done
echo max_acc_triple ${max_acc_triple}
echo best_w_triple ${best_w_triple}

#python merge.py --model ${model} --input_models llm_pruned_weights/${seed}/pruned_commonsense_qa llm_pruned_weights/${seed}/pruned_race --tune_dataset1 commonsense_qa --tune_dataset2 race --sparsity_ratio 0.5 --eval all --save_model llm_pruned_weights/${seed}/merged_commonsense_qa_race
#python merge.py --model ${model} --input_models llm_pruned_weights/${seed}/merged_commonsense_qa_race llm_pruned_weights/${seed}/pruned_winogrande --tune_dataset1 commonsense_qa --tune_dataset2 winogrande --sparsity_ratio 0.5 --eval all --save_model llm_pruned_weights/${seed}/merged_commonsense_qa_race_winogrande
