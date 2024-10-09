model_short="llama2"

# wanda, calibrate on gsm8k, evaluate on gsm8k, few-shot learning during evaluation, left-padding (baseline)
for seed in 0; do
	mkdir -p out/${model_short}/${seed}
	rm out/${model_short}/${seed}/in_context.csv
	echo "model,accuracy" >> out/${model_short}/${seed}/in_context.csv
	echo -n "dense_model," >> out/${model_short}/${seed}/in_context.csv
	python main.py --seed ${seed} --model meta-llama/Llama-2-7b-chat-hf --prune_method none --sparsity_ratio 0.5 --sparsity_type unstructured --calibration gsm8k --eval gsm8k --shot few --input_format concat --padding_side left --append_to_file out/${model_short}/${seed}/in_context.csv
	echo "" >> out/${model_short}/${seed}/in_context.csv
	for i in 5 10 15 20 25; do
		echo -n $i examples in pruning input, >> out/${model_short}/${seed}/in_context.csv
		python main.py --seed ${seed} --model meta-llama/Llama-2-7b-chat-hf --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --calibration gsm8k --eval gsm8k --shot few --input_format concat --padding_side left --num_incontext $i --skip_dense_eval --append_to_file out/${model_short}/${seed}/in_context.csv
		echo "" >> out/${model_short}/${seed}/in_context.csv
	done
done
