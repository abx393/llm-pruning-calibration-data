#### generate benchmark datasets
#shots=0
#for task in boolq rte hellaswag winogrande openbookqa arc_easy arc_challenge; do
#python -u lib/generate_task_data.py --output-file ${task}-${shots}.jsonl --task-name ${task} --num-fewshot ${shots}
#done

# wanda, calibrate on boolq, evaluate on boolq, few-shot learning during evaluation, left-padding
python main.py --model huggyllama/llama-7b --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval svamp --shot few --input_format concat --padding_side left
python main.py --model huggyllama/llama-7b --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval svamp --shot few --input_format concat --padding_side left --rationale
python main.py --model huggyllama/llama-7b --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval svamp --shot few --input_format single --padding_side left
python main.py --model huggyllama/llama-7b --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval svamp --shot few --input_format single --padding_side left --rationale

# wanda, calibrate on C4, evaluate on boolq, few-shot learning during evaluation, left-padding (baseline)
python main.py --model huggyllama/llama-7b --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration c4 --eval boolq --shot few --input_format single --padding_side left

# wanda, calibrate on gsm8k, evaluate on boolq, few-shot learning during evaluation, left-padding
python main.py --model huggyllama/llama-7b --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration gsm8k --eval boolq --shot few --input_format concat --padding_side left
python main.py --model huggyllama/llama-7b --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration gsm8k --eval boolq --shot few --input_format concat --padding_side left --rationale
python main.py --model huggyllama/llama-7b --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration gsm8k --eval boolq --shot few --input_format single --padding_side left
python main.py --model huggyllama/llama-7b --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration gsm8k --eval boolq --shot few --input_format single --padding_side left --rationale

# magnitude, few-shot learning during evaluation (baseline)
python main.py --model huggyllama/llama-7b --prune_method magnitude --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --eval boolq --shot few

