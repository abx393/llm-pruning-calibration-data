
#model="huggyllama/llama-7b"
model="meta-llama/Llama-2-7b-chat-hf"
# wanda, calibrate on anli, evaluate on anli, few-shot learning during evaluation, left-padding
python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration anli --eval anli --shot few --input_format concat --padding_side left
python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration anli --eval anli --shot few --input_format single --padding_side left

# wanda, calibrate on C4, evaluate on anli, few-shot learning during evaluation, left-padding (baseline)
python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration c4 --eval anli --shot few --input_format single --padding_side left

# wanda, calibrate on svamp, evaluate on anli, few-shot learning during evaluation, left-padding
python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval anli --shot few --input_format concat --padding_side left
python main.py --model ${model} --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval anli --shot few --input_format single --padding_side left

# magnitude, few-shot learning during evaluation (baseline)
python main.py --model ${model} --prune_method magnitude --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --eval anli --shot few
