# sparsegpt, calibrate on svamp, evaluate on svamp, few-shot learning during evaluation, left-padding
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval svamp --shot few --input_format concat --padding_side left
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval svamp --shot few --input_format concat --padding_side left --rationale
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval svamp --shot few --input_format single --padding_side left
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval svamp --shot few --input_format single --padding_side left --rationale
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration svamp --eval svamp --shot few --input_format autoregressive --padding_side left

# sparsegpt, calibrate on C4, evaluate on svamp, few-shot learning during evaluation, left-padding (baseline)
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration c4 --eval svamp --shot few --input_format single --padding_side left

# magnitude, few-shot learning during evaluation (baseline)
python main.py --model huggyllama/llama-7b --prune_method magnitude --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --eval svamp --shot few

# sparsegpt, calibrate on gsm8k, evaluate on svamp, few-shot learning during evaluation, left-padding
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration gsm8k --eval svamp --shot few --input_format concat --padding_side left
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration gsm8k --eval svamp --shot few --input_format concat --padding_side left --rationale
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration gsm8k --eval svamp --shot few --input_format single --padding_side left
python main.py --model huggyllama/llama-7b --prune_method sparsegpt --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --calibration gsm8k --eval svamp --shot few --input_format single --padding_side left --rationale

# magnitude, few-shot learning during evaluation (baseline)
python main.py --model huggyllama/llama-7b --prune_method magnitude --sparsity_ratio 0.5 --sparsity_type unstructured --save out/llama_7b/unstructured/ --eval svamp --shot few

