# wanda, calibrate on Pile, evaluate on mawps
for i in 0.1 0.3 0.5 0.7 0.9; do
	python main.py --nsamples 128 --model meta-llama/Llama-2-7b-chat-hf --prune_method wanda --sparsity_ratio $i --sparsity_type unstructured --calibration pile --eval mawps --shot few --input_format concat --padding_side left
done
