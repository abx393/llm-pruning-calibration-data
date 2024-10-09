# wanda, calibrate on Pile, evaluate on wikitext, few-shot learning during evaluation, left-padding (baseline)
for i in 64 128 256 512 1024; do
	python main.py --nsamples $i --model meta-llama/Llama-2-7b-chat-hf --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration pile --eval wikitext --shot few --input_format concat --padding_side left
done
