# wanda, calibrate on C4, evaluate on svamp, few-shot learning during evaluation, left-padding (baseline)
for i in 128 256 512 1024 2048; do
	python main.py --nsamples 128 --model meta-llama/Llama-2-7b-chat-hf --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration c4 --eval wikitext --shot few --input_format concat --padding_side left --data_seqlen $i
done
