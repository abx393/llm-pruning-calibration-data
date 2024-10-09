for i in 7 8 9 10 11; do
	python main.py --nsamples $((128 / (2**($i-7)))) --model meta-llama/Llama-2-7b-chat-hf --prune_method wanda --sparsity_ratio 0.5 --sparsity_type unstructured --calibration c4 --eval wikitext --shot few --input_format concat --padding_side left --data_seqlen $((2**$i))
done

