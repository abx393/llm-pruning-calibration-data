import argparse
import os 
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaConfig
from huggingface_hub import login
from importlib.metadata import version

from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_acc, eval_bleu, run_benchmarking, evaluate_task_result
import pdb
import os

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

with open("pat.txt", "r") as f:
    pat = f.read().strip()

login(token=pat)

def get_llm(args, model_name, cache_dir="llm_weights", mode='llama'):
    if args.calibration == 'svamp':
        if mode == "llama":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                low_cpu_mem_usage=True,
                device_map="auto",
                offload_folder="/content/offload",
                use_auth_token=True
            )
            model.seqlen = args.seqlen
        elif mode == "gpt2":
            model = AutoModelForCausalLM.from_pretrained('gpt2',
                                                         torch_dtype=torch.float16,
                                                         cache_dir=cache_dir,
                                                         low_cpu_mem_usage=True,
                                                         device_map="auto",
                                                         offload_folder="/content/offload")
            model.seqlen = 96
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
            offload_folder="offload",
            use_auth_token=True
        )
        model.seqlen = args.seqlen

    return model

def main():
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    os.environ['TORCH_USE_CUDA_DSA'] = '1'
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt", "none"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--append_to_file', type=str, default=None, help='File to append results to.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--calibration', type=str, nargs='+', default='c4',
            choices=['c4', 'oscar', 'redpajama', 'pile', 'gsm8k', 'svamp', 'mawps', 'anli_r1', 'anli_r2',
                'anli_r3', 'esnli', 'rte', 'boolq', 'commonsense_qa', 'race',
                'winogrande', 'wmt14', 'iwslt', 'ellipses', 'random'])
    parser.add_argument('--rationale', action='store_true')
    parser.add_argument('--eval_rationale', action='store_true')
    parser.add_argument('--eval', type=str, default='wikitext',
            choices=['wikitext', 'redpajama', 'oscar', 'gsm8k', 'svamp', 'mawps', 'anli_r1',
                'anli_r2', 'anli_r3', 'esnli', 'rte', 'boolq',
                'commonsense_qa', 'race', 'winogrande', 'wmt14', 'iwslt', 'all'])
    parser.add_argument('--skip_dense_eval', action='store_true')
    parser.add_argument('--input_format', type=str, default='concat', choices=['autoregressive', 'single', 'concat', 'zero'])
    parser.add_argument('--benchmark_task', type=str, default='boolq', choices=['boolq', 'rte', 'hellaswag', 'winogrande', 'openbookqa', 'arc_easy', 'arc_challenge'])
    parser.add_argument('--padding_side', type=str, default='left', choices=['left', 'right'])
    parser.add_argument('--shot', type=str, default='few', choices=['few', 'zero'])
    parser.add_argument('--attention_mask_type', type=str, default='configured', choices=['default', 'configured'])
    parser.add_argument('--seqlen', type=int, default=2048)
    parser.add_argument('--data_seqlen', type=int, default=None)
    parser.add_argument('--num_incontext', type=int)
    parser.add_argument('--num_cot_steps', type=int)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    print('args', args)

    #ppl_datasets = []
    ppl_datasets = ['wikitext', 'redpajama', 'oscar']
    acc_datasets = ['gsm8k', 'svamp', 'mawps', 'esnli', 'anli_r1', 'anli_r2', 'anli_r3', 'commonsense_qa', 'race', 'winogrande']
    bleu_datasets = ['wmt14', 'iwslt']

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    model_name = args.model.split("/")[-1]
    print(f"loading llm model {args.model}")
    model = get_llm(args, args.model, args.cache_dir)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, padding_side=args.padding_side, token=pat)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    #if args.calibration != "anli":
    #print('Running Benchmarking')
    #run_benchmarking(model, tokenizer, args.benchmark_task)
    #print('Evaluating on Benchmarks')
    #evaluate_task_result(args.benchmark_task)

    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)
    if not args.skip_dense_eval:
        print('Dense Model Scores:')
        if args.eval == 'all':
            for dataset in ppl_datasets:
                ppl = eval_ppl(model, tokenizer, device, verbose=args.verbose)
                print(f"Perplexity on {dataset}: {ppl}")
            for dataset in acc_datasets:
                acc = eval_acc(args, model, tokenizer, device, dataset=dataset, verbose=args.verbose)
                print(f"Acc. on {dataset}: {acc}")
            for dataset in bleu_datasets:
                bleu = eval_bleu(args, model, tokenizer, device, dataset=dataset, verbose=args.verbose)['bleu']
                print(f"Acc. on {dataset}: {acc}")
        elif args.eval in ppl_datasets:
            ppl = eval_ppl(model, tokenizer, device, verbose=args.verbose)
            print(f"Perplexity on {args.eval}: {ppl}")
        elif args.eval in acc_datasets:
            acc = eval_acc(args, model, tokenizer, device, dataset=args.eval, shot=args.shot, verbose=args.verbose)
            print(f"Acc. on {args.eval}: {acc}")
        elif args.eval in bleu_datasets:
            bleu = eval_bleu(args, model, tokenizer, device, dataset=args.eval, shot=args.shot, verbose=args.verbose)['bleu']
            print(f"bleu on {args.eval}: {bleu}")

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################

    print("*"*30)
    sparsity_ratio = check_sparsity(args, model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*"*30)
    ################################################################
    print('Sparse Model Scores:')
    if args.eval == 'all':
        for dataset in ppl_datasets:
            ppl = eval_ppl(model, tokenizer, device, verbose=args.verbose)
            print(f"Perplexity on {dataset}: {ppl}")
        for dataset in acc_datasets:
            acc = eval_acc(args, model, tokenizer, device, dataset=dataset, verbose=args.verbose)
            print(f"Acc. on {dataset}: {acc}")
        for dataset in bleu_datasets:
            bleu = eval_bleu(args, model, tokenizer, device, dataset=dataset, verbose=args.verbose)['bleu']
            print(f"Acc. on {dataset}: {acc}")
    elif args.eval in ppl_datasets:
        ppl = eval_ppl(model, tokenizer, device, verbose=args.verbose)
        print(f"Perplexity on {args.eval}: {ppl}")
    elif args.eval in acc_datasets:
        acc = eval_acc(args, model, tokenizer, device, dataset=args.eval, shot=args.shot, verbose=args.verbose)
        print(f"Acc. on {args.eval}: {acc}")
    elif args.eval in bleu_datasets:
        bleu = eval_bleu(args, model, tokenizer, device, dataset=args.eval, shot=args.shot, verbose=args.verbose)['bleu']
        print(f"bleu on {args.eval}: {bleu}")

    #if args.calibration != "anli":
    #print('Running Benchmarking')
    #run_benchmarking(model, tokenizer, args.benchmark_task)
    #print('Evaluating on Benchmarks')
    #evaluate_task_result(args.benchmark_task)
    #sys.exit(1)

    if args.save:
        args.save = os.path.join(args.save, model_name, args.prune_method)
        if not os.path.exists(args.save):
            os.makedirs(args.save)
        save_filepath = os.path.join(args.save, f'calibration_{args.calibration}_format_'
                                                f'{args.input_format}_eval_{args.eval}_{args.padding_side}_padding_'
                                                f'{args.shot}_shot_evaluation.txt')
        with open(save_filepath, "w") as f:
            if args.eval in ppl_datasets:
                print("actual_sparsity\tppl", file=f, flush=True)
                print(f"{sparsity_ratio:.4f}\t{ppl:.4f}", file=f, flush=True)
            elif args.eval in acc_datasets:
                print("actual_sparsity\tacc", file=f, flush=True)
                print(f"{sparsity_ratio:.4f}\t{acc:.4f}", file=f, flush=True)
            elif args.eval in bleu_datasets:
                print("actual_sparsity\tbleu", file=f, flush=True)
                print(f"{sparsity_ratio:.4f}\t{bleu:.4f}", file=f, flush=True)

    if args.append_to_file:
        with open(args.append_to_file, "a") as f:
            if args.eval in ppl_datasets:
                print(f"{ppl:.4f}", file=f, flush=True, end="")
            elif args.eval in acc_datasets:
                print(f"{acc:.4f}", file=f, flush=True, end="")
            elif args.eval in bleu_datasets:
                print(f"{bleu:.4f}", file=f, flush=True, end="")

    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        model.save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)

if __name__ == '__main__':
    main()
