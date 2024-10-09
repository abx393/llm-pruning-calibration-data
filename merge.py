import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch

from lib.prune import check_sparsity, find_layers, prune_magnitude
from lib.eval import eval_ppl, eval_acc

with open("pat.txt", "r") as f:
    pat = f.read().strip()

login(token=pat)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='Path to model')
    parser.add_argument('--input_models', nargs='+', type=str, help='File path to weights of models')
    parser.add_argument('--weights', nargs='+', type=float, help='Weight of each input model')
    parser.add_argument('--tune_dataset1', type=str, help='Performance on which dataset to choose the hyperparameters based on')
    parser.add_argument('--tune_dataset2', type=str, help='Performance on which dataset to choose the hyperparameters based on')
    parser.add_argument('--rationale', action='store_true')
    parser.add_argument('--eval_rationale', action='store_true')
    parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Target sparsity ratio of pruned model')
    parser.add_argument('--eval', type=str, default='svamp', choices=['wikitext', 'gsm8k', 'svamp', 'mawps', 'anli_r1', 'anli_r2', 'anli_r3', 'esnli', 'rte', 'boolq', 'commonsense_qa', 'race', 'winogrande', 'all'])
    parser.add_argument('--padding_side', type=str, default='left', choices=['left', 'right'], help='Padding side')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--save', type=str, default=None, help='Path to save model accuracy metric')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    print('args', args)
    model_paths = args.input_models
    print('model_paths', model_paths)
    w = args.weights[0]
    print('weights', w)

    ppl_datasets = []
    #ppl_datasets = ['wikitext']
    acc_datasets = ['gsm8k', 'svamp', 'mawps', 'anli_r1', 'anli_r2', 'anli_r3', 'commonsense_qa', 'race', 'winogrande']

    device = torch.device("cuda:0")

    max_acc = 0
    best_w = 0
    print('w', w)
    models = []
    for model_path in model_paths:
        print('model_path', model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map={"model": 0, "lm_head": 0},
            offload_folder="offload",
            use_auth_token=True
        )
        model.seqlen = 2048
        print('model.seqlen', model.seqlen)
        models.append(model)
    print('len(models)', len(models))

    layers_all = []
    for model in models:
        layers = model.model.layers
        layers_all.append(layers)

    for i in range(len(layers_all[0])):
        subsets = []
        for j in range(len(layers_all)):
            layer = layers_all[j][i]
            subset = find_layers(layer)
            subsets.append(subset)

        for name in subsets[0]:
            #print('type(subsets[0][name].weight)', type(subsets[0][name].weight))
            W_sum = w * subsets[0][name].weight.data
            #print('type(W_sum)', type(W_sum))
            for k in range(1, len(layers_all)):
                W_k = subsets[k][name].weight.data
                W_sum += (1-w) * W_k


    sparsity_ratio = check_sparsity(args, models[0])
    print(f"sparsity after merge {sparsity_ratio:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, padding_side=args.padding_side, token=pat)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    models[0].resize_token_embeddings(len(tokenizer))

    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = models[0].hf_device_map["lm_head"]

    prune_magnitude(args, models[0], tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0)
    sparsity_ratio = check_sparsity(args, models[0])
    print(f"sparsity after merge and magnitude pruning {sparsity_ratio:.4f}")
    if args.eval == 'all':
        for dataset in ppl_datasets:
            ppl = eval_ppl(models[0], tokenizer, device, verbose=args.verbose)
            print(f"Perplexity on {dataset}: {ppl}")

        acc1 = 0
        acc2 = 0
        for dataset in acc_datasets:
            acc = eval_acc(args, models[0], tokenizer, device, dataset=dataset, verbose=args.verbose)
            print(f"Acc. on {dataset}: {acc}")
            if dataset == args.tune_dataset1:
                acc1 = acc
            if dataset == args.tune_dataset2:
                acc2 = acc
        acc_avg = (acc1 + acc2) / 2
        with open(args.save, 'w') as f:
            f.write(str(acc_avg))

    elif args.eval in ppl_datasets:
        ppl = eval_ppl(models[0], tokenizer, device, verbose=args.verbose)
        print(f"Perplexity on {dataset}: {ppl}")
    elif args.eval in acc_datasets:
        acc = eval_acc(args, models[0], tokenizer, device, dataset=args.eval, shot=args.shot, verbose=args.verbose)
        print(f"Acc. on {args.eval}: {acc}")

    with torch.no_grad():
        torch.cuda.empty_cache()


    """
    print('best_w', best_w)
    models = []
    for model_path in model_paths:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
            offload_folder="offload",
            use_auth_token=True
        )
        model.seqlen = 2048
        models.append(model)

    layers_all = []
    for model in models:
        layers = model.model.layers
        layers_all.append(layers)

    for i in range(len(layers_all[0])):
        subsets = []
        for j in range(len(layers_all)):
            layer = layers_all[j][i]
            subset = find_layers(layer)
            subsets.append(subset)

        for name in subsets[0]:
            #print('type(subsets[0][name].weight)', type(subsets[0][name].weight))
            W_sum = best_w * subsets[0][name].weight.data.to(device)
            #print('type(W_sum)', type(W_sum))
            for k in range(1, len(layers_all)):
                W_k = subsets[k][name].weight.data.to(device)
                W_sum += (1-best_w) * W_k


    sparsity_ratio = check_sparsity(args, models[0])
    print(f"sparsity after merge {sparsity_ratio:.4f}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False, padding_side=args.padding_side, token=pat)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    models[0].resize_token_embeddings(len(tokenizer))

    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = models[0].hf_device_map["lm_head"]

    prune_magnitude(args, models[0], tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0)
    sparsity_ratio = check_sparsity(args, models[0])
    print(f"sparsity after merge and magnitude pruning {sparsity_ratio:.4f}")
    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        models[0].save_pretrained(args.save_model)
        tokenizer.save_pretrained(args.save_model)
    """
if __name__ == "__main__":
    main()
