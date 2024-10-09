import time 
import heapq 
import torch 
import torch.nn as nn 
from .sparsegpt import SparseGPT 
from .layerwrapper import WrappedGPT
from .data import get_loaders

from pdb import set_trace as st 

def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def check_sparsity(args, model):
    use_cache = model.config.use_cache 
    model.config.use_cache = False 

    layers = model.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        if args.verbose:
            print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(args, model, dataloader, device, pad_token, mode='llama', prepend_calibration=None):
    print('prepare_calibration_input')
    use_cache = model.config.use_cache
    model.config.use_cache = False

    if mode == 'gpt2':
        layers = model.transformer.h
    elif mode == 'llama':
        layers = model.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    print("prepare_calibration_input model.seqlen", model.seqlen)
    inps = torch.zeros((args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    print("prepare_calibration_input inps.size()", inps.size())
    attention_masks = torch.zeros((args.nsamples, 1, 1, model.seqlen, model.seqlen), device=device)
    attention_masks.requires_grad = False
    cache = {"i": 0, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            if prepend_calibration is not None:
                #print("forward inp.size()", inp.size())
                #print("forward prepend_calibration.size()", prepend_calibration.size())
                inp[:,:prepend_calibration.size()[1],:] = prepend_calibration

            inps[cache['i']] = inp
            if 'attention_mask' in kwargs:
                attention_masks[cache['i']] = kwargs['attention_mask']

            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
                #print("position_ids", kwargs["position_ids"])
                #print("position_ids.size()", kwargs["position_ids"].size())

            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])

    for i, batch in enumerate(dataloader):
        inp = batch[0].to(device)
        if prepend_calibration is not None:
            inp = torch.cat((torch.ones(1, prepend_calibration.size()[1]).to(device), inp), dim=1).type(torch.LongTensor)

        #print("inp.size()", inp.size())
        try:
            #print('pad_token', pad_token)
            attention_mask = torch.tensor([[0 if inp[0][j] == pad_token else 1 for j in range(inp.size()[1])]]).to(device).type(torch.LongTensor)
            #print('attention_mask', attention_mask)
            #print('inp', inp)
            model(inp.to(device), attention_mask=attention_mask.to(device))
            #model(inp.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    #print("model.seqlen", model.seqlen)
    #print("default_attention_mask.size()", default_attention_mask.size())
    return inps, outs, attention_masks, position_ids

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_random(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_mask = torch.randint(2, W.size()) == 1
            W[W_mask] = 0

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)
                print("magnitude W_mask", W_mask)

            W[W_mask] = 0

def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prepend_calibration=None, prune_n=0, prune_m=0, mode='llama'):
    use_cache = model.config.use_cache
    model.config.use_cache = False 

    print("loading calibration data")
    pad_token = None
    dataloader = []
    print("args.calibration", args.calibration)
    for calibration_data in args.calibration:
        print("prune_wanda calibration_data", calibration_data)
        print("model.seqlen", model.seqlen)
        seqlen = model.seqlen
        if prepend_calibration is not None:
            seqlen -= prepend_calibration.size()[1]
        curr_dataloader, _, pad_token = get_loaders(calibration_data, nsamples=args.nsamples // len(args.calibration), seed=args.seed,
                                                seqlen=seqlen, data_seqlen=args.data_seqlen, tokenizer=tokenizer, 
                                                rationale=args.rationale, input_format=args.input_format,
                                                padding_side=args.padding_side, verbose=args.verbose, num_incontext=args.num_incontext, num_cot_steps=args.num_cot_steps)
        print("get_loaders done")
        dataloader.extend(curr_dataloader)

    print("dataset loading complete")
    with torch.no_grad():
        inps, outs, configured_attention_masks, position_ids = prepare_calibration_input(args, model, dataloader, device, pad_token, prepend_calibration=prepend_calibration)

    if mode == 'gpt2':
        layers = model.transformer.h
    elif mode == 'llama':
        layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        if f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, configured_attention_masks, position_ids = inps.to(dev), outs.to(dev), configured_attention_masks.to(dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=configured_attention_masks[j], position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            if args.verbose:
                print(f"pruning layer {i} name {name}")
            if mode == 'gpt2':
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((-1, 1)))
            elif mode == 'llama':
                W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    if args.verbose:
                        print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            subset[name].weight.data[W_mask] = 0  ## set weights to zero 

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=configured_attention_masks[j], position_ids=position_ids)[0]
        inps, outs = outs, inps

    model.config.use_cache = use_cache 
    torch.cuda.empty_cache()

@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0, mode='llama', prepend_calibration=None):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    pad_token = None
    dataloader = []
    print("args.calibration", args.calibration)
    for calibration_data in args.calibration:
        print("prune_sparsegpt calibration_data", calibration_data)
        print("model.seqlen", model.seqlen)
        seqlen = args.data_seqlen if args.data_seqlen is not None else model.seqlen
        curr_dataloader, _, pad_token = get_loaders(calibration_data, nsamples=args.nsamples // len(args.calibration), seed=args.seed,
                                                seqlen=seqlen, tokenizer=tokenizer, 
                                                rationale=args.rationale, input_format=args.input_format,
                                                padding_side=args.padding_side, verbose=args.verbose, num_incontext=args.num_incontext, num_cot_steps=args.num_cot_steps)
        print("get_loaders done")
        dataloader.extend(curr_dataloader)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    if mode == 'gpt2':
        layers = model.transformer.h
    elif mode == 'llama':
        layers = model.model.layers

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=device
    )
    attention_masks = torch.zeros((args.nsamples, 1, 1, model.seqlen, model.seqlen), device=device)
    attention_masks.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            if 'attention_mask' in kwargs:
                attention_masks[cache['i']] = kwargs['attention_mask']
            if 'position_ids' in kwargs:
                cache['position_ids'] = kwargs['position_ids']
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        inp, _ = batch
        try:
            #model(inp.to(device))
            attention_mask = torch.tensor([[0 if inp[0][j] == pad_token else 1 for j in range(inp.size()[1])]])
            #print('attention_mask', attention_mask)
            #print('inp', inp)
            model(inp.to(device), attention_mask=attention_mask.to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    #attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            if args.verbose:
                print(f"layer {i} device {dev}")
            inps, outs, attention_masks, position_ids = inps.to(dev), outs.to(dev), attention_masks.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j], position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            if args.verbose:
                print(i, name)
                print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j], position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
