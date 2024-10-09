import argparse
import gc
import copy
from transformers import AutoTokenizer, LlamaTokenizer, AutoModelForCausalLM, LlamaForCausalLM, get_linear_schedule_with_warmup, Trainer, TrainingArguments
from huggingface_hub import login
from peft import PromptEmbedding, PromptTuningConfig, get_peft_model
from lib.data import get_loaders #, get_svamp_individual
from lib.prune import prune_wanda, prune_random, prune_magnitude, check_sparsity
from lib.eval import eval_ppl
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import os
os.environ["WANDB_PROJECT"] = "llama_prompt_tuning"

with open("pat.txt", "r") as f:
    pat = f.read().strip()

login(token=pat)
with torch.no_grad():
    torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument('--nsamples', type=int, default=1, help='Number of calibration samples.')
parser.add_argument('--num_incontext', type=int, default=None)
parser.add_argument('--calibration', type=str, nargs='+', default='c4',
        choices=['c4', 'gsm8k', 'svamp', 'mawps', 'anli_r1', 'anli_r2',
            'anli_r3', 'esnli', 'rte', 'boolq', 'commonsense_qa', 'race',
            'winogrande', 'wmt14'])
parser.add_argument('--sparsity_ratio', type=float, default=0.5, help='Sparsity level')
parser.add_argument("--sparsity_type", type=str, default="unstructured", choices=["unstructured", "4:8", "2:4"])
parser.add_argument('--input_format', type=str, default='concat', choices=['autoregressive', 'single', 'concat'])
parser.add_argument('--padding_side', type=str, default='left', choices=['left', 'right'])
parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
parser.add_argument('--rationale', action='store_true')
parser.add_argument('--use_variant', action="store_true", help="whether to use the wanda variant described in the appendix")
parser.add_argument('--verbose', action='store_true')
args = parser.parse_args()
#torch.random.manual_seed(args.seed)

device = torch.device("cuda:0")

model_name="huggyllama/llama-7b"
cache_dir = "llm_weights"
dense_llama_model = LlamaForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    cache_dir=cache_dir,
    low_cpu_mem_usage=True,
    device_map="auto",
    offload_folder="/content/offload",
    use_auth_token=True
)
dense_llama_model.seqlen = 256
tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left', token=pat)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
dense_llama_model.resize_token_embeddings(len(tokenizer))

#train_dataloader = DataLoader(get_svamp_individual(tokenizer, split='train'), batch_size=64, shuffle=True)
#eval_dataloader = DataLoader(get_svamp_individual(tokenizer, split='test'), batch_size=64, shuffle=True)

init_text = "Please carefully examine the weight matrix within the model, as it may contain errors. It is crucial to verify its accuracy and make any necessary adjustments to ensure optimal performance."
config = PromptTuningConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=len(tokenizer(init_text)["input_ids"]),
        #token_dim=768,
        #num_transformer_submodules=1,
        #num_attention_heads=32,
        #num_layers=32,
        prompt_tuning_init="TEXT",
        prompt_tuning_init_text=init_text,
        tokenizer_name_or_path=model_name
)

model = get_peft_model(dense_llama_model, config)
model.print_trainable_parameters()

lr = 3e-2
num_epochs = 3
batch_size = 1

version_commit_sha = "607bd4c8450a42878aa9ddc051a65a055450ef87"
traindata = load_dataset('allenai/c4', revision=version_commit_sha, data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', verification_mode='no_checks').select(range(300))
#traindata.set_format('torch')
valdata = load_dataset('allenai/c4', revision=version_commit_sha, data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', verification_mode='no_checks').select(range(300))
#valdata.set_format('torch')

def prepare_input(example):
    tokenized_example = tokenizer(example['text'], truncation=True, padding='max_length', max_length=model.seqlen, return_tensors='pt')
    tokenized_example['input_ids'] = tokenized_example['input_ids']
    tokenized_example['labels'] = tokenized_example['input_ids'].clone()
    tokenized_example['attention_mask'] = tokenized_example['attention_mask']
    #print('tokenized_example size', tokenized_example['input_ids'].size())
    return tokenized_example

train_dataset = traindata.map(prepare_input)
train_dataset = train_dataset.remove_columns(traindata.column_names)
val_dataset = valdata.map(prepare_input)
val_dataset = val_dataset.remove_columns(valdata.column_names)
print('soft prompt tokens before ', model.get_prompt(1))
print('soft prompt tokens before size ', model.get_prompt(1).size())

train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * num_epochs)
)




####
prune_n, prune_m = 0, 0
base_model = copy.deepcopy(dense_llama_model)
prune_wanda(args, base_model, tokenizer, device=device, prepend_calibration=model.get_prompt(1), prune_n=prune_n, prune_m=prune_m)

ppl = eval_ppl(base_model, tokenizer, device, verbose=args.verbose)
print(f"Perplexity before learning softpromp: {ppl}")

####

print('starting training...')
for epoch in range(num_epochs):
    print('starting epoch ', epoch)
    model.train()
    total_loss = 0
    print('len(train_dataloader)', len(train_dataloader))
    for step, batch in enumerate(tqdm(train_dataloader)):
        """
        base_model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
            offload_folder="/content/offload",
            use_auth_token=True
        )
        base_model.seqlen = 256
        base_model.resize_token_embeddings(len(tokenizer))
        """
        
        base_model = copy.deepcopy(dense_llama_model)

        prune_wanda(args, base_model, tokenizer, device=device, prepend_calibration=model.get_prompt(1), prune_n=prune_n, prune_m=prune_m)
        model.base_model = base_model

        for k in batch:
            #print("k", k)
            batch[k] = torch.Tensor(batch[k]).type(torch.LongTensor)
            #print("batch[k]", batch[k])
            #print("type(batch[k])", type(batch[k]))
            #print("batch[k].size()", batch[k].size())
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        if step < len(train_dataloader) - 1:
            model.base_model = None
            del base_model
            gc.collect()
            torch.cuda.empty_cache()

    model.eval()
    val_loss = 0
    for step, batch in enumerate(tqdm(val_dataloader)):
        for k in batch:
            #print("k", k)
            batch[k] = torch.Tensor(batch[k]).type(torch.LongTensor)
            #print("batch[k]", batch[k])
            #print("type(batch[k])", type(batch[k]))
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        loss = outputs.loss
        val_loss += loss.detach().float()

    val_epoch_loss = val_loss / len(val_dataloader)
    val_ppl = torch.exp(val_epoch_loss)
    train_epoch_loss = total_loss / len(train_dataloader)
    train_ppl = torch.exp(train_epoch_loss)
    print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {val_ppl=} {val_epoch_loss=}")

    """
    trainer = Trainer(model, args=TrainingArguments("out/",
        num_train_epochs=num_epochs, per_device_train_batch_size=batch_size, report_to="wandb", run_name="llama_prompt_tuning", remove_unused_columns=False, logging_steps=1), train_dataset=train_dataset, eval_dataset=val_dataset, tokenizer=tokenizer)
    trainer.train()
    """

    print(f"soft prompt tokens after {epoch=} {model.get_prompt(1)}")
    print(f"soft prompt tokens after size {model.get_prompt(1).size()}")
    dense_model = copy.deepcopy(dense_llama_model)
    """
    dense_model = LlamaForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        device_map="auto",
        offload_folder="/content/offload",
        use_auth_token=True
    )
    dense_model.seqlen = 256
    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False, padding_side='left', token=pat)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dense_model.resize_token_embeddings(len(tokenizer))
    """

    ppl = eval_ppl(dense_model, tokenizer, device, verbose=args.verbose)
    print(f"{epoch=} Perplexity before pruning: {ppl}")

    prune_n, prune_m = 0, 0
    prune_wanda(args, dense_model, tokenizer, device=device, prepend_calibration=model.get_prompt(1), prune_n=prune_n, prune_m=prune_m)

    ppl = eval_ppl(dense_model, tokenizer, device, verbose=args.verbose)
    print(f"{epoch=} Perplexity after pruning: {ppl}")
    del dense_model
    gc.collect()
    torch.cuda.empty_cache()

