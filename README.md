# [EMNLP 2024] Is C4 Dataset Optimal for Pruning? An Investigation of Calibration Data for LLM Pruning
## Abstract
Network pruning has emerged as a potential solution to make LLMs cheaper to
deploy. However, existing LLM pruning approaches universally rely on the C4
dataset as the calibration data for calculating pruning scores, leaving its
optimality unexplored. In this study, we evaluate the choice of calibration
data on LLM pruning, across a wide range of datasets that are most commonly
used in LLM training and evaluation, including four pertaining datasets as well
as three categories of downstream tasks encompassing nine datasets. Each
downstream dataset is prompted with In-Context Learning (ICL) and
Chain-of-Thought (CoT), respectively. Besides the already intriguing
observation that the choice of calibration data significantly impacts the
performance of pruned LLMs, our results also uncover several subtle and often
unexpected findings, summarized as follows: (1) C4 is not the optimal choice
for LLM pruning, even among commonly used pre-training datasets; (2) arithmetic
datasets—when used as calibration data—performs on par or even better than
pre-training datasets; (3) pruning with downstream datasets does not
necessarily help the corresponding downstream task, compared to pre-training
data; (4) ICL is widely beneficial to all data categories, whereas CoT is only
useful on certain tasks. Our findings shed light on the importance of carefully
selecting calibration data for LLM pruning and pave the way for more efficient
deployment of these powerful models in real-world applications. We release our
code at: https://github.com/abx393/llm-pruning-calibration-data.

## Setup
Installation instructions can be found in [INSTALL.md](INSTALL.md).

Please generate a HuggingFace [user access token](https://huggingface.co/docs/hub/security-tokens)
and create a file `pat.txt` in the top-level directory of this repository and write the access token
in this file.

## Usage
We provide a quick overview of the arguments:  
- `--model`: The identifier for the LLaMA model on the Hugging Face model hub.
- `--cache_dir`: Directory for loading or storing LLM weights. The default is `llm_weights`.
- `--prune_method`: We have implemented three pruning methods, namely [`magnitude`, `wanda`, `sparsegpt`].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--use_variant`: Whether to use the Wanda variant, default is `False`. 
- `--save`: Specifies the directory where the result will be stored
- `--calibration`: The dataset to calibrate on
- `--rationale`: If this flag is included, calibrate on input + rationale
- `--eval`: The dataset to evaluate on
- `--input_format`: The style of calibration data
- `--verbose`: If this flag is included, print intermediate results to stdout
- `--seed`: type=int, default=0, help='Seed for sampling the calibration data.'
- `--nsamples`: default=128, help='Number of calibration samples.'
- `--prune_method`: type=str, choices=["magnitude", "wanda", "sparsegpt", "none"]
- `--cache_dir`: default="llm_weights", type=str
- `--save`: type=str, default=None, help='Path to save results.'
- `--append_to_file`: type=str, default=None, help='File to append results to.'
- `--save_model`: type=str, default=None, help='Path to save the pruned model.'
- `--calibration`: type=str, nargs='+', default='c4',
            choices=['c4', 'oscar', 'redpajama', 'pile', 'gsm8k', 'svamp', 'mawps', 'anli_r1', 'anli_r2',
                'anli_r3', 'esnli', 'rte', 'boolq', 'commonsense_qa', 'race',
                'winogrande', 'wmt14', 'iwslt', 'ellipses', 'random'])
- `--rationale`: If flag is included, include CoT rationale in answer portion
  of Q-A pairs in calibration samples.
- `--eval_rationale`: If flag is included, at evaluation time, include CoT
  rationale in in-context examples in prompt.
- `--eval`: type=str, default='wikitext',
            choices=['wikitext', 'redpajama', 'oscar', 'gsm8k', 'svamp', 'mawps', 'anli_r1',
                'anli_r2', 'anli_r3', 'esnli', 'rte', 'boolq',
                'commonsense_qa', 'race', 'winogrande', 'wmt14', 'iwslt', 'all']
- `--skip_dense_eval`: If flag is included, skip evaluation of dense model
  (before pruning).
- `--input_format`: type=str, default='concat', choices=['single', 'concat', 'zero'])
- `--seqlen`: Length of context window in tokens. Default is 2048.
- `--data_seqlen`: Number of meaningful tokens in each calibration sample, the
  remaining portion of context window is filled with padding tokens.
- `--num_incontext`: Number of in-context Q-A pairs in each calibration sample.
- `--num_cot_steps`: Number of CoT steps.
- `--verbose`: If flag is included, print debugging output.

## Example
```sh
python main.py \
    --model huggyllama/llama-7b \
    --seed 0
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/0/ 
```

We also have several example scripts to run experiments in various settings in the
`scripts` directory.

## Experiments
#### Pruning Methods
* Wanda, SparseGPT
#### Models
* Llama 2-Chat 7B, LLaMA 7B

#### Calibration Datasets Used
##### Text:
* C4, WikiText
##### Arithmetic QA:
* GSM8K, SVAMP, MAWPS
##### Natural Language Inference:
* e-SNLI, ANLI R1, ANLI R3
##### Commonsense QA:
* CommonSenseQA, RACE, WinoGrande

## Citation
