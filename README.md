# [EMNLP 2024] Is C4 Dataset Optimal for Pruning? An Investigation of Calibration Data for LLM Pruning
https://arxiv.org/abs/2410.07461

If you find this repository useful, please consider citing:
```
@article{bandari2024c4datasetoptimalpruning,
      title={Is C4 Dataset Optimal for Pruning? An Investigation of Calibration
Data for LLM Pruning}, 
      author={Abhinav Bandari and Lu Yin and Cheng-Yu Hsieh and Ajay Kumar
Jaiswal and Tianlong Chen and Li Shen and Ranjay Krishna and Shiwei Liu},
      year={2024},
      eprint={2410.07461},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.07461}, 
}
```

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
- `--prune_method`: Choices are ["magnitude", "wanda", "sparsegpt", "none"].
- `--sparsity_ratio`: Denotes the percentage of weights to be pruned.
- `--sparsity_type`: Specifies the type of sparsity [`unstructured`, `2:4`, `4:8`].
- `--save`: Specifies the directory where the result will be stored
- `--calibration`: choices=['c4', 'oscar', 'redpajama', 'pile', 'gsm8k', 'svamp', 'mawps', 'anli_r1', 'anli_r2', 'anli_r3', 'esnli', 'rte', 'boolq', 'commonsense_qa', 'race', 'winogrande', 'wmt14', 'ellipses', 'random']
- `--seed`: Seed for sampling the calibration data. Default is 0.
- `--nsamples`: Number of calibration samples. Default=128.
- `--cache_dir`: File path of directory to cache weights.
  Default="llm_weights".
- `--input_format`: Default is 'concat'. Choices=['single', 'concat', 'zero'].
- `--seqlen`: Length of context window in tokens. Default is 2048.
- `--data_seqlen`: Number of meaningful tokens in each calibration sample, the
  remaining portion of context window is filled with padding tokens.
- `--num_incontext`: Number of in-context Q-A pairs in each calibration sample.
- `--num_cot_steps`: Number of CoT reasoning steps for each Q-A pair in
  calibration samples. Only used if `--rationale` is included.
- `--rationale`: If flag is included, include CoT rationale in answer portion
  of Q-A pairs in calibration samples.
- `--eval_rationale`: If flag is included, at evaluation time, include CoT
  rationale in in-context examples in prompt.
- `--eval`: Default is 'wikitext'. Choices=['wikitext', 'redpajama', 'oscar', 'gsm8k', 'svamp', 'mawps', 'anli_r1', 'anli_r2', 'anli_r3', 'esnli', 'rte', 'boolq',
                'commonsense_qa', 'race', 'winogrande', 'all']
- `--skip_dense_eval`: If flag is included, skip evaluation of dense model
  (before pruning).
- `--verbose`: If this flag is included, print intermediate results to stdout.
- `--append_to_file`: File to append results to.
- `--save_model`: Path to save the pruned model.

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
* C4, Pile, Oscar, RedPajama
##### Arithmetic QA:
* GSM8K, SVAMP, MAWPS
##### Natural Language Inference:
* e-SNLI, ANLI R1, ANLI R3
##### Commonsense QA:
* CommonSenseQA, RACE, WinoGrande

