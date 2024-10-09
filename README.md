# Is C4 Dataset Optimal for Pruning? An Investigation of Calibration Data for LLM Pruning
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

## Example
```sh
python main.py \
    --model huggyllama/llama-7b \
    --prune_method wanda \
    --sparsity_ratio 0.5 \
    --sparsity_type unstructured \
    --save out/llama_7b/2-4/wanda/ 
```

## Experiments
#### Calibration Datasets Used
##### Text:
* C4
* wikitext
##### Arithmetic QA:
* GSM8K
* SVAMP
* MAWPS
##### Natural Language Inference:
* ESNLI
* ANLI R1
* ANLI R3
##### Commonsense QA:
* CommonsenseQA
* RACE
* winogrande
