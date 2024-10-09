# Code adapted from https://github.com/IST-DASLab/sparsegpt/blob/master/datautils.py

import numpy as np
import string
import json
import re
import random
import torch
from datasets import load_dataset, Dataset, DatasetDict
import pdb

DATASET_ROOT = 'datasets'

class DatasetLoader(object):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None):
        self.data_root = DATASET_ROOT
        self.dataset_name = dataset_name
        self.source_dataset_name = source_dataset_name
        self.dataset_version = dataset_version
        self.has_valid = has_valid
        self.split_map = split_map

        self.batch_size = batch_size
        self.train_batch_idxs = train_batch_idxs
        self.test_batch_idxs = test_batch_idxs
        self.valid_batch_idxs = valid_batch_idxs

        assert self.split_map is not None


    def load_from_source(self):
        if self.source_dataset_name is None:
            self.source_dataset_name = self.dataset_name
        if self.dataset_version is None:
            datasets = load_dataset(self.source_dataset_name)
        else:
            datasets = load_dataset(self.source_dataset_name, self.dataset_version)
        return datasets


    def to_json(self, datasets):
        for k, v in self.split_map.items():
            datasets[v].to_json(f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_{k}.json')


    def load_from_json(self):
        data_files = {
            'train': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_train.json',
            'test': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_test.json',
        }

        if self.has_valid:
            data_files.update({'valid': f'{self.data_root}/{self.dataset_name}/{self.dataset_name}_valid.json',})

        datasets = load_dataset('json', data_files=data_files)
        datasets = self._post_process(datasets)

        # subsample training dataset if needed
        num_train = len(datasets['train'])
        idxs = list()
        for idx in self.train_batch_idxs:
            idxs += range(idx*self.batch_size, (idx+1)*self.batch_size)
        datasets['train'] = Dataset.from_dict(datasets['train'][[idx for idx in idxs if idx < num_train]])

        return datasets


    def load_llm_preds(self, split):
        labels = list()
        rationales = list()
        for idx in getattr(self, f'{split}_batch_idxs'):
            with open(f'{self.data_root}/{self.dataset_name}/llm/{split}_CoT_{idx}.json') as f:
                outputs = json.load(f)

            for output in outputs:
                rationale, label = self._parse_llm_output(output)

                rationales.append(rationale)
                labels.append(label)

        return rationales, labels


    def load_gpt_preds(self, split):
        labels = list()
        rationales = list()

        with open(f'{self.data_root}/gpt-neox/{self.dataset_name}/{split}.json') as f:
            outputs = json.load(f)

        for output in outputs:
            rationale, label = self._parse_gpt_output(output)

            rationales.append(rationale)
            labels.append(label)

        return rationales, labels


    def _post_process(self, datasets):
        raise NotImplementedError


    def _parse_llm_output(self, output):
        raise NotImplementedError


    def _parse_gpt_output(self, output):
        raise NotImplementedError

class SVAMPDatasetLoader(DatasetLoader):
    def __init__(self):
        dataset_name = 'svamp'
        source_dataset_name = 'svamp'
        dataset_version = None
        has_valid = False
        split_map = {
            'train': 'train',
            'test': 'test',
        }
        batch_size = 500
        train_batch_idxs = range(2)
        test_batch_idxs = range(1)


        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=None)


    def load_from_source(self):
        with open(f'{self.data_root}/{self.dataset_name}/SVAMP.json') as f:
            original_dataset = json.load(f)

        dataset = list()
        for data in original_dataset:
            input = f'{data["Body"]}\n{data["Question"]}'
            equation = data["Equation"]

            dataset.append({
                'input': input,
                'label': equation,
            })

        idxs = np.random.RandomState(seed=0).permutation(len(dataset))
        print('len(dataset) ', len(dataset))
        train_idxs = idxs[:800]
        test_idxs = idxs[800:]

        train_dataset = Dataset.from_list(np.array(dataset)[train_idxs].tolist())
        test_dataset = Dataset.from_list(np.array(dataset)[test_idxs].tolist())

        datasets = DatasetDict({
            'train': train_dataset,
            'test': test_dataset
        })

        return datasets


    def _post_process(self, datasets):
        return datasets


    def _parse_llm_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label

        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label

    def _parse_gpt_output(self, output):
        rationale_label = output.split('Q:')[0]
        rationale_label = rationale_label.rstrip().lstrip()
        try:
            rationale, label = rationale_label.split('The answer is')
        except:
            rationale = ' '
            label = ' '
            return rationale, label

        rationale = rationale.rstrip()
        try:
            label = re.search(r'\(.*\)', label).group(0)
        except:
            label = ' '

        return rationale, label

class ANLIDatasetLoader(DatasetLoader):
    def __init__(self, dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                 batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs):

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def label_idx2text(example):
            if example['label'] == 0:
                example['label'] = 'entailment'
            elif example['label'] == 1:
                example['label'] = 'neutral'
            elif example['label'] == 2:
                example['label'] = 'contradiction'
            return example

        datasets = datasets.map(label_idx2text)
        datasets = datasets.remove_columns(['uid', 'reason'])

        return datasets

    def _parse_llm_output(self, output):
        try:
            rationale, label = output.split("Premise:")[0].rstrip().split("So the answer is")
        except:
            rationale = ''
            label = ''

        rationale = rationale.rstrip()
        label = label.lstrip()[:-1]

        return rationale, label

    def _parse_gpt_output(self, output):
        try:
            rationale, label = output.split("Premise:")[0].rstrip().lstrip().split("So the answer is")
        except:
            try:
                rationale, label = output.split("Premise:")[0].rstrip().lstrip().split("The answer is")
            except:
                rationale = ''
                label = ''

        rationale = rationale.rstrip()
        label = label.lstrip()[:-1]

        return rationale, label

class ANLI1DatasetLoader(ANLIDatasetLoader):
    def __init__(self):
        dataset_name = 'anli1'
        source_dataset_name = 'anli'
        dataset_version = None
        has_valid = True
        split_map = {
            'train': 'train_r1',
            'valid': 'dev_r1',
            'test': 'test_r1',
        }
        batch_size = 5000
        train_batch_idxs = range(4)
        test_batch_idxs = range(1)
        valid_batch_idxs = range(1)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

class ESNLIDatasetLoader(DatasetLoader):
    def __init__(self, subset='full'):
        dataset_name = 'esnli'
        source_dataset_name = 'esnli'
        dataset_version = None
        has_valid = True
        split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        batch_size = 5500
        if subset == 'full':
            train_batch_idxs = range(100)
        elif subset == 'small':
            train_batch_idxs = range(10)
        else:
            raise ValueError
        test_batch_idxs = range(2)
        valid_batch_idxs = range(2)

        super().__init__(dataset_name, source_dataset_name, dataset_version, has_valid, split_map,
                         batch_size, train_batch_idxs, test_batch_idxs, valid_batch_idxs=valid_batch_idxs)

    def _post_process(self, datasets):

        def prepare_input(example):
            if example['label'] == 0:
                example['label'] = 'entailment'
            elif example['label'] == 1:
                example['label'] = 'neutral'
            elif example['label'] == 2:
                example['label'] = 'contradiction'

            return example

        datasets = datasets.map(prepare_input)
        datasets = datasets.remove_columns(['explanation_1', 'explanation_2', 'explanation_3'])

        return datasets

    def _parse_llm_output(self, output):
        rationale = output.split("Answer:")[0].rstrip()
        try:
            label = output.split("Answer: ")[1].split("Premise")[0].rstrip()
        except:
            label = ' '

        return rationale, label

    def _parse_gpt_output(self, output):
        rationale = output.split("Answer:")[0].rstrip().lstrip()
        try:
            label = output.split("Answer: ")[1].split("Premise")[0].rstrip()
        except:
            label = ' '

        return rationale, label

# Set seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

# Wrapper for tokenized input IDs
class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

# Load and process wikitext2 dataset
def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    # Load train and test datasets
    traindata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', verification_mode='no_checks')
    testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test', verification_mode='no_checks')

    # Encode datasets
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_random(nsamples, seed, seqlen, tokenizer):
    print("get_random")
    trainloader = []
    for _ in range(nsamples):
        trainenc_concat = None
        while True:
            random_text = ''.join(random.choice(string.ascii_uppercase + string.ascii_lowercase + string.digits) for _ in range(1000))
            trainenc = tokenizer(random_text, return_tensors='pt')
            if trainenc_concat is None:
                trainenc_concat = trainenc
            else:
                for key in trainenc_concat:
                    trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

            if trainenc_concat.input_ids.shape[1] > seqlen:
                trainenc = trainenc_concat
                break

        inp = trainenc.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, None, None

def get_ellipses(nsamples, seed, seqlen, tokenizer):
    print("get_ellipses")
    ellipses = "." * 1000
    trainloader = []
    for _ in range(nsamples):
        trainenc_concat = None
        while True:
            trainenc = tokenizer(ellipses, return_tensors='pt')
            if trainenc_concat is None:
                trainenc_concat = trainenc
            else:
                for key in trainenc_concat:
                    trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

            if trainenc_concat.input_ids.shape[1] > seqlen:
                trainenc = trainenc_concat
                break

        inp = trainenc.input_ids[:, :seqlen]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    return trainloader, None, None
    
# Load and process c4 dataset
def get_c4(nsamples, seed, seqlen, tokenizer, data_seqlen=None):
    print("get_c4")
    # Load train and validation datasets
    version_commit_sha = "607bd4c8450a42878aa9ddc051a65a055450ef87"
    traindata = load_dataset('allenai/c4', revision=version_commit_sha, data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train', verification_mode='no_checks')
    valdata = load_dataset('allenai/c4', revision=version_commit_sha, data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation', verification_mode='no_checks')
    pad_token = tokenizer(traindata[0]['text'], return_tensors='pt', padding='max_length', max_length=100000).input_ids[0][0]
    print('pad_token', pad_token)

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if data_seqlen is not None and trainenc.input_ids.shape[1] > data_seqlen:
                break
            if trainenc.input_ids.shape[1] > seqlen:
                break
        if data_seqlen is not None:
            i = random.randint(0, trainenc.input_ids.shape[1] - data_seqlen - 1)
            j = i + data_seqlen
        else:
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen

        inp = trainenc.input_ids[:, i:j]
        if data_seqlen is not None:
            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)

        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata[:1100]['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc, None

def get_redpajama(nsamples, seed, seqlen, tokenizer):
    print("get_redpajama")
    # Load train and validation datasets
    data = load_dataset('togethercomputer/RedPajama-Data-V2', 'sample', split='train', verification_mode='no_checks')
    print("len(data)", len(data))
    traindata = data #load_dataset('togethercomputer/RedPajama-Data-V2', 'sample', split='train', verification_mode='no_checks')
    #valdata = data

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['raw_content'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    """
    valenc = tokenizer(' '.join(valdata[:1100]['raw_content']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    """
    valenc = None
    return trainloader, valenc, None

def get_oscar(nsamples, seed, seqlen, tokenizer):
    print("get_oscar")
    # Load train and validation datasets
    traindata = load_dataset('oscar', 'unshuffled_deduplicated_en', split='train[:5000]', verification_mode='no_checks')
    valdata = load_dataset('oscar', 'unshuffled_deduplicated_en', split='train[5000:10000]', verification_mode='no_checks')
    print('len(traindata)', len(traindata))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            print('i=', i)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc, None

def get_pile(nsamples, seed, seqlen, tokenizer):
    print("get_pile")
    # Load train and validation datasets
    traindata = load_dataset('monology/pile-uncopyrighted', split='train[:5000]', verification_mode='no_checks')
    valdata = load_dataset('monology/pile-uncopyrighted', split='train[5000:5100]', verification_mode='no_checks')
    print('len(traindata)', len(traindata))

    # Generate samples from training set
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            print('i=', i)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    # Prepare validation dataset
    valenc = tokenizer(' '.join(valdata['text']), return_tensors='pt')
    valenc = valenc.input_ids[:, :(256 * seqlen)]
    valenc = TokenizerWrapper(valenc)
    return trainloader, valenc, None


def get_gsm8k(nsamples, seed, seqlen, tokenizer, rationale=False, padding_side='left', input_format='single', mode='llama', num_incontext=None, num_cot_steps=None, verbose=False):
    def prepare_input(example):
        question = example['question']
        answer = example['answer']
        rationale, label = answer.split('####')
        rationale, label = rationale.strip(), label.strip()

        example['prompt'] = f'Question: {question}\nAnswer: '
        example['label'] = label
        example['rationale'] = rationale
        example['rationale_label'] = f'{rationale}. So the answer is {label}.'
        example['question_label'] = f'Question: {question}\nAnswer: {label}'
        example['question_rationale_label'] = f'Question: {question}\nAnswer: {rationale}. So the answer is {label}.'

        return example

    traindata = load_dataset('gsm8k', 'main', split='train', verification_mode='no_checks')
    #traindata_iterable = load_dataset('gsm8k', 'main', split='train', streaming=True)
    #traindata_shuffled = traindata_iterable.shuffle(seed=seed)
    #traindata_len = 1000
    traindata_len = len(traindata)
    #traindata = traindata_shuffled.take(traindata_len)
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('gsm8k', 'main', split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['question'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['question'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]


    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            #print('concat')
            trainenc_concat = None
            idx = 0
            num_examples = num_incontext if num_incontext is not None else float('inf')
            while idx < num_examples:
                i = random.randint(0, traindata_len - 1)
                idx += 1
                if rationale:
                    trainenc = tokenizer(traindata[i]['question_rationale_label'], return_tensors='pt')
                    if num_cot_steps:
                        num_steps = traindata[i]['rationale_label'].count('. ')
                        print('num_steps', num_steps)
                        if num_cot_steps != num_steps:
                            continue
                else:
                    trainenc = tokenizer(traindata[i]['question_label'], return_tensors='pt')
                #print('tokenized data')

                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            print(f'{idx} in-context samples')
            if num_incontext is not None:
                inp = trainenc_concat.input_ids
                if padding_side == 'left':
                    inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                elif padding_side == 'right':
                    inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)
            else:
                i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
                j = i + seqlen
                inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'single':
            i = random.randint(0, traindata_len - 1)
            if rationale:
                trainenc = tokenizer(traindata[i]['question_rationale_label'], padding='max_length', max_length=seqlen, return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['question_label'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))
            
        elif input_format == 'zero':
            i = random.randint(0, traindata_len - 1)
            trainenc = tokenizer(traindata[i]['prompt'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))


    test_prompt = testdata[:nsamples]['prompt']
    test_answer = testdata[:nsamples]['label']

    return trainloader, (test_prompt, test_answer), None

    

def get_svamp_rationale(nsamples, seed, seqlen, tokenizer, input_format='single', padding_side='left', mode='llama', verbose=False):
    print('get_svamp_rationale')
    dataset_loader_svamp = SVAMPDatasetLoader()
    datasets_svamp = dataset_loader_svamp.load_from_json()

    llm_rationales_svamp = {}
    llm_labels_svamp = {}
    question_rationale_label = {}
    prompts = {}
    labels = {}
    for split in ['train', 'test']:
        llm_rationales_svamp[split], llm_labels_svamp[split] = dataset_loader_svamp.load_llm_preds(split=split)

        question_rationale_label[split] = np.char.array(datasets_svamp[split]['input']) + '\nAnswer: ' + np.char.array(
            llm_rationales_svamp[split]) + '. So the answer is ' + datasets_svamp[split]['label']
        prompts[split] = 'Question: ' + np.char.array(datasets_svamp[split]['input']) + '.\nAnswer: '
        labels[split] = np.char.array(datasets_svamp[split]['label'])

    traindata = Dataset.from_dict({'question_rationale_label': question_rationale_label['train'],
                                   'prompt': prompts['train'],
                                   'label': labels['train']})
    testdata = Dataset.from_dict({'question_rationale_label': question_rationale_label['test'],
                                   'prompt': prompts['test'],
                                   'label': labels['test']})
    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['question_rationale_label'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['question_rationale_label'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['question_rationale_label'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['question_rationale_label'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['question_rationale_label'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:nsamples]['prompt']
    test_answer = testdata[:nsamples]['label']

    return trainloader, (test_prompt, test_answer), pad_token
def get_svamp_individual(tokenizer, split='train'):
    def prepare_input(example):
        question = example['Body'] + ' ' + example['Question']
        answer = example['Equation']

        example['prompt'] = f'Question: {question}\nAnswer: '
        example['answer'] = answer
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example
    
    data = load_dataset('ChilleD/SVAMP', split=split, verification_mode='no_checks')
    data = traindata.map(prepare_input)

    return data

def get_svamp(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    if rationale:
        get_svamp_rationale(nsamples, seed, seqlen, tokenizer, input_format=input_format, padding_side=padding_side, mode=mode, verbose=verbose)
    def prepare_input(example):
        question = example['Body'] + ' ' + example['Question']
        answer = example['Equation']

        example['prompt'] = f'Question: {question}\nAnswer: '
        example['answer'] = answer
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('ChilleD/SVAMP', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('ChilleD/SVAMP', split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_anli(nsamples, seed, seqlen, tokenizer, difficulty = 1, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    classes = ['entailment', 'neutral', 'contradiction']
    def prepare_input(example):
        premise = example['premise']
        hypothesis = example['hypothesis']
        answer = classes[example['label']]

        example['prompt'] = f'Premise: {premise}\nHypothesis: {hypothesis}\nLabel: '
        example['answer'] = answer
        example['answer_rationale'] = example['reason'] + ' So the answer is ' + answer + '.'
        example['prompt_answer'] = example['prompt'] + example['answer']
        example['prompt_answer_rationale'] = example['prompt'] + example['answer_rationale']

        return example

    version_commit_sha = 'bf206833154d4fcaf5e3b01b8bf17d4d15213cb1'
    traindata = load_dataset('anli', split='train_r{}'.format(difficulty), verification_mode='no_checks', revision=version_commit_sha)
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('anli', split='test_r{}'.format(difficulty), verification_mode='no_checks', revision=version_commit_sha)
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                if rationale:
                    trainenc = tokenizer(traindata[i]['prompt_answer_rationale'], return_tensors='pt')
                else:
                    trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')

                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            if rationale:
                trainenc = tokenizer(traindata[i]['prompt_answer_rationale'], return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            if rationale:
                trainenc = tokenizer(traindata[i]['prompt_answer_rationale'], return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_esnli(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    classes = ['entailment', 'neutral', 'contradiction']
    def prepare_input(example):
        premise = example['premise']
        hypothesis = example['hypothesis']
        answer = classes[example['label']]

        example['prompt'] = f'Premise: {premise}\nHypothesis: {hypothesis}\nLabel: '
        example['answer'] = answer
        example['answer_rationale'] = example['explanation_1'] + ' So the answer is ' + answer + '.'
        example['prompt_answer'] = example['prompt'] + example['answer']
        example['prompt_answer_rationale'] = example['prompt'] + example['answer_rationale']

        return example

    traindata = load_dataset('esnli', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('esnli', split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                if rationale:
                    trainenc = tokenizer(traindata[i]['prompt_answer_rationale'], return_tensors='pt')
                else:
                    trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')

                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            if rationale:
                trainenc = tokenizer(traindata[i]['prompt_answer_rationale'], return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))
        elif input_format == 'zero':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_esnli_old(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    dataset_loader_esnli = ESNLIDatasetLoader()
    datasets_esnli = dataset_loader_esnli.load_from_json()

    llm_rationales_svamp = {}
    llm_labels_svamp = {}
    premise_hypothesis_rationale_label = {}
    premise_hypothesis_label = {}
    prompts = {}
    labels = {}
    for split in ['train', 'test']:
        llm_rationales_svamp[split], llm_labels_svamp[split] = dataset_loader_esnli.load_llm_preds(split=split)

        premise_hypothesis_rationale_label[split] = 'Premise: ' + np.char.array(datasets_esnli[split]['premise']) + '\nHypothesis: ' + np.char.array(datasets_esnli[split]['hypothesis']) + '\nLabel: ' + np.char.array(llm_rationales_svamp[split]) + '. So the answer is ' + datasets_esnli[split]['label']
        prompts[split] = 'Premise: ' + np.char.array(datasets_esnli[split]['premise']) + '\nHypothesis: ' + np.char.array(datasets_esnli[split]['hypothesis']) + '\nLabel: '
        labels[split] = np.char.array(datasets_esnli[split]['label'])
        premise_hypothesis_label[split] = prompts[split] + labels[split]

    traindata = Dataset.from_dict({'premise_hypothesis_rationale_label': premise_hypothesis_rationale_label['train'],
                                   'premise_hypothesis_label': premise_hypothesis_label['train'],
                                   'prompt': prompts['train'],
                                   'label': labels['train']})
    testdata = Dataset.from_dict({'premise_hypothesis_rationale_label': premise_hypothesis_rationale_label['test'],
                                  'premise_hypothesis_label': premise_hypothesis_label['test'],
                                  'prompt': prompts['test'],
                                  'label': labels['test']})
    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['premise_hypothesis_rationale_label'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['premise_hypothesis_rationale_label'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                if rationale:
                    trainenc = tokenizer(traindata[i]['premise_hypothesis_rationale_label'], return_tensors='pt')
                else:
                    trainenc = tokenizer(traindata[i]['premise_hypothesis_label'], return_tensors='pt')

                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            if rationale:
                trainenc = tokenizer(traindata[i]['premise_hypothesis_rationale_label'], return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['premise_hypothesis_label'], return_tensors='pt')

            answer_start_enc = tokenizer(['\nLabel:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            if rationale:
                trainenc = tokenizer(traindata[i]['premise_hypothesis_rationale_label'], padding='max_length', max_length=seqlen, return_tensors='pt')
            else:
                trainenc = tokenizer(traindata[i]['premise_hypothesis_label'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['label']

    return trainloader, (test_prompt, test_answer), pad_token

def get_mawps(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        question = example['question']
        answer = example['expression']

        example['prompt'] = f'Question: {question}\nAnswer: '
        example['answer'] = answer
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('MU-NLPC/Calc-mawps', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('MU-NLPC/Calc-mawps', split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_rte(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        premise = example['text1']
        hypothesis = example['text2']
        label = example['label_text']

        example['prompt'] = f'Premise: {premise}\nHypothesis: {hypothesis}\nLabel: '
        example['answer'] = label
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('yangwang825/rte', 'main', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('yangwang825/rte', 'main', split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_boolq(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        question = example['passage'] + example['question']
        label = str(example['answer'])

        example['prompt'] = f'Question: {question}\nAnswer: '
        example['answer'] = label
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('boolq', 'main', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('boolq', 'main', split='validation', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_commonsense_qa(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        question = example['question']
        choices = example['choices']
        label = example['answerKey']
        choices_formatted = ''
        for idx, letter in enumerate(choices['label']):
            choices_formatted += f'{letter}. {choices["text"][idx]}'
            if idx < len(choices['label']) - 1:
                choices_formatted += '\n'

        example['prompt'] = f'Question: {question}\nChoices:\n{choices_formatted}\nAnswer: '
        example['answer'] = label
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('commonsense_qa', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('commonsense_qa', split='validation', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_race(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        article = example['article']
        question = example['question']
        options = example['options']
        label = example['answer']
        options_formatted = ''
        letters = 'ABCDE'
        for idx, option in enumerate(options):
            options_formatted += f'{letters[idx]}. {option}\n'

        example['prompt'] = f'Article: {article}\nQuestion: {question}\nOptions:\n{options_formatted}\nAnswer: '
        example['answer'] = label
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('race', 'all', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('race', 'all', split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_winogrande(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        sentence = example['sentence']
        option1 = example['option1']
        option2 = example['option2']
        answer = example['answer']
        options_formatted = f'1. {option1}\n2. {option2}'

        example['prompt'] = f'Sentence: {sentence}\nOptions:\n{options_formatted}\nAnswer: '
        example['answer'] = answer
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('winogrande', 'winogrande_m', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('winogrande', 'winogrande_m', split='validation', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))

        elif input_format == 'autoregressive':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
            answer_start_enc = tokenizer(['\nAnswer:'])
            train_ids = trainenc.input_ids
            if mode == 'gpt2':
                answer_ids = torch.tensor(answer_start_enc.input_ids)
            elif mode == 'llama':
                # Todo: fix hardcoding later
                answer_ids = torch.tensor([[313, 29871]])
            for i in range(train_ids.shape[1] - answer_ids.shape[1] + 1):
                found = True
                for j in range(answer_ids.shape[1]):
                    if train_ids[0][i+j] != answer_ids[0][j]:
                        found = False
                        break
                if found:
                    k = 0
                    while len(trainloader) < nsamples and k < train_ids.shape[1] - i:
                        inp = train_ids[:, :i+1+k]
                        if padding_side == 'left':
                            inp = torch.nn.functional.pad(inp, (seqlen - inp.shape[1], 0), 'constant', pad_token)
                        elif padding_side == 'right':
                            inp = torch.nn.functional.pad(inp, (0, seqlen - inp.shape[1]), 'constant', pad_token)

                        tar = inp.clone()
                        tar[:, :-1] = -100
                        trainloader.append((inp, tar))
                        k += 1

        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))

    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_wmt14(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        translation = example['translation']
        #print('type(translation)', type(translation))
        english_version = translation["en"]
        french_version = translation["fr"]

        example['prompt'] = f'English:{english_version}\nFrench:'
        example['answer'] = french_version
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('wmt/wmt14', 'fr-en', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('wmt/wmt14', 'fr-en', split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))
    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token

def get_iwslt(nsamples, seed, seqlen, tokenizer, rationale=False, input_format='concat', padding_side='left', mode='llama', verbose=False):
    def prepare_input(example):
        translation = example['translation']
        #print('type(translation)', type(translation))
        english_version = translation["en"]
        french_version = translation["fr"]

        example['prompt'] = f'English:{english_version}\nFrench:'
        example['answer'] = french_version
        example['prompt_answer'] = example['prompt'] + example['answer']

        return example

    traindata = load_dataset('iwslt2017', 'iwslt2017-en-fr', split='train', verification_mode='no_checks')
    traindata = traindata.map(prepare_input)

    testdata = load_dataset('iwslt2017', 'iwslt2017-en-fr', split='test', verification_mode='no_checks')
    testdata = testdata.map(prepare_input)

    if padding_side == 'left':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][0]
    elif padding_side == 'right':
        pad_token = tokenizer(traindata[0]['prompt_answer'], return_tensors='pt', padding='max_length', max_length=seqlen).input_ids[0][-1]
    #print('pad_token', pad_token)

    random.seed(seed)
    trainloader = []
    while len(trainloader) < nsamples:
        if input_format == 'concat':
            trainenc_concat = None
            while True:
                i = random.randint(0, len(traindata) - 1)
                trainenc = tokenizer(traindata[i]['prompt_answer'], return_tensors='pt')
                if trainenc_concat is None:
                    trainenc_concat = trainenc
                else:
                    for key in trainenc_concat:
                        trainenc_concat[key] = torch.concat([trainenc_concat[key], trainenc[key]], dim=1)

                if trainenc_concat.input_ids.shape[1] > seqlen:
                    trainenc = trainenc_concat
                    break
            i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
            j = i + seqlen
            inp = trainenc.input_ids[:, i:j]
            tar = inp.clone()
            tar[:, :-1] = -100
            trainloader.append((inp, tar))
        elif input_format == 'single':
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['prompt_answer'], padding='max_length', max_length=seqlen, return_tensors='pt')

            inp = trainenc.input_ids
            tar = inp.clone()
            tar[:, :-1] = -100

            trainloader.append((inp, tar))
    test_prompt = testdata[:]['prompt']
    test_answer = testdata[:]['answer']

    return trainloader, (test_prompt, test_answer), pad_token


# Function to select the appropriate loader based on dataset name
def get_loaders(name, nsamples=128, seed=0, seqlen=2048, data_seqlen=None, tokenizer=None, rationale=False, input_format='single', difficulty=1, padding_side='left', mode='llama', num_incontext=None, num_cot_steps=None, verbose=False):
    if 'wikitext2' in name:
        return get_wikitext2(nsamples, seed, seqlen, tokenizer)
    if "c4" in name:
        return get_c4(nsamples, seed, seqlen, tokenizer, data_seqlen=data_seqlen)
    if "redpajama" in name:
        return get_redpajama(nsamples, seed, seqlen, tokenizer)
    if "oscar" in name:
        return get_oscar(nsamples, seed, seqlen, tokenizer)
    if "pile" in name:
        return get_pile(nsamples, seed, seqlen, tokenizer)
    if 'gsm8k' in name:
        return get_gsm8k(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, num_incontext=num_incontext, num_cot_steps=num_cot_steps, verbose=verbose)
    if 'svamp' in name:
        return get_svamp(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'mawps' in name:
        return get_mawps(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'anli' in name:
        return get_anli(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, difficulty=int(name[-1]), padding_side=padding_side, mode=mode, verbose=verbose)
    if 'esnli' in name:
        return get_esnli(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'rte' in name:
        return get_rte(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'boolq' in name:
        return get_boolq(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'commonsense_qa' in name:
        return get_commonsense_qa(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'race' in name:
        return get_race(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'winogrande' in name:
        return get_winogrande(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'wmt14' in name:
        return get_wmt14(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'iwslt' in name:
        return get_iwslt(nsamples, seed, seqlen, tokenizer, input_format=input_format, rationale=rationale, padding_side=padding_side, mode=mode, verbose=verbose)
    if 'ellipses' in name:
        return get_ellipses(nsamples, seed, seqlen, tokenizer)
    if 'random' in name:
        return get_random(nsamples, seed, seqlen, tokenizer)

