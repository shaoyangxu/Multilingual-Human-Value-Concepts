import json
import random
import math
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
seed=1
random.seed(seed)

draw_colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
draw_colors_models = ["gold", "orange","darkgoldenrod", "lime", "limegreen", "forestgreen", "cyan", "deepskyblue" , "royalblue"] # "aqua", "deepskyblue", "royalblue"
draw_hatchs = ['/', '', '|', '-', '+', 'x', 'o', 'O', '.', '*']

llama_exclude = ["ny","sw","ta","te"]
bloomz_exclude = ["ja","ko","fi","hu"]
qwen_exclude = ["ny","sw","ta","te"]
shared_langs = ["en", "zh", "fr", "es", "pt", "vi", "ca", "id"]

ratio = {
    "llama":{
        "ca": "0.04",
        "en": "89.70",
        "fr": "0.16",
        "id": "0.03",
        "pt": "0.09",
        "zh": "0.13",
        "es": "0.13",
        "vi": "0.08",
        "ja": "0.10",
        "ko": "0.06",
        "fi": "0.03",
        "hu": "0.03"
    },
    "qwen":{
        "ca": "0.04", #
        "en": "50",
        "fr": "0.16",
        "id": "0.03", # 
        "pt": "0.09",
        "zh": "60",
        "es": "0.13",
        "vi": "0.08",
        "ja": "0.10",
        "ko": "0.06",
        "fi": "0.03", # 
        "hu": "0.03" #
    },
    "bloomz":{
        "ca": "1.10206",
        "ny": "0.00007", # 
        "en": "30.03774",
        "fr": "12.89844",
        "id": "1.23708",
        "pt": "4.91041",
        "zh": "16.16741",
        "es": "10.84550",
        "sw": "0.01465", # 
        "ta": "0.49485", #
        "te": "0.18541", # 
        "vi": "2.70733"
    }
}

def is_high(lang1, lang2, model):
    if "llama" in model:
        high_langs = ["en"]
    elif "qwen" in model:
        high_langs = ["zh", "en"]
    elif "bloom" in model:
        high_langs = ["en", "zh", "fr", "es", "pt", "vi", "id", "ca"]
    
    if lang1 in high_langs and lang2 in high_langs:
        return "all_high"
    elif lang1 not in high_langs and lang2 not in high_langs:
        return "all_low"
    else:
        return "cross"

def sorted_by_ratio(langs, model_name):
    if "llama" in model_name:
        ratio_dict = ratio["llama"]
    if "bloomz" in model_name:
        ratio_dict = ratio["bloomz"]
    if "qwen" in model_name:
        ratio_dict = ratio["qwen"]

    langs = list(sorted(langs, key=lambda x:float(ratio_dict[x]), reverse=True))
    # if "qwen" in model_name:
    #     new_langs = ["en", "zh", "fr", "es", "pt", "vi", "ca", "id", "fi", "hu", "ja", "ko"]
    return langs

def get_new_models(model):
    if model == "llama2-chat-7B":
        return "LLaMA2-chat-7B"
    elif model == "llama2-chat-13B":
        return "LLaMA2-chat-13B"
    elif model ==  "llama2-chat-70B":
        return "LLaMA2-chat-70B"
    elif model == "qwen-chat-1B8":
        return "Qwen-chat-1B8"
    elif model == "qwen-chat-7B":
        return "Qwen-chat-7B"
    elif model == "qwen-chat-14B":
        return "Qwen-chat-14B"
    elif model == "bloomz-560M":
        return "BLOOMZ-560M"
    elif model == "bloomz-1B7":
        return "BLOOMZ-1B7"
    elif model == "bloomz-7B1":
        return "BLOOMZ-7B1"
    
def get_new_langs(langs, model_name):
    new_langs = []
    for lang in langs:
        is_in = True
        if "llama" in model_name and lang in llama_exclude:
            is_in = False
        if "bloomz" in model_name and lang in bloomz_exclude:
            is_in = False
        if "qwen" in model_name and lang in qwen_exclude:
            is_in = False
        if is_in:
            new_langs.append(lang)
            
    new_langs = sorted_by_ratio(new_langs, model_name)

    return new_langs

def get_hidden_layers(model_name):
    if model_name == "llama2-chat-7B":
        hidden_layers = list(range(1, 32 + 1))
    elif "llama2-chat-13B" in model_name:
        hidden_layers = list(range(1, 40 + 1))
    elif "llama2-chat-70B" in model_name:
        hidden_layers = list(range(1, 80 + 1))
    elif model_name == "bloomz-560M":
        hidden_layers = list(range(1, 24 + 1))
    elif model_name == "bloomz-1B1":
        hidden_layers = list(range(1, 24 + 1))
    elif model_name == "bloomz-1B7":
        hidden_layers = list(range(1, 24 + 1))
    elif model_name == "bloomz-3B":
        hidden_layers = list(range(1, 30 + 1))
    elif model_name == "bloomz-7B1":
        hidden_layers = list(range(1, 30 + 1))
    elif model_name == "qwen-chat-1B8":
        hidden_layers = list(range(1, 24 + 1))
    elif model_name == "qwen-chat-7B":
        hidden_layers = list(range(1, 32 + 1))
    elif model_name == "qwen-chat-14B":
        hidden_layers = list(range(1, 40 + 1))
    elif model_name == "qwen-chat-1B8-before":
        hidden_layers = list(range(1, 24 + 1))
    elif model_name == "qwen-chat-7B-before":
        hidden_layers = list(range(1, 32 + 1))
    elif model_name == "qwen-chat-14B-before":
        hidden_layers = list(range(1, 40 + 1))
    return hidden_layers

def rename_model(m):
    strs = m.split("-")
    n = strs[0]
    if n == "llama2":
        n = "LLaMA2"
    elif n == "bloomz":
        n = "BLOOMZ"
    elif n == "qwen":
        n = "Qwen"
    s = strs[-1]
    return n, s

def refine_template(template, concept="", model_name=""):
    # if model_name == "llama2-chat" and concept == "harmfulness":
    #     print("Attention: with [INST] tokens")
    #     return "[INST] {input} [/INST] "
    # else:
    return template

def load_model_tokenizer(model_name, model_size):
    model_path = ""
    if model_name == "llama2":
        if model_size == "7B":
            model_path = "/data/syxu/data/pretrained-models/llama2/7B"
        elif model_size == "13B":
            model_path = "/data/syxu/data/pretrained-models/llama2/13B"
        template = "{input}"
    elif model_name == "llama2-chat":
        if model_size == "7B":
            model_path = "/data/syxu/data/pretrained-models/llama2/7B-chat"
        elif "13B" in model_size:
            model_path = "/data/syxu/data/pretrained-models/llama2/13B-chat"
        elif "70B" in model_size:
            model_path = "/data/cuimenglong/LLaMA/llama-2-70b-chat-hf/"
        # template = "{input}" # bad concept recognition performance
        template = "[INST] {input} [/INST] "
    elif model_name == "bloomz":
        if model_size == "560M":
            model_path = "/data/syxu/data/pretrained-models/bloomz/560M"
        elif model_size == "1B1":
            model_path = "/data/syxu/data/pretrained-models/bloomz/1B1"
        elif model_size == "1B7":
            model_path = "/data/syxu/data/pretrained-models/bloomz/1B7"
        elif model_size == "3B":
            model_path = "/data/syxu/data/pretrained-models/bloomz/3B"
        elif model_size == "7B1":
            model_path = "/data/syxu/data/pretrained-models/bloomz/7B1"
        template = "{input}"
    elif model_name == "bloom":
        if model_size == "7B1":
            model_path = "/data/cordercorder/llm-sft/data/pretrained_models/bloom-7b1"
        template = "{input}"
    elif model_name == "qwen-chat":
        if model_size == "1B8":
            model_path = "/data1/cuimenglong/huggingface/models/Qwen-1_8B-Chat/"
        elif model_size == "7B":
            model_path = "/data1/cuimenglong/huggingface/models/Qwen-7B-Chat/"
        elif model_size == "14B":
            model_path = "/data1/cuimenglong/huggingface/models/Qwen-14B-Chat/"
        elif model_size == "72B":
            model_path = "/data1/cuimenglong/huggingface/models/Qwen-72B-Chat/"
        # template = "<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n" # bad concept recognition performance
        template = "{input}"
    # build
    if "llama2" in model_name:
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", token=True).eval()
        use_fast_tokenizer = "LlamaForCausalLM" not in model.config.architectures
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=use_fast_tokenizer, padding_side="left", legacy=False, token=True)
        tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
        tokenizer.bos_token_id = 1
    elif "bloom" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto").eval()
    elif "qwen" in model_name:
        tokenizer = AutoTokenizer.from_pretrained(
                                        model_path,
                                        pad_token='<|extra_0|>',
                                        eos_token='<|endoftext|>',
                                        padding_side='left',
                                        trust_remote_code=True
                                    )
        model = AutoModelForCausalLM.from_pretrained(
                                        model_path,
                                        pad_token_id=tokenizer.pad_token_id,
                                        device_map="auto",
                                        trust_remote_code=True
                                    ).eval()
        # model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="auto").eval()
    return model, tokenizer, template

def data_count(data, sub_split=False):
    cot = 0
    cot_dict = {}
    if not sub_split:
        for k, v in data.items():
            cot += len(v)
        return cot
    else:
        for k, v in data.items():
            cot_dict[k] = len(v)
        return cot_dict

def get_pair(sample, concept, template):
    pos, neg = sample['pos'], sample['neg']
    if concept == "deontology":
        scenario = sample['scenario']
        pos = scenario + " " + pos
        neg = scenario + " " + neg
    elif concept == "truthfulness":
        question = sample['question']
        pos = "Question: " + question + " " + "Answer: " + pos
        neg = "Question: " + question + " " + "Answer: " + neg
    else:
        assert len(sample) == 3
    return template.format(input=pos), template.format(input=neg)

def filter(data):
    new_data = {}
    filter_num = 0
    for k1, v1 in data.items():
        new_data[k1] = {}
        for k2, v2 in v1.items():
            if v2['pos'] != v2['neg']:
                new_data[k1][k2] = v2
            else:
                filter_num += 1
    return filter_num, new_data

def read_data(concept, lang="en", template=None, split=0.8, sub_split=False, random_direction=False):
    seed=1
    random.seed(seed)
    data_path = f"data/{concept}/{concept}_{lang}.json"
    with open(data_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    filter_num, data = filter(data)
    cot = data_count(data, sub_split=sub_split)

    train_num = math.floor(cot * 0.8)
    test_num = cot - train_num

    if split <= 0.8:
        train_num = math.floor(cot * split) # constant test set, changeable train set
    elif split >=1 : # sample num
        train_num = min(int(split), train_num)

    if not sub_split:
        samples = []
        for k1, v1 in data.items():
            for k2, v2 in v1.items():
                new_key = k1 + "|" + k2
                v2['key'] = new_key
                samples.append(v2)
    else:
        samples = []
    random.shuffle(samples)
    paired_data = []
    for sample in samples:
        pos, neg = get_pair(sample, concept, template)
        paired_data.append([pos, neg])

    train_data, test_data = paired_data[: train_num], paired_data[-test_num: ]
    assert len(test_data) == test_num

    train_label = []
    for d in train_data:
        pos = d[0]
        random.shuffle(d)
        train_label.append([s == pos for s in d])
    if random_direction:
        for label in train_label:
            random.shuffle(label)

    train_data = np.concatenate(train_data).tolist()
    test_data = np.concatenate(test_data).tolist()
    test_label = [[1,0]* len(test_data)]

    print("-"*10 + concept + " "+ lang +"-"*10)
    print(f"train_num({split}):{train_num} | test_num(0.2):{test_num} | filter_num:{filter_num}")

    return {
        "train": {"data": train_data, "labels": train_label},
        "test": {"data": test_data, "labels": test_label},
    }