import argparse
from utils import read_data, load_model_tokenizer, refine_template
import os
from repe import repe_pipeline_registry
from transformers import pipeline
import tqdm
import time
import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--concept",
        type=str,
        required=True,
        nargs="+",
        choices=["deontology","fairness", "harmfulness", "morality", "toxicity", "truthfulness", "utilitarianism"]
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        nargs="+",
        choices=["en", "ca", "fr", "id", "pt", "zh", "es", "vi", "ny", "sw", "ta", "te", "ja", "ko", "fi", "hu"]
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="llama2-chat",
        choices=["llama2", "llama2-chat", "bloomz", "bloom", "qwen-chat"] #, "bloom", "bloomz"]
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="7B",
        choices=["7B", "13B", "70B", "560M", "1B1", "1B7", "3B", "7B1", "1B8", "14B"] # 560m, 1b1, 1b7, 3b, 7b
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.8
    )
    parser.add_argument(
        "--random-direction",
        action="store_true"
    )
    args = parser.parse_args()
    return args

def get_data(lang, concept, template, split, random_direction):
    return read_data(
        concept=concept,
        lang=lang,
        template=template,
        split=split,
        sub_split=False,
        random_direction=random_direction
    )

def get_rep_reader(train_data, read_pipeline, hidden_layers, direction_method, n_difference):
    return read_pipeline.get_directions(
        train_data['data'], 
        rep_token=-1, 
        hidden_layers=hidden_layers, 
        n_difference=n_difference, 
        train_labels=train_data['labels'], 
        direction_method=direction_method,
    )


if __name__ == "__main__":
    args = get_args()
    repe_pipeline_registry()

    concepts = args.concept
    langs = args.lang
    split = args.split
    model_name = args.model_name
    model_size = args.model_size
    random_direction = args.random_direction
    direction_method = 'cluster_mean'
    n_difference = 0
    model, tokenizer, template = None, None, None
    if split == 0.8:
        save_dir = os.path.join("res", f"{model_name}-{model_size}")
    else:
        save_dir = os.path.join("res", f"{model_name}-{model_size}-{split}")
    os.makedirs(save_dir, exist_ok=True)
    for concept in concepts:
        concept_save_dir = os.path.join(save_dir, concept)
        os.makedirs(concept_save_dir, exist_ok=True)
        if random_direction:
            concept_reader_path = os.path.join(concept_save_dir, "random_direction_reader_dict.pk")
        else:
            concept_reader_path = os.path.join(concept_save_dir, "reader_dict.pk")
        t1 = time.time()
        # create reader
        if not os.path.exists(concept_reader_path):
            if model == None:
                model, tokenizer, template = load_model_tokenizer(model_name, model_size)
                read_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
                template = refine_template(template)
                hidden_layers = list(range(1, model.config.num_hidden_layers + 1))
            reader_dict = {}
            pbar = tqdm.tqdm(langs)
            for lang in pbar:
                data = get_data(lang, concept, template, split, random_direction)
                pbar.set_description("create reader")
                rep_reader = get_rep_reader(train_data=data['train'],
                                            read_pipeline=read_pipeline,
                                            hidden_layers=hidden_layers,
                                            direction_method=direction_method,
                                            n_difference=n_difference)
                reader_dict[lang] = rep_reader
            torch.save(reader_dict, concept_reader_path)
        print("time passed:", time.time() - t1)