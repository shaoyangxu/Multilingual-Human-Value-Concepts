import argparse
from utils import read_data, load_model_tokenizer, refine_template, get_hidden_layers, draw_colors_models, get_new_models, rename_model
import os
from repe import repe_pipeline_registry
from transformers import pipeline
import numpy as np
import tqdm
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
plt.rc('font',family='Times New Roman')
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
    parser.add_argument(
        "--cross-model", # llama2-chat-7B,llama2-chat-13B,llama2-chat-70B,qwen-chat-1B8,qwen-chat-7B,qwen-chat-14B,bloomz-560M,bloomz-1B7,bloomz-7B1
        type=str,
        default="",
        help="analyze multilingual concept recognition accuracy of all model"
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

def compute_acc(langs, hidden_layers, reader_dict, data_dict, to_classify_langs=None):
    res_dict = {}
    if to_classify_langs == None:
        to_classify_langs = langs

    all_num = len(langs) * len(to_classify_langs)
    with tqdm.tqdm(total=all_num) as pbar:

        pbar.set_description('Cross lingual classification')
        for l2 in to_classify_langs: # target
            l2_hidden_state = read_pipeline._batched_string_to_hiddens(
                data_dict[l2]['test']['data'], 
                rep_token=-1,
                hidden_layers=hidden_layers, 
                batch_size=4,
                which_hidden_states=None)
            for l1 in langs: # source
                if l1 not in res_dict:
                    res_dict[l1] = {}
                rep_reader = reader_dict[l1]
                test_data = data_dict[l2]['test']
                H_tests = read_pipeline(
                    test_data['data'], 
                    rep_token=-1,
                    hidden_layers=hidden_layers, 
                    rep_reader=rep_reader,
                    batch_size=32,
                    precomputed_hidden_state=l2_hidden_state)
                results = {}
                for layer in hidden_layers:
                    H_test = [H[layer] for H in H_tests] 
                    H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]

                    sign = rep_reader.direction_signs[layer]
                    eval_func = min if sign == -1 else max
                    
                    cors = np.mean([eval_func(H) == H[0] for H in H_test])
                    results[layer] = cors
                res_dict[l1][l2] = results
                pbar.update(1)
    return res_dict

def draw_res_all(langs, res_dict, save_name):
    fig, ax = plt.subplots(figsize=(28, 6))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1)  # 设置左边框的线宽
    ax.spines['bottom'].set_linewidth(1)  # 设置底边框的线宽

    max_res_dict = {}
    min_, max_ = 100, -100
    concepts = list(res_dict.keys())
    concept_res_dict_ = {}
    for concept in concepts:
        concept_res_dict_[concept] = {}
        for model in res_dict[concept]:
            concept_res_dict_[concept][model] = {}
            for l1 in langs:
                concept_res_dict_[concept][model][l1] = {}
                for l2 in langs:
                    res = max(list(res_dict[concept][model][l1][l2].values()))
                    concept_res_dict_[concept][model][l1][l2] = res

    for model in res_dict[concepts[0]].keys():
        max_res_dict[model] = {}
        for l1 in langs:
            max_res_dict[model][l1] = {}
            for l2 in langs:
                concept_res_lst = [concept_res_dict_[c][model][l1][l2] for c in concepts]
                concept_res = sum(concept_res_lst) / len(concept_res_lst)
                max_res_dict[model][l1][l2] = concept_res
            min_ = min(max_res_dict[model][l1][l1], min_)
            max_ = max(max_res_dict[model][l1][l1], max_)

    bar_width = 0.085
    model_interval = 0.01 # 间隔最大为1
    color_dict = {}
    self_acc_dict = {}
    start_dict = {}

    llama_labels = [i for i in max_res_dict if "llama" in i]
    bloom_labels = [i for i in max_res_dict if "bloom" in i]
    qwen_labels = [i for i in max_res_dict if "qwen" in i]
    min_len = min(len(llama_labels), len(bloom_labels))
    min_len = min(min_len, len(qwen_labels))
    labels = []
    for i in range(min_len):
        labels.append(llama_labels[i])
        labels.append(qwen_labels[i])
        labels.append(bloom_labels[i])
    labels.extend(llama_labels[min_len:])
    labels.extend(qwen_labels[min_len:])
    labels.extend(bloom_labels[min_len:])

    for idx, model in enumerate(max_res_dict):
        self_acc = []
        trans_mean_acc = []
        trans_min_acc = []
        trans_max_acc = []
        for l1 in langs:
            self_acc.append(max_res_dict[model][l1][l1])
            trans_acc = []
            for l2 in langs:
                if l1 != l2:
                    trans_acc.append(max_res_dict[model][l2][l1])
            mean_acc = sum(trans_acc)/len(trans_acc)
            max_acc = max(trans_acc)
            min_acc = min(trans_acc)
            trans_mean_acc.append(mean_acc)
            trans_min_acc.append(max(0, mean_acc-min_acc))
            trans_max_acc.append(max(0, max_acc-mean_acc))
        index = np.arange(len(langs))

        self_acc_dict[model] = self_acc
        color_dict[model] = draw_colors_models[idx]
        start_dict[model] = index + (bar_width + model_interval) * idx

    for idx, model in enumerate(labels):
        index = np.arange(len(langs))

        plt.bar(start_dict[model], self_acc_dict[model], bar_width, color=color_dict[model], edgecolor="black", label=get_new_models(model), zorder=2) #  hatch='//',

    compare_models = ['llama2-chat-7B', 'qwen-chat-7B', 'bloomz-7B1']
    for idx in range(len(start_dict[labels[0]])):
        x_lst = []
        y_lst = []
        for model in compare_models:
            x_lst.append(start_dict[model][idx].item())
            y_lst.append(self_acc_dict[model][idx].item())
            plt.plot(x_lst, y_lst, color='black', marker='o', linestyle='--', linewidth=1)
            
    plt.ylim((max(min_ - 0.01, 0), min(max_ + 0.01, 1)))
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    # 添加标签和标题
    plt.ylabel('Accuracy', fontsize=30)
    plt.xticks(index + ( (bar_width+model_interval) /2 ) * (len(max_res_dict) -1), langs)
    plt.tick_params(labelsize=26)
    bbox_to_anchor = (0.5, 1.21)
    def percentage_formatter(x, pos):
        return f'{x*100:.0f}%'
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    legend = ax.legend(prop = {'size':23}, ncols=3, loc='upper center', bbox_to_anchor=bbox_to_anchor, framealpha=1)
    legend.get_frame().set_linewidth(0)
    
    plt.grid(axis="y", zorder=2)
    legend.set_zorder(1)
    print(save_name+".jpg")
    plt.savefig(save_name+".jpg")
    plt.savefig(save_name+".svg")
    plt.cla()

def convert_latex_table(langs, res_dict, concept=""):
    max_res_dict = {}
    for model in res_dict:
        max_res_dict[model] = {}
        for l1 in langs:
            max_res_dict[model][l1] = {}
            for l2 in langs:
                res = max(list(res_dict[model][l1][l2].values()))
                max_res_dict[model][l1][l2] = res
    model_size_res_dict = {}
    for model in max_res_dict:
        mn, ms = rename_model(model)
        if mn not in model_size_res_dict:
            model_size_res_dict[mn] = {}
        model_size_res_dict[mn][ms] = max_res_dict[model]
    lines = []
    lines.append(r"\begin{subtable}[b]{\textwidth}")
    lines.append("\centering")
    lines.append(r"\resizebox{0.95\columnwidth}!{")
    lines.append(r"\begin{tabular}{l" + "r|" +"c" * (len(langs)) + "|c}") # no @{}
    lines.append(r"\toprule")
    strr = r"\multicolumn{2}{l|}{\textbf{" + concept.capitalize() +"}}"
    for lang in langs:
        strr += r" & {\textbf{"
        strr += lang
        strr += "}}"
    strr += r"& {\textbf{Avg}}"
    lines.append(strr + r"\\")
    lines.append(r"\hline")
    for mn in model_size_res_dict:
        m_num = len(model_size_res_dict[mn])
        strr = r"\multirow{" + str(m_num) + "}{*}"
        if mn in ["LLaMA2", "Qwen"]:
            strr += r"{\textbf{\makecell[c]{" + mn +r" \\ -chat}}}"
        else:
            strr += r"{\textbf{"+ mn +"}}"
        for idx, ms in enumerate(model_size_res_dict[mn]):
            if idx == 0:
                strr += f"& {ms}"
            else:
                strr = "{} & " + ms
            res_lst = []
            for l1 in langs:
                res = round(model_size_res_dict[mn][ms][l1][l1]*100, 1)
                res_lst.append(res)
                strr += f"& {res} "
            mean_res = round(sum(res_lst) / len(res_lst), 1)
            strr += f"& {mean_res}"
            lines.append(strr + r"\\")
        lines.append(r"\hline")
    
    lines.pop(-1) # "\hline"
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append("}")
    lines.append(r"\end{subtable}")
    print("\n".join(lines))

if __name__ == "__main__":
    args = get_args()
    repe_pipeline_registry()

    concepts = args.concept
    langs = args.lang
    split = args.split
    model_name = args.model_name
    model_size = args.model_size
    random_direction = args.random_direction
    cross_model = args.cross_model
    model, tokenizer, template = None, None, None
    if split == 0.8:
        save_dir = os.path.join("res", f"{model_name}-{model_size}")
    else:
        save_dir = os.path.join("res", f"{model_name}-{model_size}-{split}")
    os.makedirs(save_dir, exist_ok=True)
    if not cross_model:
        for concept in concepts:
            concept_save_dir = os.path.join(save_dir, concept)
            os.makedirs(concept_save_dir, exist_ok=True)
            if random_direction:
                concept_reader_path = os.path.join(concept_save_dir, "random_direction_reader_dict.pk")
                concept_acc_path = os.path.join(concept_save_dir, "random_direction_acc.pk")
            else:
                concept_reader_path = os.path.join(concept_save_dir, "reader_dict.pk")
                concept_acc_path = os.path.join(concept_save_dir, "acc.pk")
            assert os.path.exists(concept_reader_path), "concept_reader_path not exist!"
            reader_dict = torch.load(concept_reader_path)
            t1 = time.time()
            # recognize concept
            if not os.path.exists(concept_acc_path):
                if model == None:
                    model, tokenizer, template = load_model_tokenizer(model_name, model_size)
                    read_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
                    template = refine_template(template)
                    hidden_layers = list(range(1, model.config.num_hidden_layers + 1))
                    data_dict = {}
                    pbar = tqdm.tqdm(langs)
                    for lang in pbar:
                        pbar.set_description("read data")
                        data = get_data(lang, concept, template, split, random_direction)
                        data_dict[lang] = data
                res_dict = compute_acc(langs, hidden_layers, reader_dict, data_dict)
                torch.save(res_dict, concept_acc_path)
            else:
                data = torch.load(concept_acc_path)
            print("time passed:", time.time() - t1)
    else:
        model_name_lst = cross_model.split(",")
        concept_res_dict = {}
        for concept in concepts:
            model_res_dict = {}
            for model_name in model_name_lst:
                hidden_layers = get_hidden_layers(model_name)
                if split == 0.8:
                    save_dir = os.path.join("res", model_name)
                else:
                    save_dir = os.path.join("res", f"{model_name}-{split}")

                concept_save_dir = os.path.join(save_dir, concept)
                concept_acc_path = os.path.join(concept_save_dir, "acc.pk")

                res_dict = torch.load(concept_acc_path)
                model_res_dict[model_name] = res_dict
            convert_latex_table(langs, model_res_dict, concept)
            concept_res_dict[concept] = model_res_dict
            key = "|".join(model_name_lst)
            if split == 0.8:
                save_dir = os.path.join("res", key)
            else:
                save_dir = os.path.join("res", f"{key}-{split}")
            os.makedirs(save_dir, exist_ok=True)


        draw_save_name = os.path.join(save_dir, f"recognition_accuracy_of_all_model")
        draw_res_all(langs, concept_res_dict, draw_save_name)