import argparse
from utils import read_data, load_model_tokenizer, refine_template, draw_colors, draw_colors_models, rename_model, get_hidden_layers, get_new_langs, is_high, get_new_models
import os
from repe import repe_pipeline_registry
from transformers import pipeline
import matplotlib
import numpy as np
import tqdm
import time
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import AxesGrid
from matplotlib.patheffects import withStroke
import math


lang_score = torch.load("lang_sim.pt")
type_lst = ['genetic', 'geographic', 'syntactic', 'phonological'] # 'inventory', 'featural'
plt.rc('font',family='Times New Roman')
# /home/syxu/anaconda3/envs/bm/lib/python3.9/site-packages/matplotlib/mpl-data/fonts/ttf/
"""
deontology fairness harmfulness morality toxicity truthfulness utilitarianism
morality deontology utilitarianism fairness truthfulness toxicity harmfulness

en ca fr id pt zh es vi ny sw ta te ja ko fi hu
llama:
    en fr zh es ja pt vi ko ca id fi hu 
bloom:
    en zh fr es pt vi id ca ta te sw ny
share:
    en fr zh es vi pt ca id
llama-all:
    en fr zh es pt vi ca id ja ko fi hu ta te sw ny
model:
    llama2-chat-7B,llama2-chat-13B,llama2-chat-70B,qwen-chat-1B8,qwen-chat-7B,qwen-chat-14B,bloomz-560M,bloomz-1B7,bloomz-7B1
"""
# python run.py --concept deontology fairness harmfulness morality toxicity truthfulness utilitarianism --lang en fr zh es ja pt vi ko ca id fi hu --model-name llama2-chat --model-size 7B --draw

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
        "--draw",
        action="store_true"
    )
    parser.add_argument(
        "--cross-model",
        type=str,
        default="",
    )
    parser.add_argument(
        "--not-load",
        action="store_true"
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

def analysis_sim(layers, langs, sim_dict, model_name):
    new_langs = get_new_langs(langs, model_name)

    # print("analysis_sim: only 8 shared!")
    # new_langs = "en fr zh es vi pt ca id".split()
    all_sim = [] # [layer1, layer2, ..]
    lang_means = {} # {"l1": [mean all layer]}
    for i, layer_idx in enumerate(layers):
        layer_sim = [] # len(langs) ^ 2 
        for l1 in new_langs:
            lang_sim = [] # 1 target lang with all source langs
            for l2 in new_langs:
                if l1 == l2:
                    continue
                cosine_similarity = sim_dict[layer_idx][l1][l2]
                lang_sim.append(cosine_similarity)
            layer_sim.extend(lang_sim)
            if l1 not in lang_means:
                lang_means[l1] = []
            lang_means[l1].append(sum(lang_sim)/len(lang_sim))
        all_sim.append(layer_sim)
    return all_sim, lang_means

def draw_acc(hidden_layers, langs, res_dict, save_name):

    def mscatter(x,y,ax=None, m=None, **kw):
        import matplotlib.markers as mmarkers
        if not ax: ax=plt.gca()
        sc = ax.scatter(x,y,**kw)
        if (m is not None) and (len(m)==len(x)):
            paths = []
            for marker in m:
                if isinstance(marker, mmarkers.MarkerStyle):
                    marker_obj = marker
                else:
                    marker_obj = mmarkers.MarkerStyle(marker)
                path = marker_obj.get_path().transformed(
                            marker_obj.get_transform())
                paths.append(path)
            sc.set_paths(paths)
        return sc
    
    fig, ax = plt.subplots()

    x_lst = []
    y_lst = []
    c_lst = []
    m_lst = []
    cm = plt.get_cmap()
    num = len(langs)
    for x_idx, l1 in enumerate(langs):
        for c_idx, l2 in enumerate(langs):
            max_res = max(list(res_dict[l2][l1].values())) # max_res = max(list(res_dict[l1][l2].values()))
            x_lst.append(x_idx)
            y_lst.append(max_res)
            c_lst.append(cm(1.*c_idx/num))
            if l1 == l2:
                m_lst.append("*")
            else:
                m_lst.append("o")
    mscatter(x=x_lst, y=y_lst, c=c_lst, m=m_lst, ax=ax)
    plt.xticks([index for index in list(range(len(langs)))], langs)
    print(save_name+".jpg")
    plt.savefig(save_name+".jpg")
    plt.savefig(save_name+".svg")
    plt.cla()

def draw_acc2(hidden_layers, langs, res_dict, save_name):
    plt.figure(figsize=(25, 4))
    max_res_dict = {}

    min_, max_ = 100, -100
    for l1 in langs:
        max_res_dict[l1] = {}
        for l2 in langs:
            res = max(list(res_dict[l1][l2].values()))
            min_ = min(res, min_)
            max_ = max(res, max_)
            max_res_dict[l1][l2] = res

    self_acc = []
    trans_mean_acc = []
    trans_min_acc = []
    trans_max_acc = []
    for l1 in langs:
        self_acc.append(max_res_dict[l1][l1])
        trans_acc = []
        for l2 in langs:
            if l1 != l2:
                trans_acc.append(max_res_dict[l2][l1])
        mean_acc = sum(trans_acc)/len(trans_acc)
        max_acc = max(trans_acc)
        min_acc = min(trans_acc)
        trans_mean_acc.append(mean_acc)
        trans_min_acc.append(mean_acc-min_acc)
        trans_max_acc.append(max_acc-mean_acc)

    # 设置柱状图的宽度
    bar_width = 0.2

    # 计算柱状图的位置
    index = np.arange(len(langs))

    # 绘制柱状图
    plt.bar(index, self_acc, bar_width, label='self')
    plt.bar(index + bar_width, trans_mean_acc, bar_width, yerr=[trans_min_acc, trans_max_acc], capsize=5, label='trans')
    plt.ylim((min_ - 0.01, max_ + 0.01))
    # 添加标签和标题
    plt.xlabel('Languages')
    plt.ylabel('Performance')
    plt.title('Language Performance and Transfer Performance')
    plt.xticks(index + bar_width / 2, langs)
    plt.legend()
    print(save_name+".jpg")
    plt.savefig(save_name+".jpg")
    plt.savefig(save_name+".svg")
    plt.cla()

def draw_acc2_all(hidden_layers, langs, res_dict, save_name, all_concept=False, concept=""):
    fig, ax = plt.subplots(figsize=(28, 6))
    # plt.figure(figsize=(28, 6))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_linewidth(1)  # 设置左边框的线宽
    ax.spines['bottom'].set_linewidth(1)  # 设置底边框的线宽
    max_res_dict = {}
    min_, max_ = 100, -100
    if not all_concept:
        for model in res_dict:
            # print(model)
            # print(len(res_dict[model].keys()))
            max_res_dict[model] = {}
            for l1 in langs:
                max_res_dict[model][l1] = {}
                for l2 in langs:
                    res = max(list(res_dict[model][l1][l2].values()))
                    min_ = min(res, min_)
                    max_ = max(res, max_)
                    max_res_dict[model][l1][l2] = res
    else:
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
    trans_mean_acc_dict = {}
    trans_min_acc_dict = {}
    trans_max_acc_dict = {}
    start_dict = {}
    llama_labels = [i for i in max_res_dict if "llama" in i]
    bloom_labels = [i for i in max_res_dict if "bloom" in i]
    qwen_labels = [i for i in max_res_dict if "qwen" in i]
    model_cot = 0
    model_cot = model_cot + 1 if len(llama_labels) > 0 else model_cot
    model_cot = model_cot + 1 if len(bloom_labels) > 0 else model_cot
    model_cot = model_cot + 1 if len(bloom_labels) > 0 else model_cot
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
        trans_mean_acc_dict[model] = trans_mean_acc
        trans_min_acc_dict[model] = trans_min_acc
        trans_max_acc_dict[model] = trans_max_acc
        color_dict[model] = draw_colors_models[idx]
        start_dict[model] = index + (bar_width + model_interval) * idx

    for idx, model in enumerate(labels):
        # 设置柱状图的宽度
        # 计算柱状图的位置
        index = np.arange(len(langs))

        # 绘制柱状图
        # plt.bar(index + (bar_width*2 + model_interval) * idx, self_acc, bar_width, color=draw_colors_models[idx], edgecolor="black", hatch='//')
        # plt.bar(index + bar_width + (bar_width*2 + model_interval) * idx, trans_mean_acc, bar_width, yerr=[trans_min_acc, trans_max_acc], capsize=5, label=model, color=draw_colors_models[idx], edgecolor="black")
        plt.bar(start_dict[model], self_acc_dict[model], bar_width, color=color_dict[model], edgecolor="black", label=get_new_models(model), zorder=2) #  hatch='//',

    compare_models = ['llama2-chat-7B', 'qwen-chat-7B', 'bloomz-7B1']
    for idx in range(len(start_dict[labels[0]])):
        x_lst = []
        y_lst = []
        for model in compare_models:
            x_lst.append(start_dict[model][idx].item())
            y_lst.append(self_acc_dict[model][idx].item())
            plt.plot(x_lst, y_lst, color='black', marker='o', linestyle='--', linewidth=1)
            
    print(min_)
    print(max_)
    # if concept == "harmfulness":
    #     plt.ylim((max(min_ - 0.5, 0), min(max_ + 0.5, 1)))
    # else:
    plt.ylim((max(min_ - 0.01, 0), min(max_ + 0.01, 1)))
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    # 添加标签和标题
    # plt.xlabel('Languages', fontsize=25)
    plt.ylabel('Accuracy', fontsize=30)
    # plt.title('Accuracy and Cross-lingual Accuracy', )
    # plt.xticks(index + bar_width / 2 + (bar_width + model_interval /2 ) * (len(max_res_dict) -1), langs)
    plt.xticks(index + ( (bar_width+model_interval) /2 ) * (len(max_res_dict) -1), langs)
    plt.tick_params(labelsize=26)

    # legend = ax.legend(labels, prop = {'size':23}, ncols=3, loc='upper center', bbox_to_anchor=(0.5, 1.17), labels=color_lst)
    # if concept == "harmfulness":
    #     bbox_to_anchor = (0.5, 1.30)
    # else:
    bbox_to_anchor = (0.5, 1.21)
    # if concept == "harmfulness":
    #     bbox_to_anchor = (0.5, 1.23)
    def percentage_formatter(x, pos):
        return f'{x*100:.0f}%'
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    legend = ax.legend(prop = {'size':23}, ncols=3, loc='upper center', bbox_to_anchor=bbox_to_anchor, framealpha=1)
    # legend.set_alpha(1)
    # for i, text in enumerate(legend.get_texts()):
    #     text.set_color(color_dict[text.get_text()])
    legend.get_frame().set_linewidth(0)
    
    plt.grid(axis="y", zorder=2)
    legend.set_zorder(1)

    # plt.tight_layout()
    print(save_name+".jpg")
    plt.savefig(save_name+".jpg")
    plt.savefig(save_name+".svg")
    plt.cla()

def draw_acc_heatmap(langs, model_concept_res_dict, save_dir, save_name, per_model=False):

    class CenteredNorm(Normalize):  
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    max_res_dict = {}
    trans_res_dict = {}
    for model, concept_res_dict in model_concept_res_dict.items():
        new_langs = get_new_langs(langs, model)
        max_res_dict[model] = {}
        trans_res_dict[model] = {}
        # max
        for l1 in new_langs:
            max_res_dict[model][l1] = {}
            for l2 in new_langs:
                concept_res_lst = []
                for concept, res_dict in concept_res_dict.items():
                    res = max(list(res_dict[l1][l2].values()))
                    concept_res_lst.append(res)
                concept_mean = sum(concept_res_lst) / len(concept_res_lst)
                max_res_dict[model][l1][l2] = concept_mean
    # min_ = 1000
    # max_ = -1000
    # for idx, model in enumerate(max_res_dict):
    #     res_dict = max_res_dict[model]
    #     new_langs = get_new_langs(langs, model)
    #     for lang1 in new_langs:
    #         for lang2 in new_langs:
    #             diff = (res_dict[lang1][lang2] - res_dict[lang2][lang2])*100
    #             min_ = min(min_, diff)
    #             max_ = max(max_, diff)
    # print("min", min_)
    # print("max", max_)
    # for model, res_dict in max_res_dict.items():
    #     new_langs = get_new_langs(langs, model)
    #     data = []
    #     for lang1 in new_langs:
    #         line = []
    #         for lang2 in new_langs:
    #             # diff = res_dict[lang1][lang2] - res_dict[lang2][lang2]
    #             # if diff >= 0:
    #             #     res = 1
    #             # else:
    #             #     res = -1
    #             # line.append(res) # round((res_dict[lang1][lang2] - res_dict[lang2][lang2])*100,2)
    #             line.append(round((res_dict[lang1][lang2] - res_dict[lang2][lang2])*100,2))
    #         data.append(line)
    #     data = np.array(data)
    #     # midpoint = 0
    #     # from matplotlib import colormaps
    #     # colormaps = list(colormaps)#[:10]
    #     # colormaps = ["coolwarm"]
    #     # fig, axs = plt.subplots(len(colormaps), 1, figsize=(10, 2 * len(colormaps)))
    #     # for idx, color in enumerate(colormaps):
    #         # ax = axs[idx]
    #         # ax.set_title(color, fontsize=8)
    #     plt.xticks(np.arange(len(new_langs)), labels=new_langs)
    #     plt.yticks(np.arange(len(new_langs)), labels=new_langs)    
    #     plt.title("Change in Cross-lingual Classification Performance")
    #     for i in range(len(new_langs)):
    #         for j in range(len(new_langs)):
    #             text = plt.text(j, i, data[i, j], ha="center", va="center", color="w")
    #     plt.ylabel("source language")
    #     plt.xlabel("target language")
    #     heatmap = plt.imshow(data, cmap="coolwarm") # cmap='viridis') # , norm=CenteredNorm(midpoint=midpoint)
    #     plt.colorbar() # heatmap, ax=ax, orientation='horizontal'
    #     plt.tight_layout()
    #     # save_path = os.path.join(save_dir, "all_color" + "-" +model+"-"+save_name)
    #     # plt.savefig(save_path+".jpg")
    #     # plt.savefig(save_path+".svg")
    #     # print(save_path+".jpg")
    #     if per_model:
    #         save_path = os.path.join(save_dir, model+"-"+save_name)
    #         plt.savefig(save_path+".jpg")
    #         plt.savefig(save_path+".svg")
    #         print(save_path+".jpg")
    #         # plt.tight_layout(pad=1)
    #         # plt.cla()
    #         plt.clf()
    #     # exit()
    # model_lst = ["llama2-chat-7B", "bloomz-7B1"]
    
    # norm = Normalize(vmin=min_, vmax=max_)


    fig = plt.figure(figsize=(20, 15))
    # fig = plt.figure(figsize=(20, 10))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(3, 3),
                    axes_pad=(0.1,0.6),
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.15,
                    share_all=False,
                    label_mode="all",
                    )
    # fig.subplots_adjust(wspace=100)
    base = 50
    def rescale(num, base, min_, max_):
        a = 1 # min
        b = base # max, log_base^base = 1
        r = (b-a)*(num-min_)/(max_-min_) + a # https://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
        return r
    # for idx, model in enumerate(max_res_dict):
    #     ax = grid.axes_all[idx]
    #     new_langs = get_new_langs(langs, model)
    #     ax.set_xticks(np.arange(len(new_langs)) + idx, labels=new_langs)
    # pearson_corr_dict = {}
    for idx, model in enumerate(max_res_dict):
        ax = grid[idx]
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        res_dict = max_res_dict[model]
        new_langs = get_new_langs(langs, model)
        data = []
        trans = []
        no_trans = []
        for lang1 in new_langs:
            line = []
            for lang2 in new_langs:
                # diff = res_dict[lang1][lang2] - res_dict[lang2][lang2]
                # if diff >= 0:
                #     res = 1
                # else:
                #     res = -1
                # line.append(res) # round((res_dict[lang1][lang2] - res_dict[lang2][lang2])*100,2)
                diff = res_dict[lang1][lang2] - res_dict[lang2][lang2]
                line.append(diff)
                if diff >= 0:
                    trans.append(diff)
                else:
                    no_trans.append(abs(diff))
            data.append(line)
        data = np.array(data)


        # for t in type_lst:
        #     if t not in pearson_corr_dict:
        #         pearson_corr_dict[t] = {}
        #     type_score = []
        #     type_score_no_center = []
        #     for lang1 in new_langs:
        #         line = []
        #         line_no_center = []
        #         for lang2 in new_langs:
        #             score = lang_score[lang1][lang2][t]
        #             line.append(1-score)
        #             if lang1 != lang2:
        #                 line_no_center.append(1-score)
        #         type_score.append(line)
        #         type_score_no_center.append(line_no_center)
        #     type_score = np.array(type_score)
        #     type_score_no_center = np.array(type_score_no_center)
        #     correlation_matrix = np.corrcoef(data_no_center.flatten(), type_score_no_center.flatten())

        trans_min = min(trans)
        trans_max = max(trans)

        no_trans_min = min(no_trans)
        no_trans_max = max(no_trans)
        normalized_data = np.zeros_like(data)


            # ax.set_yticks(np.arange(len(new_langs)), labels=new_langs)
        ax.get_shared_y_axes().remove(ax)
        yticker = matplotlib.axis.Ticker()
        ax.yaxis.major = yticker

        yloc = matplotlib.ticker.AutoLocator()
        yfmt = matplotlib.ticker.ScalarFormatter()

        ax.yaxis.set_major_locator(yloc)
        ax.yaxis.set_major_formatter(yfmt)
        if idx in [0,3,6]:
            ax.set_yticks(np.arange(len(new_langs)), labels=new_langs)
            ax.set_yticklabels(new_langs, fontsize=16)
        else:
            ax.set_yticks([])
        ax.set_title(get_new_models(model), fontsize=18)
        # if i % 3 == 0:
        ax.get_shared_x_axes().remove(ax)
        xticker = matplotlib.axis.Ticker()
        ax.xaxis.major = xticker

        xloc = matplotlib.ticker.AutoLocator()
        xfmt = matplotlib.ticker.ScalarFormatter()

        ax.xaxis.set_major_locator(xloc)
        ax.xaxis.set_major_formatter(xfmt)
        # else:
        #     ax.set_yticks([])
            # ax.yaxis.set_major_locator(plt.NullLocator())
        ax.set_xticks(np.arange(len(new_langs))) # , labels=new_langs
        ax.set_xticklabels(new_langs, fontsize=16)
        for i in range(len(new_langs)):
            for j in range(len(new_langs)):
                text = ax.text(j, i, round(data[i, j]*100,2), ha="center", va="center", color="w")
                # text.set_path_effects([withStroke(linewidth=0.1, foreground='black')])
                item = data[i, j].item()
                if item >= 0:
                    normalized_data[i, j] = math.log(rescale(item, base, trans_min, trans_max), base) # 0-1 -> 1-2 -> log 2
                else:
                    normalized_data[i, j] = - math.log(rescale(abs(item), base, no_trans_min, no_trans_max), base)
                
                # if item >= 0:
                #     normalized_data[i, j] = (item - trans_min) / (trans_max - trans_min) # 0-1 -> 1-2 -> log 2
                # else:
                #     normalized_data[i, j] = - (abs(item) - no_trans_min) / (no_trans_max - no_trans_min)
        im = ax.imshow(normalized_data, cmap="coolwarm", vmin=-1, vmax=1) #e , vmin=min_, vmax=max_) # , norm=norm
    fig.text(0.5, 0.07, 'Target Languages', ha='center', va='center', fontsize=20)
    fig.text(0.2, 0.5, 'Source Languages', ha='center', va='center', rotation='vertical', fontsize=20)
    cbar = grid.cbar_axes[0].colorbar(im) # = cbar = ax.cax.colorbar(im)
    cbar.set_ticks([])
    # cbar.ax.text(1.5, 0.5, 'high', rotation=270, va='center', ha='left', fontsize=14)
    # cbar.ax.text(-0.5, 0.5, 'low', rotation=90, va='center', ha='right', fontsize=14)
    cbar.ax.set_yticks(np.arange(-1, 1.1, 1))
    cbar.ax.set_yticklabels(['Negative', 'Medium', 'Positive'], fontsize=18)
    # for idx, ax in enumerate(grid):
    #     # ax.set_axis_off()
    #     # im = ax.imshow(np.random.random((16,16)), vmin=0, vmax=1)

    # fig, axs = plt.subplots(1, len(model_lst), figsize=(10, 2 * len(model_lst)))
    # heatmap_lst = []
    # for idx, model in enumerate(model_lst):
    #     ax = axs[idx]
    #     res_dict = max_res_dict[model]
    #     new_langs = get_new_langs(langs, model)
    #     data = []
    #     for lang1 in new_langs:
    #         line = []
    #         for lang2 in new_langs:
    #             # diff = res_dict[lang1][lang2] - res_dict[lang2][lang2]
    #             # if diff >= 0:
    #             #     res = 1
    #             # else:
    #             #     res = -1
    #             # line.append(res) # round((res_dict[lang1][lang2] - res_dict[lang2][lang2])*100,2)
    #             line.append(round((res_dict[lang1][lang2] - res_dict[lang2][lang2])*100,2))
    #         data.append(line)
    #     data = np.array(data)
    #     # ax = axs[idx]
    #     # ax.set_title(color, fontsize=8)
    #     ax.set_xticks(np.arange(len(new_langs)), labels=new_langs)
    #     ax.set_yticks(np.arange(len(new_langs)), labels=new_langs)    
    #     # ax.set_title("Change in Cross-lingual Classification Performance")
    #     for i in range(len(new_langs)):
    #         for j in range(len(new_langs)):
    #             text = ax.text(j, i, data[i, j], ha="center", va="center", color="w")
    #     ax.set_ylabel("source language")
    #     ax.set_xlabel("target language")
    #     heatmap = ax.imshow(data, cmap="coolwarm") # cmap='viridis') # , norm=CenteredNorm(midpoint=midpoint)
    #     heatmap_lst.append(heatmap)
    # plt.colorbar(heatmap_lst, ax=axs) # heatmap, ax=ax, orientation='horizontal'
    # plt.tight_layout()
    save_path = os.path.join(save_dir, "all_model" + "-" +model+"-"+save_name)
    plt.savefig(save_path+".jpg")
    plt.savefig(save_path+".svg")
    print(save_path+".jpg")


def draw_acc_heatmap_7B(langs, model_concept_res_dict, save_dir, save_name, per_model=False):

    class CenteredNorm(Normalize):  
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
            self.midpoint = midpoint
            Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
            x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
            return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))
    max_res_dict = {}
    trans_res_dict = {}
    for model, concept_res_dict in model_concept_res_dict.items():
        new_langs = get_new_langs(langs, model)
        max_res_dict[model] = {}
        trans_res_dict[model] = {}
        # max
        for l1 in new_langs:
            max_res_dict[model][l1] = {}
            for l2 in new_langs:
                concept_res_lst = []
                for concept, res_dict in concept_res_dict.items():
                    res = max(list(res_dict[l1][l2].values()))
                    concept_res_lst.append(res)
                concept_mean = sum(concept_res_lst) / len(concept_res_lst)
                max_res_dict[model][l1][l2] = concept_mean


    fig = plt.figure(figsize=(20, 7))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=(0.4,0.6),
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.15,
                    share_all=False,
                    label_mode="all",
                    )
    base = 50
    def rescale(num, base, min_, max_):
        a = 1 # min
        b = base # max, log_base^base = 1
        r = (b-a)*(num-min_)/(max_-min_) + a # https://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
        return r
    for idx, model in enumerate(max_res_dict):
        ax = grid[idx]
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        res_dict = max_res_dict[model]
        new_langs = get_new_langs(langs, model)
        data = []
        trans = []
        no_trans = []
        for lang1 in new_langs:
            line = []
            for lang2 in new_langs:
                diff = res_dict[lang1][lang2] - res_dict[lang2][lang2]
                line.append(diff)
                if diff >= 0:
                    trans.append(diff)
                else:
                    no_trans.append(abs(diff))
            data.append(line)
        data = np.array(data)

        trans_min = min(trans)
        trans_max = max(trans)

        no_trans_min = min(no_trans)
        no_trans_max = max(no_trans)
        normalized_data = np.zeros_like(data)


            # ax.set_yticks(np.arange(len(new_langs)), labels=new_langs)
        ax.get_shared_y_axes().remove(ax)
        yticker = matplotlib.axis.Ticker()
        ax.yaxis.major = yticker

        yloc = matplotlib.ticker.AutoLocator()
        yfmt = matplotlib.ticker.ScalarFormatter()

        ax.yaxis.set_major_locator(yloc)
        ax.yaxis.set_major_formatter(yfmt)
        ax.set_yticks(np.arange(len(new_langs)), labels=new_langs)
        ax.set_yticklabels(new_langs, fontsize=16)

        ax.set_title(get_new_models(model), fontsize=23)
        ax.get_shared_x_axes().remove(ax)
        xticker = matplotlib.axis.Ticker()
        ax.xaxis.major = xticker

        xloc = matplotlib.ticker.AutoLocator()
        xfmt = matplotlib.ticker.ScalarFormatter()

        ax.xaxis.set_major_locator(xloc)
        ax.xaxis.set_major_formatter(xfmt)
        ax.set_xticks(np.arange(len(new_langs))) # , labels=new_langs
        ax.set_xticklabels(new_langs, fontsize=16)
        for i in range(len(new_langs)):
            for j in range(len(new_langs)):
                text = ax.text(j, i, round(data[i, j]*100,2), ha="center", va="center", color="w")
                item = data[i, j].item()
                if item >= 0:
                    normalized_data[i, j] = math.log(rescale(item, base, trans_min, trans_max), base) # 0-1 -> 1-2 -> log 2
                else:
                    normalized_data[i, j] = - math.log(rescale(abs(item), base, no_trans_min, no_trans_max), base)
        im = ax.imshow(normalized_data, cmap="coolwarm", vmin=-1, vmax=1) #e , vmin=min_, vmax=max_) # , norm=norm
        ax.tick_params(axis='both', which='major', labelsize=20)
    fig.text(0.5, 0.08, 'Target Languages', ha='center', va='center', fontsize=23)
    fig.text(0.095, 0.5, 'Source Languages', ha='center', va='center', rotation='vertical', fontsize=23)
    cbar = grid.cbar_axes[0].colorbar(im) # = cbar = ax.cax.colorbar(im)
    cbar.set_ticks([])
    cbar.ax.set_yticks(np.arange(-1, 1.1, 1))
    cbar.ax.set_yticklabels(['Negative', 'Medium', 'Positive'], fontsize=23)
    save_path = os.path.join(save_dir, "all_model" + "-" +model+"-"+save_name)
    plt.savefig(save_path+".jpg")
    plt.savefig(save_path+".svg")
    print(save_path+".jpg")

def draw_sim_heatmap(langs, model_concept_sim_dict, save_dir, save_name, per_model=False):
    max_sim_dict = {}
    for model, concept_sim_dict in model_concept_sim_dict.items():
        new_langs = get_new_langs(langs, model)
        max_sim_dict[model] = {}
        # max
        for l1 in new_langs:
            max_sim_dict[model][l1] = {}
            for l2 in new_langs:
                if l1 == l2:
                    concept_mean = 1
                else:
                    concept_sim_lst = []
                    for concept, sim_dict in concept_sim_dict.items():
                        # sim = sum(list(sim_dict[l1][l2].values())) / len(list(sim_dict[l1][l2].values()))
                        sim = max(list(sim_dict[l1][l2].values()))
                        concept_sim_lst.append(sim)
                    concept_mean = sum(concept_sim_lst) / len(concept_sim_lst)
                max_sim_dict[model][l1][l2] = concept_mean

    fig = plt.figure(figsize=(20, 15))
    # fig = plt.figure(figsize=(20, 10)
    
    min_ = 100
    max_ = -100
    for idx, model in enumerate(max_sim_dict):
        sim_dict = max_sim_dict[model]
        new_langs = get_new_langs(langs, model)
        for lang1 in new_langs:
            for lang2 in new_langs:
                sim = sim_dict[lang1][lang2]
                max_ = max(max_, sim)
                min_ = min(min_, sim)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(3, 3),
                    axes_pad=(0.1,0.6),
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.15,
                    share_all=False,
                    label_mode="all",
                    )
    pearson_corr_dict1 = {}
    pearson_corr_dict2 = {}
    for idx, model in enumerate(max_sim_dict):
        ax = grid[idx]
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        sim_dict = max_sim_dict[model]
        new_langs = get_new_langs(langs, model)
        data = []
        data_no_center = []
        same_d = []
        no_same_d = []
        for lang1 in new_langs:
            line = []
            for lang2 in new_langs:
                if lang1 != lang2:
                    data_no_center.append(sim)
                # diff = sim_dict[lang1][lang2] - sim_dict[lang2][lang2]
                sim = sim_dict[lang1][lang2]
                line.append(sim)
                if sim >= 0:
                    same_d.append(sim)
                else:
                    no_same_d.append(sim)
            data.append(line)
        data = np.array(data)
        data_no_center = np.array(data_no_center)

        for t in type_lst:
            if t not in pearson_corr_dict1:
                pearson_corr_dict1[t] = {}
            type_score = []
            type_score_no_center = []
            for lang1 in new_langs:
                line = []
                line_no_center = []
                for lang2 in new_langs:
                    score = lang_score[lang1][lang2][t]
                    line.append(1-score)
                    if lang1 != lang2:
                        line_no_center.append(1-score)
                type_score.append(line)
                type_score_no_center.append(line_no_center)
            type_score = np.array(type_score)
            type_score_no_center = np.array(type_score_no_center)
            correlation_matrix = np.corrcoef(data_no_center.flatten(), type_score_no_center.flatten())

            pearson_corr = correlation_matrix[0, 1]

            pearson_corr_dict1[t][model] = pearson_corr
        
        for t in type_lst:
            if t not in pearson_corr_dict2:
                pearson_corr_dict2[t] = {}
            type_score = []
            type_score_no_center = []
            not_cross, not_cross_data, high, high_data, low, low_data = [],[],[],[],[],[]
            for lang1 in new_langs:
                line = []
                line_no_center = []
                for lang2 in new_langs:
                    score = lang_score[lang1][lang2][t]
                    line.append(1-score)
                    if lang1 != lang2:
                        line_no_center.append(1-score)
                        if is_high(lang1, lang2, model) in ["all_high", "cross"]:
                            high.append(1-score)
                            high_data.append(sim_dict[lang1][lang2])
                        elif is_high(lang1, lang2, model) == "all_low":
                            low.append(1-score)
                            low_data.append(sim_dict[lang1][lang2])
                type_score.append(line)
                type_score_no_center.append(line_no_center)
            type_score = np.array(type_score)
            type_score_no_center = np.array(type_score_no_center)
            # direct
            # correlation_matrix = np.corrcoef(data_no_center.flatten(), type_score_no_center.flatten())
            # pearson_corr = correlation_matrix[0, 1]
            # pearson_corr_dict[t][model] = pearson_corr
            # avg
            high_correlation_matrix = np.corrcoef(high, high_data)
            low_correlation_matrix = np.corrcoef(low, low_data)
            pearson_corr_dict2[t][model] = (high_correlation_matrix[0, 1] + low_correlation_matrix[0, 1]) / 2


        normalized_data = np.zeros_like(data)
        ax.get_shared_y_axes().remove(ax)
        yticker = matplotlib.axis.Ticker()
        ax.yaxis.major = yticker

        yloc = matplotlib.ticker.AutoLocator()
        yfmt = matplotlib.ticker.ScalarFormatter()

        ax.yaxis.set_major_locator(yloc)
        ax.yaxis.set_major_formatter(yfmt)
        if idx in [0,3,6]:
            ax.set_yticks(np.arange(len(new_langs)), labels=new_langs)
            ax.set_yticklabels(new_langs, fontsize=16)
        else:
            ax.set_yticks([])
        ax.set_title(get_new_models(model), fontsize=18)
        # if i % 3 == 0:
        ax.get_shared_x_axes().remove(ax)
        xticker = matplotlib.axis.Ticker()
        ax.xaxis.major = xticker

        xloc = matplotlib.ticker.AutoLocator()
        xfmt = matplotlib.ticker.ScalarFormatter()

        ax.xaxis.set_major_locator(xloc)
        ax.xaxis.set_major_formatter(xfmt)

        ax.set_xticks(np.arange(len(new_langs))) # , labels=new_langs
        ax.set_xticklabels(new_langs, fontsize=16)
        for i in range(len(new_langs)):
            for j in range(len(new_langs)):
                text = ax.text(j, i, round(data[i, j],2), ha="center", va="center", color="w")
                item = data[i, j].item()
                normalized_data[i, j] = item
        im = ax.imshow(normalized_data, cmap="coolwarm", vmin=min_, vmax=max_)
    cbar = grid.cbar_axes[0].colorbar(im) # = cbar = ax.cax.colorbar(im)
    cbar.set_ticks([])
    cbar.ax.set_yticks(np.arange(min_, max_+0.01, (max_ - min_)))
    cbar.ax.set_yticklabels(['Dissimilar', 'Similar'], fontsize=18)
    save_path = os.path.join(save_dir, "all_model" + "-" +model+"-"+save_name)
    plt.savefig(save_path+".jpg")
    plt.savefig(save_path+".svg")
    print(save_path+".jpg")

    # model_lst = ['llama', 'qwen', 'bloomz']
    # for t, dic in pearson_corr_dict.items():
    #     print(t)
    #     # print("\t", dic)
    #     model_mean = []
    #     for model in model_lst:
    #         one_model_mean = []
    #         for k, v in dic.items():
    #             if model in k:
    #                 one_model_mean.append(v)
    #         one_model_mean = sum(one_model_mean) / len(one_model_mean)
    #         model_mean.append(one_model_mean)
    #         print(f"{model}: {round(one_model_mean,2)}")
    #     print(round(np.mean(list(dic.values())).item(),2))


    # lines = []
    # lines.append(r"\begin{tabular}{l|" + "cccc" * 3 + "}")
    # lines.append(r"\toprule")
    # lines.append(r"{} & \multicolumn{4}{c|}{\textbf{llama}} & \multicolumn{4}{c|}{\textbf{qwen}} & \multicolumn{4}{c|}{\textbf{bloomz}}" + r"\\")
    # lines.append(r"\hline")
    # lines.append(r"{} & {\textbf{7B}} & {\textbf{13B}} & {\textbf{70B}} & {\textbf{avg}} & {\textbf{1B8}} & {\textbf{7B}} & {\textbf{14B}} & {\textbf{avg}} & {\textbf{560M}} & {\textbf{1B7}} & {\textbf{7B1}} & {\textbf{avg}}" + r"\\")
    # lines.append(r"\hline")
    # for t, dic in pearson_corr_dict.items():
    #     strr = t
    #     for model in model_lst:
    #         one_model_mean = []
    #         for k, v in dic.items():
    #             if model in k:
    #                 one_model_mean.append(v)
    #                 strr += "& {" + str(round(v, 2)) + "}"
    #         one_model_mean = sum(one_model_mean) / len(one_model_mean)
    #         strr  += "& {" + str(round(one_model_mean, 2)) + "}"
    #     lines.append(strr + r"\\")
    #     lines.append(r"\hline")
    # lines.pop(-1)
    # lines.append(r"\bottomrule")
    # lines.append(r"\end{tabular}")

    # print("\n".join(lines))
    new_type_lst = ["genetic",  "syntactic", "geographic", "phonological"]
    lines = []
    lines.append(r"\begin{tabular}{l|" + "cc|" * 4 + "}")
    lines.append(r"\toprule")
    strr ="{} "
    for new_type in new_type_lst:
        strr += r"& \multicolumn{2}{c|}{" + new_type.capitalize() + "} "
    lines.append(strr + r"\\")
    # lines.append(r"{} & \multicolumn{2}{c|}{\textbf{Genetic}} & \multicolumn{2}{c|}{\textbf{Geographic}} & \multicolumn{2}{c|}{\textbf{Syntactic}} & \multicolumn{2}{c|}{\textbf{Phonological}}" + r"\\")
    lines.append(r"\hline")
    lines.append(r"{} & {Direct} & {Split} & {Direct} & {Split} & {Direct} & {Split} & {Direct} & {Split}" + r"\\")
    lines.append(r"\hline")

    for model in max_sim_dict.keys():
        strr = model
        for t in new_type_lst:
            strr += "& {" + str(round(pearson_corr_dict1[t][model], 2)) + "}"
            strr += "& {" + str(round(pearson_corr_dict2[t][model], 2)) + "}"
        lines.append(strr + r"\\")
        lines.append(r"\hline")
    lines.pop(-1)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))

def draw_sim_heatmap_7B(langs, model_concept_sim_dict, save_dir, save_name, per_model=False):
    max_sim_dict = {}
    for model, concept_sim_dict in model_concept_sim_dict.items():
        new_langs = get_new_langs(langs, model)
        max_sim_dict[model] = {}
        # max
        for l1 in new_langs:
            max_sim_dict[model][l1] = {}
            for l2 in new_langs:
                if l1 == l2:
                    concept_mean = 1
                else:
                    concept_sim_lst = []
                    for concept, sim_dict in concept_sim_dict.items():
                        # sim = sum(list(sim_dict[l1][l2].values())) / len(list(sim_dict[l1][l2].values()))
                        sim = max(list(sim_dict[l1][l2].values()))
                        concept_sim_lst.append(sim)
                    concept_mean = sum(concept_sim_lst) / len(concept_sim_lst)
                max_sim_dict[model][l1][l2] = concept_mean

    fig = plt.figure(figsize=(20, 6))
    # fig = plt.figure(figsize=(20, 10)
    
    min_ = 100
    max_ = -100
    for idx, model in enumerate(max_sim_dict):
        sim_dict = max_sim_dict[model]
        new_langs = get_new_langs(langs, model)
        for lang1 in new_langs:
            for lang2 in new_langs:
                sim = sim_dict[lang1][lang2]
                max_ = max(max_, sim)
                min_ = min(min_, sim)
    grid = AxesGrid(fig, 111,
                    nrows_ncols=(1, 3),
                    axes_pad=(0.4,0.6),
                    cbar_mode='single',
                    cbar_location='right',
                    cbar_pad=0.15,
                    share_all=False,
                    label_mode="all",
                    )
    pearson_corr_dict1 = {}
    pearson_corr_dict2 = {}
    for idx, model in enumerate(max_sim_dict):
        ax = grid[idx]
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        sim_dict = max_sim_dict[model]
        new_langs = get_new_langs(langs, model)
        data = []
        data_no_center = []
        same_d = []
        no_same_d = []
        for lang1 in new_langs:
            line = []
            for lang2 in new_langs:
                if lang1 != lang2:
                    data_no_center.append(sim)
                # diff = sim_dict[lang1][lang2] - sim_dict[lang2][lang2]
                sim = sim_dict[lang1][lang2]
                line.append(sim)
                if sim >= 0:
                    same_d.append(sim)
                else:
                    no_same_d.append(sim)
            data.append(line)
        data = np.array(data)
        data_no_center = np.array(data_no_center)

        for t in type_lst:
            if t not in pearson_corr_dict1:
                pearson_corr_dict1[t] = {}
            type_score = []
            type_score_no_center = []
            for lang1 in new_langs:
                line = []
                line_no_center = []
                for lang2 in new_langs:
                    score = lang_score[lang1][lang2][t]
                    line.append(1-score)
                    if lang1 != lang2:
                        line_no_center.append(1-score)
                type_score.append(line)
                type_score_no_center.append(line_no_center)
            type_score = np.array(type_score)
            type_score_no_center = np.array(type_score_no_center)
            correlation_matrix = np.corrcoef(data_no_center.flatten(), type_score_no_center.flatten())

            pearson_corr = correlation_matrix[0, 1]

            pearson_corr_dict1[t][model] = pearson_corr


        for t in type_lst:
            if t not in pearson_corr_dict2:
                pearson_corr_dict2[t] = {}
            type_score = []
            type_score_no_center = []
            not_cross, not_cross_data, high, high_data, low, low_data = [],[],[],[],[],[]
            for lang1 in new_langs:
                line = []
                line_no_center = []
                for lang2 in new_langs:
                    score = lang_score[lang1][lang2][t]
                    line.append(1-score)
                    if lang1 != lang2:
                        line_no_center.append(1-score)
                        if is_high(lang1, lang2, model) in ["all_high", "cross"]:
                            high.append(1-score)
                            high_data.append(sim_dict[lang1][lang2])
                        elif is_high(lang1, lang2, model) == "all_low":
                            low.append(1-score)
                            low_data.append(sim_dict[lang1][lang2])
                type_score.append(line)
                type_score_no_center.append(line_no_center)
            type_score = np.array(type_score)
            type_score_no_center = np.array(type_score_no_center)
            # direct
            # correlation_matrix = np.corrcoef(data_no_center.flatten(), type_score_no_center.flatten())
            # pearson_corr = correlation_matrix[0, 1]
            # pearson_corr_dict[t][model] = pearson_corr
            # avg
            high_correlation_matrix = np.corrcoef(high, high_data)
            low_correlation_matrix = np.corrcoef(low, low_data)
            pearson_corr_dict2[t][model] = (high_correlation_matrix[0, 1] + low_correlation_matrix[0, 1]) / 2
        
            # not_cross_correlation_matrix = np.corrcoef(not_cross, not_cross_data)
            # pearson_corr_dict[t][model] = not_cross_correlation_matrix[0, 1]

        normalized_data = np.zeros_like(data)
        ax.get_shared_y_axes().remove(ax)
        yticker = matplotlib.axis.Ticker()
        ax.yaxis.major = yticker

        yloc = matplotlib.ticker.AutoLocator()
        yfmt = matplotlib.ticker.ScalarFormatter()

        ax.yaxis.set_major_locator(yloc)
        ax.yaxis.set_major_formatter(yfmt)
        ax.set_yticks(np.arange(len(new_langs)), labels=new_langs)
        ax.set_yticklabels(new_langs, fontsize=16)
        ax.set_title(get_new_models(model), fontsize=23)
        # if i % 3 == 0:
        ax.get_shared_x_axes().remove(ax)
        xticker = matplotlib.axis.Ticker()
        ax.xaxis.major = xticker

        xloc = matplotlib.ticker.AutoLocator()
        xfmt = matplotlib.ticker.ScalarFormatter()

        ax.xaxis.set_major_locator(xloc)
        ax.xaxis.set_major_formatter(xfmt)

        ax.set_xticks(np.arange(len(new_langs))) # , labels=new_langs
        ax.set_xticklabels(new_langs, fontsize=16)
        for i in range(len(new_langs)):
            for j in range(len(new_langs)):
                text = ax.text(j, i, round(data[i, j],2), ha="center", va="center", color="w")
                item = data[i, j].item()
                normalized_data[i, j] = item
        im = ax.imshow(normalized_data, cmap="coolwarm", vmin=min_, vmax=max_)
        ax.tick_params(axis='both', which='major', labelsize=20)
    cbar = grid.cbar_axes[0].colorbar(im) # = cbar = ax.cax.colorbar(im)
    cbar.set_ticks([])
    cbar.ax.set_yticks(np.arange(min_, max_+0.01, (max_ - min_)))
    cbar.ax.set_yticklabels(['Dissimilar', 'Similar'], fontsize=23)
    save_path = os.path.join(save_dir, "all_model" + "-" +model+"-"+save_name)
    plt.savefig(save_path+".jpg")
    plt.savefig(save_path+".svg")
    print(save_path+".jpg")

    print("pearson_corr_dict1")
    model_lst = ['llama', 'qwen', 'bloomz']
    for t, dic in pearson_corr_dict1.items():
        print(t)
        # print("\t", dic)
        model_mean = []
        for model in model_lst:
            one_model_mean = []
            for k, v in dic.items():
                if model in k:
                    one_model_mean.append(v)
            one_model_mean = sum(one_model_mean) / len(one_model_mean)
            model_mean.append(one_model_mean)
            print(f"{model}: {round(one_model_mean,2)}")
        print(round(np.mean(list(dic.values())).item(),2))

    print("pearson_corr_dict2")
    for t, dic in pearson_corr_dict2.items():
        print(t)
        # print("\t", dic)
        model_mean = []
        for model in model_lst:
            one_model_mean = []
            for k, v in dic.items():
                if model in k:
                    one_model_mean.append(v)
            one_model_mean = sum(one_model_mean) / len(one_model_mean)
            model_mean.append(one_model_mean)
            print(f"{model}: {round(one_model_mean,2)}")
        print(round(np.mean(list(dic.values())).item(),2))
    print("latex")

    new_type_lst = ["genetic",  "syntactic", "geographic", "phonological"]
    lines = []
    lines.append(r"\begin{tabular}{l|" + "cc|" * 4 + "}")
    lines.append(r"\toprule")
    strr ="{} "
    for new_type in new_type_lst:
        strr += r"& \multicolumn{2}{c|}{" + new_type.capitalize() + "} "
    lines.append(strr + r"\\")
    # lines.append(r"{} & \multicolumn{2}{c|}{\textbf{Genetic}} & \multicolumn{2}{c|}{\textbf{Geographic}} & \multicolumn{2}{c|}{\textbf{Syntactic}} & \multicolumn{2}{c|}{\textbf{Phonological}}" + r"\\")
    lines.append(r"\hline")
    lines.append(r"{} & {Direct} & {Split} & {Direct} & {Split} & {Direct} & {Split} & {Direct} & {Split}" + r"\\")
    lines.append(r"\hline")

    for model in max_sim_dict.keys():
        strr = model
        for t in new_type_lst:
            strr += "& {" + str(round(pearson_corr_dict1[t][model], 2)) + "}"
            strr += "& {" + str(round(pearson_corr_dict2[t][model], 2)) + "}"
        lines.append(strr + r"\\")
        lines.append(r"\hline")
    lines.pop(-1)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    print("\n".join(lines))

def draw_res_all_model(langs, model_concept_res_dict, save_name):
    # fig, ax = plt.subplots(figsize=(8, 8))
    fig, ax = plt.subplots()

    mean_res_dict = {}
    for model, concept_res_dict in model_concept_res_dict.items():
        mean_res_dict[model] = {}

        concept_res_lst = []
        for concept, res_dict in concept_res_dict.items():
            langs_res_lst = []
            new_langs = get_new_langs(langs, model)
            for lang in new_langs:
                lang_res = list(res_dict[lang][lang].values())
                langs_res_lst.append(lang_res)
            langs_res_lst = np.array(langs_res_lst)
            langs_res_lst = np.mean(langs_res_lst, axis=0)
            concept_res_lst.append(langs_res_lst)
        concept_res_lst = np.vstack(concept_res_lst)
        concept_res_lst = np.mean(concept_res_lst, axis=0)
        mean_res_dict[model] = concept_res_lst
    
    idx = 0
    for model, res in mean_res_dict.items():

        x = np.arange(len(res)) / len(res)

        # 绘制折线图
        # plt.plot(x, res, marker='o', markersize='3',label=model, color=draw_colors_models[idx])

        if "7B" in model:
            plt.plot(x, res, markersize='3',label=get_new_models(model), color=draw_colors_models[idx])
        else:
            plt.plot(x, res, markersize='3',label=get_new_models(model), color=draw_colors_models[idx], linestyle='--')

        idx += 1
    # plt.yticks(np.arange(-1, 1.1, 0.2))
    # plt.ylim([-1, 1])
    size = 15
    # plt.grid(axis="x")
    plt.grid()
    plt.legend(loc='lower center', prop = {'size':size-2.5}, title = "Model", title_fontsize=size-2.5, bbox_to_anchor=(0.6, 0.0))
    plt.xlabel('Layer (relative to network depth)', fontsize=size+2)
    plt.ylabel('Accuracy', fontsize=size+2)
    def percentage_formatter(x, pos):
        return f'{x*100:.0f}%'
    ax.yaxis.set_major_formatter(FuncFormatter(percentage_formatter))
    # plt.title('Performance of Conceptual Direction Classification', fontsize=size+2)
    plt.tick_params(labelsize=size+2)
    current_size = fig.get_size_inches()
    new_size = (current_size[0], current_size[1] + 0.5)
    fig.set_size_inches(new_size)
    print(save_name+".jpg")
    plt.savefig(save_name+".jpg")
    plt.savefig(save_name+".svg")
    # plt.tight_layout(pad=1)
    plt.cla()

def convert_latex_table_res_trans(langs, model_concept_res_dict):
    max_res_dict = {}
    trans_res_dict = {}
    for model, concept_res_dict in model_concept_res_dict.items():
        new_langs = get_new_langs(langs, model)
        max_res_dict[model] = {}
        trans_res_dict[model] = {}
        # max
        for l1 in new_langs:
            max_res_dict[model][l1] = {}
            for l2 in new_langs:
                concept_res_lst = []
                for concept, res_dict in concept_res_dict.items():
                    res = max(list(res_dict[l1][l2].values()))
                    concept_res_lst.append(res)
                concept_mean = sum(concept_res_lst) / len(concept_res_lst)
                max_res_dict[model][l1][l2] = concept_mean
    
        # res_trans
        for l2 in new_langs:
            self_acc = max_res_dict[model][l2][l2]
            trans_acc_lst = []
            for l1 in new_langs:
                if l1 == l2:
                    continue
                trans_acc = max_res_dict[model][l1][l2]
                trans_acc_lst.append(trans_acc - self_acc)
            trans_res_dict[model][l2] = trans_acc_lst
    
    lines = []
    for model, res_dict in trans_res_dict.items():
        keys = list(res_dict.keys()) + ["avg"]
        if model in ["llama2-chat-7B", "bloomz-560M", "qwen-chat-1B8"]:
            if model == "qwen-chat-1B8":
                lines.append(r"\begin{tabular}{l" + "c" + "c|" + "c" * (len(res_dict.keys())-2) + "|c}") # no @{}
            else:
                lines.append(r"\begin{tabular}{l" + "c|" + "c" * (len(res_dict.keys())-1) + "|c}") # no @{}
            lines.append(r"\toprule")
            strr = r"{}"
            # keys.pop(0)
            # if "qwen" in model:
            #     zh_idx = keys.index("zh")
            #     en_idx = keys.index("en")
            #     if zh_idx > en_idx:
            #         keys.pop(zh_idx)
            #         keys.pop(en_idx)
            #     else:
            #         keys.pop(en_idx)
            #         keys.pop(zh_idx)
            #     keys = ["zh", "en"] + keys
            for key in keys:
                strr += r" & {\textbf{"
                strr += key
                strr += "}}"
            lines.append(strr + r"\\")
            lines.append(r"\hline")
        res_lst = []
        for lang, res in res_dict.items():
            # data = np.array(res)
            # mean_value = np.mean(data).item()
            # std_deviation = np.std(data).item()
            
            # mean_value = round(mean_value*100, 2)
            # std_deviation = round(std_deviation*100, 2)
            trans_rate = 0
            for num in res:
                if num >= 0:
                    trans_rate += 1
            trans_rate = int(trans_rate/(len(res))*100)
            res_lst.append(trans_rate)
            # res_lst.append(f"{mean_value}({std_deviation})")
            # mean_lst.append(mean_value)
            # std_lst.append(std_deviation)
        # avg = round(sum(mean_lst) / len(mean_lst),2)
        # avg_std = round(sum(std_lst) / len(std_lst),2)
        # res_lst.append(avg, avg_std) # f"{avg}({avg_std})")
        # res_lst.pop(0)
        # if "qwen" in model:
        #     en_res = res_lst[en_idx]
        #     zh_res = res_lst[zh_idx]
        #     if zh_idx > en_idx:
        #         res_lst.pop(zh_idx)
        #         res_lst.pop(en_idx)
        #     else:
        #         res_lst.pop(en_idx)
        #         res_lst.pop(zh_idx)
        #     # res_lst.pop(en_idx)
        #     # res_lst.pop(zh_idx)
        #     res_lst = [zh_res, en_res] + res_lst
        #     res_lst.append(int(sum(res_lst[2:])/len(res_lst[2:])))
        # else:
        #     res_lst.append(int(sum(res_lst[1:])/len(res_lst[1:])))
        res_lst.append(int(sum(res_lst)/len(res_lst)))
        strr = model
        for res in res_lst:
            strr += f"& {res}"
        lines.append(strr + r"\\")
        if model in ["llama2-chat-70B", "bloomz-7B1", "qwen-chat-14B"]:
            lines.append(r"\bottomrule")
            lines.append(r"\end{tabular}")
            print("\n".join(lines))
            lines = []

    
    # llama
        
def draw_sim(layers, langs, sims, save_name):
    means = np.mean(sims, axis=1)
    lower_bounds = np.min(sims, axis=1)
    upper_bounds = np.max(sims, axis=1)
    variances = np.var(sims, axis=1)

    # 生成 x 轴坐标
    x = np.arange(len(means))

    # 绘制折线图
    plt.plot(x, means, label='Mean')
    # for idx, lang in enumerate(langs):
        # lang_mean = np.array(lang_means[lang])
        # plt.plot(x, lang_mean, label=lang, color=color_lst[idx])
    plt.fill_between(x, lower_bounds, upper_bounds, alpha=0.2, label='Bounds')
    # plt.fill_between(x, means - variances, means + variances, alpha=0.2, label='Variance Range')
    # 设置图例和标签
    plt.legend()
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Direction Similarity')

    print(save_name+".jpg")
    plt.savefig(save_name+".jpg")
    plt.savefig(save_name+".svg")
    plt.cla()

def draw_sim_all(layers, langs, sims_dict, save_name):
    idx = 0
    max_sim_dict = {}
    concept_mean = []
    for concept, sims in sims_dict.items():
        means = np.mean(sims, axis=1) # lang
        # std_deviation = np.std(sims, axis=1)
        lower_bounds = np.min(sims, axis=1)
        upper_bounds = np.max(sims, axis=1)
        # 生成 x 轴坐标
        x = np.arange(len(means))
        concept_mean.append(means)
        max_sim = max(means).item() # layer
        max_sim_dict[concept] = round(max_sim, 2)
        # 绘制折线图
        plt.plot(x, means, label=concept, color=draw_colors[idx])
        # for idx, lang in enumerate(langs):
            # lang_mean = np.array(lang_means[lang])
            # plt.plot(x, lang_mean, label=lang, color=color_lst[idx])
        plt.fill_between(x, lower_bounds, upper_bounds, alpha=0.2, color=draw_colors[idx])
        # plt.fill_between(x, means - std_deviation, means + std_deviation, alpha=0.2, label='Variance Range')
        idx += 1
    max_sim_dict["avg"] = round(sum(list(max_sim_dict.values())) / len(max_sim_dict.values()),2)
    for k,v in max_sim_dict.items():
        max_sim_dict[k] = str(v)
    plt.yticks(np.arange(-1, 1.1, 0.2))
    # plt.ylim([-1, 1])
    plt.legend(loc='lower right')
    plt.xlabel('Layer')
    plt.ylabel('Cosine Similarity')
    plt.title('Direction Similarity')

    print(save_name+".jpg")
    plt.savefig(save_name+".jpg")
    plt.savefig(save_name+".svg")
    plt.cla()
    print("\t".join(list(max_sim_dict.keys())))
    print("\t".join(list(max_sim_dict.values())))

def draw_sim_all_model(langs, model_sim_dict, save_name):
    # fig, ax = plt.subplots(figsize=(8, 8))
    fig, ax = plt.subplots()
    mean_sim_dict = {}
    for model, sim_dict in model_sim_dict.items():
        means_all_concept = []
        for concept, sims in sim_dict.items(): # sim [[132] * 32]
            means = np.mean(sims, axis=1) # lang 1 * 32
            means_all_concept.append(means) # concept num * 32
        means_all_concept = np.vstack(means_all_concept) # 1 * 32
        means_all_concept = np.mean(means_all_concept, axis=0) # concept
        mean_sim_dict[model] = means_all_concept # [1 * 32]
    idx = 0
    for model, sims in mean_sim_dict.items():

        x = np.arange(len(sims)) / len(sims)

        # 绘制折线图
        if "7B" in model:
            plt.plot(x, sims, markersize='3',label=get_new_models(model), color=draw_colors_models[idx])
        else:
            plt.plot(x, sims, markersize='3',label=get_new_models(model), color=draw_colors_models[idx], linestyle='--')
        idx += 1
    # plt.yticks(np.arange(-1, 1.1, 0.2))
    # plt.ylim([-1, 1])
    size = 15
    # plt.grid(axis="x")
    plt.grid()
    plt.legend(loc='lower center',  bbox_to_anchor=(0.57, 0.0), prop = {'size':size-2.5}, title = "Model", title_fontsize=size-2.5)
    plt.xlabel('Layer (relative to network depth)', fontsize=size+2)
    plt.ylabel('Cosine Similarity', fontsize=size+2)
    # plt.title('Cross-Lingual Similarity of Concept Vectors', fontsize=size+2)
    plt.tick_params(labelsize=size+2)
    current_size = fig.get_size_inches()
    new_size = (current_size[0], current_size[1] + 0.5)
    fig.set_size_inches(new_size)
    print(save_name+".jpg")
    plt.savefig(save_name+".jpg")
    plt.savefig(save_name+".svg")
    # plt.tight_layout(pad=1)
    plt.cla()

def compute_sim(langs, reader_dict, concept):
    sim_dict = {}
    sim_dict2 = {}
    for i, layer_idx in enumerate(list(reader_dict["en"].directions.keys())):
        sim_dict[layer_idx] = {}
        for l1 in langs:
            if l1 not in sim_dict2:
                sim_dict2[l1] = {}
            sim_dict[layer_idx][l1] = {}
            for l2 in langs:
                if l2 not in sim_dict2[l1]:
                    sim_dict2[l1][l2] = {}
                if l1 == l2:
                    continue
                v1 = reader_dict[l1].directions[layer_idx].squeeze() * reader_dict[l1].direction_signs[layer_idx][0]
                v2 = reader_dict[l2].directions[layer_idx].squeeze() * reader_dict[l2].direction_signs[layer_idx][0]
                cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                sim_dict[layer_idx][l1][l2] = cosine_similarity
                sim_dict2[l1][l2][layer_idx] = cosine_similarity
    return sim_dict, sim_dict2

def compute_acc(langs, hidden_layers, reader_dict, data_dict, concept, to_classify_langs=None):
    res_dict = {}
    if to_classify_langs == None:
        to_classify_langs = langs

    all_num = len(langs) * len(to_classify_langs)
    # precompute hidden state for test data
    # hidden_state_dict = {}
    # for l in langs:
    #     hidden_state = read_pipeline._batched_string_to_hiddens(
    #         data_dict[l]['test']['data'], 
    #         rep_token=-1,
    #         hidden_layers=hidden_layers, 
    #         batch_size=32,
    #         which_hidden_states=None)
    #     hidden_state_dict[l] = hidden_state
    # hidden_layers = [29]
    with tqdm.tqdm(total=all_num) as pbar:

        pbar.set_description('Cross lingual classification:')
        for l2 in to_classify_langs: # 被分类
            l2_hidden_state = read_pipeline._batched_string_to_hiddens(
                data_dict[l2]['test']['data'], 
                rep_token=-1,
                hidden_layers=hidden_layers, 
                batch_size=4,
                which_hidden_states=None)
            for l1 in langs: # 分类
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
                # max_acc, max_layer = -1, None
                for layer in hidden_layers:
                    H_test = [H[layer] for H in H_tests] 
                    H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]

                    sign = rep_reader.direction_signs[layer]
                    eval_func = min if sign == -1 else max
                    
                    cors = np.mean([eval_func(H) == H[0] for H in H_test])
                    # if layer == 29:
                    #     for idx, score in enumerate(H_test):
                    #         pair = test_data['data'][idx*2: idx*2+2]
                    #         print("-"*10)
                    #         print(score[0], pair[0])
                    #         print(score[1], pair[1])
                    results[layer] = cors
                    # if cors > max_acc:
                    #     max_acc = cors
                    #     max_layer = layer
                # print max layer
                res_dict[l1][l2] = results
                # layer = max_layer
                # H_test = [H[layer] for H in H_tests] 
                # H_test = [H_test[i:i+2] for i in range(0, len(H_test), 2)]
                # sign = rep_reader.direction_signs[layer]
                # eval_func = min if sign == -1 else max
                
                # cors = np.mean([eval_func(H) == H[0] for H in H_test])
                # for idx, score in enumerate(H_test):
                #     pair = test_data['data'][idx*2: idx*2+2]
                #     print("-"*10)
                #     print(score[0], pair[0])
                #     print(score[1], pair[1])
                pbar.update(1)
    return res_dict

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
    # strr = r"\textbf{" + concept.capitalize() + "}"
    # strr = r"\textbf{Model} & \textbf{Size}"
    strr = r"\multicolumn{2}{l|}{\textbf{" + concept.capitalize() +"}}"
    for lang in langs:
        strr += r" & {\textbf{"
        strr += lang
        strr += "}}"
    strr += r"& {\textbf{Avg}}"
    lines.append(strr + r"\\")
    # lines.append("{} & {} & \multicolumn{17}{c}{\emph{" + concept +"}}" + r"\\")
    # lines.append(r"\cline{1-1}")
    # lines.append(r"\textbf{Models}" + "& {}" * (len(langs)+1) + r"\\")
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
    direction_method = 'cluster_mean'
    if not cross_model:
        model, tokenizer, template = None, None, None
        if not args.not_load:
            model, tokenizer, template = load_model_tokenizer(model_name, model_size)
            read_pipeline = pipeline("rep-reading", model=model, tokenizer=tokenizer)
            template = refine_template(template)
            hidden_layers = list(range(1, model.config.num_hidden_layers + 1))
        else:
            hidden_layers = get_hidden_layers(f"{model_name}-{model_size}")
        # read params
        # hidden_layers = list(range(-1, -model.config.num_hidden_layers, -1)) # [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31]
        # hidden_layers = list(range(1, model.config.num_hidden_layers + 1)) # 1&+1: embedding

        concept_sim_dict = {}
        concept_sim_dict2 = {}
        if split == 0.8:
            save_dir = os.path.join("res", f"{model_name}-{model_size}")
        else:
            save_dir = os.path.join("res", f"{model_name}-{model_size}-{split}")
        os.makedirs(save_dir, exist_ok=True)
        for concept in concepts:
            concept_save_dir = os.path.join(save_dir, concept)
            # concept_reader_dir = os.path.join("res", concept)
            os.makedirs(concept_save_dir, exist_ok=True)
            """
            model
            concept
            """
            if random_direction:
                concept_reader_path = os.path.join(concept_save_dir, "random_direction_reader_dict.pk")
                concept_acc_path = os.path.join(concept_save_dir, "random_direction_acc.pk")
            else:
                concept_reader_path = os.path.join(concept_save_dir, "reader_dict.pk")
                concept_acc_path = os.path.join(concept_save_dir, "acc.pk")
            t1 = time.time()
            data_dict = {}
            # read data
            if not args.not_load:
                pbar = tqdm.tqdm(langs)
                for lang in pbar: # time passed: 721.3666143417358
                    pbar.set_description("read data")
                    data = get_data(lang, concept, template, split, random_direction)
                    data_dict[lang] = data
            # create reader
            if os.path.exists(concept_reader_path):
                reader_dict = torch.load(concept_reader_path)
            else:
                reader_dict = {}
                pbar = tqdm.tqdm(langs)
                for lang in pbar:
                    pbar.set_description("create reader")
                    rep_reader = get_rep_reader(train_data=data_dict[lang]['train'],
                                                read_pipeline=read_pipeline,
                                                hidden_layers=hidden_layers,
                                                direction_method=direction_method,
                                                n_difference=n_difference)
                    reader_dict[lang] = rep_reader
                torch.save(reader_dict, concept_reader_path)
            print("time passed:", time.time() - t1) # 9372.15235710144
            # 1. sim
            sim_dict, sim_dict2 = compute_sim(langs, reader_dict, concept)
            if args.draw:
                all_sim, lang_means = analysis_sim(hidden_layers, langs, sim_dict, model_name)
                concept_sim_dict[concept] = all_sim
                concept_sim_dict2[concept] = sim_dict2
                draw_save_name = os.path.join(concept_save_dir, "cosine_sim")
                draw_sim(hidden_layers, langs, all_sim, draw_save_name)
            # 2. cross classify
            if os.path.exists(concept_acc_path):
                res_dict = torch.load(concept_acc_path)
                extra_langs = []
                for lang in langs:
                    if lang not in res_dict:
                        extra_langs.append(lang)
                if len(extra_langs) != 0:
                    extra_res_dict1 = compute_acc(langs, hidden_layers, reader_dict, data_dict, concept, to_classify_langs=extra_langs)
                    extra_res_dict2 = compute_acc(extra_langs, hidden_layers, reader_dict, data_dict, concept, to_classify_langs=langs)
                    extra_res_lst = [extra_res_dict1, extra_res_dict2]
                    # combine
                    for extra_res in extra_res_lst:
                        for l1 in extra_res:
                            if l1 not in res_dict:
                                res_dict[l1] = {}
                            for l2 in extra_res[l1]:
                                res_dict[l1][l2] = extra_res[l1][l2]
                    old_concept_acc_path = os.path.join(concept_save_dir, "old_acc.pk")
                    # move to old
                    os.rename(concept_acc_path, old_concept_acc_path)
                    # save new
                    torch.save(res_dict, concept_acc_path)
                    # check
                    res_dict = torch.load(concept_acc_path)
                    extra_langs = []
                    for lang in langs:
                        if lang not in res_dict:
                            extra_langs.append(lang)
                    assert len(extra_langs) == 0
            else:
                res_dict = compute_acc(langs, hidden_layers, reader_dict, data_dict, concept)
                torch.save(res_dict, concept_acc_path)
            if len(langs) == 1:
                print(max(list(res_dict[langs[0]][langs[0]].values())))
            if args.draw:
                draw_save_name = os.path.join(concept_save_dir, "cross_acc")
                draw_acc(hidden_layers, langs, res_dict, draw_save_name)
                draw_save_name2 = os.path.join(concept_save_dir, "cross_acc2")
                draw_acc2(hidden_layers, langs, res_dict, draw_save_name2)
        if args.draw:
            draw_save_name = os.path.join(save_dir, "cosine_sim"+ "_" + model_size)
            draw_sim_all(hidden_layers, langs, concept_sim_dict, draw_save_name)
    else:
        model_name_lst = cross_model.split(",")
        concept_res_dict = {}
        model_sim_dict = {}
        model_concept_res_dict = {}
        model_concept_sim_dict = {}
        for concept in concepts:
            model_res_dict = {}
            for model_name in model_name_lst:
                if model_name not in model_sim_dict:
                    model_sim_dict[model_name] = {}
                if model_name not in model_concept_res_dict:
                    model_concept_res_dict[model_name] = {}
                if model_name not in model_concept_sim_dict:
                    model_concept_sim_dict[model_name] = {}
                hidden_layers = get_hidden_layers(model_name)
                if split == 0.8:
                    save_dir = os.path.join("res", model_name)
                else:
                    save_dir = os.path.join("res", f"{model_name}-{split}")
                concept_save_dir = os.path.join(save_dir, concept)
                concept_reader_path = os.path.join(concept_save_dir, "reader_dict.pk")
                concept_acc_path = os.path.join(concept_save_dir, "acc.pk")
                reader_dict = torch.load(concept_reader_path)
                res_dict = torch.load(concept_acc_path)
                sim_dict, sim_dict2 = compute_sim(langs, reader_dict, concept)
                all_sim, lang_means = analysis_sim(hidden_layers, langs, sim_dict, model_name)
                model_sim_dict[model_name][concept] = all_sim
                model_concept_res_dict[model_name][concept] = res_dict
                model_concept_sim_dict[model_name][concept] = sim_dict2
                model_res_dict[model_name] = res_dict
            concept_res_dict[concept] = model_res_dict
            key = "|".join(model_name_lst)
            if split == 0.8:
                save_dir = os.path.join("res", key)
            else:
                save_dir = os.path.join("res", f"{key}-{split}")
            os.makedirs(save_dir, exist_ok=True)
            if args.draw:
                draw_save_name = os.path.join(save_dir, f"{concept}-cross-acc")
                draw_acc2_all(hidden_layers, langs, model_res_dict, draw_save_name, concept=concept)
            # convert_latex_table(langs, model_res_dict, concept)
        if args.draw:
            # draw_save_name = os.path.join(save_dir, f"all-concept-cross-acc")
            # draw_acc2_all(hidden_layers, langs, concept_res_dict, draw_save_name, all_concept=True)


            # draw_save_name = os.path.join(save_dir, f"all-model-sim")
            # draw_sim_all_model(langs, model_sim_dict, draw_save_name)


            # draw_save_name = os.path.join(save_dir, f"all-model-res")
            # draw_res_all_model(langs, model_concept_res_dict, draw_save_name)

            # convert_latex_table_res_trans(langs, model_concept_res_dict)

            if len(model_concept_res_dict) == 3:
                draw_acc_heatmap_7B(langs, model_concept_res_dict, save_dir,  f"cross-acc", per_model=True)
            else:
                draw_acc_heatmap(langs, model_concept_res_dict, save_dir,  f"cross-acc", per_model=True)

            if len(model_concept_sim_dict) == 3:
                draw_sim_heatmap_7B(langs, model_concept_sim_dict, save_dir,  f"cross-sim", per_model=True)
            else:   
                draw_sim_heatmap(langs, model_concept_sim_dict, save_dir,  f"cross-sim", per_model=True)