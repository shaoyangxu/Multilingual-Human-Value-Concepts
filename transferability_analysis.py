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
import math

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
        "--draw",
        action="store_true"
    )
    parser.add_argument(
        "--cross-model",
        type=str,
        default="",
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

def draw_acc_heatmap(langs, model_concept_res_dict, save_dir, save_name):

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

    fig = plt.figure(figsize=(20, 15))

    grid = AxesGrid(fig, 111,
                    nrows_ncols=(3, 3),
                    axes_pad=(0.1,0.6),
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
    fig.text(0.5, 0.07, 'Target Languages', ha='center', va='center', fontsize=20)
    fig.text(0.2, 0.5, 'Source Languages', ha='center', va='center', rotation='vertical', fontsize=20)
    cbar = grid.cbar_axes[0].colorbar(im) # = cbar = ax.cax.colorbar(im)
    cbar.set_ticks([])
    cbar.ax.set_yticks(np.arange(-1, 1.1, 1))
    cbar.ax.set_yticklabels(['Negative', 'Medium', 'Positive'], fontsize=18)
    save_path = os.path.join(save_dir, "all_model" + "-" +model+"-"+save_name)
    plt.savefig(save_path+".jpg")
    plt.savefig(save_path+".svg")
    print(save_path+".jpg")


def draw_acc_heatmap_7B(langs, model_concept_res_dict, save_dir, save_name):

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
    model_name_lst = cross_model.split(",")
    model_concept_res_dict = {}
    for concept in concepts:
        for model_name in model_name_lst:
            if model_name not in model_concept_res_dict:
                model_concept_res_dict[model_name] = {}
            hidden_layers = get_hidden_layers(model_name)
            if split == 0.8:
                save_dir = os.path.join("res", model_name)
            else:
                save_dir = os.path.join("res", f"{model_name}-{split}")

            concept_save_dir = os.path.join(save_dir, concept)
            concept_acc_path = os.path.join(concept_save_dir, "acc.pk")

            res_dict = torch.load(concept_acc_path)
            model_concept_res_dict[model_name][concept] = res_dict
        key = "|".join(model_name_lst)
        if split == 0.8:
            save_dir = os.path.join("res", key)
        else:
            save_dir = os.path.join("res", f"{key}-{split}")
        os.makedirs(save_dir, exist_ok=True)

    if len(model_concept_res_dict) == 3:
        draw_acc_heatmap_7B(langs, model_concept_res_dict, save_dir,  f"cross_acc_7b")
    else:
        draw_acc_heatmap(langs, model_concept_res_dict, save_dir,  f"cross_acc_all")