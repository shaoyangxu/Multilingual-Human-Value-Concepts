import argparse
from utils import draw_colors_models, get_hidden_layers, get_new_langs, is_high, get_new_models, shared_langs
import os
from repe import repe_pipeline_registry
import matplotlib
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid


lang_score = torch.load("lang_sim.pt")
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

def draw_sim_heatmap(langs, model_concept_sim_dict, save_dir, save_name):
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
                        sim = max(list(sim_dict[l1][l2].values()))
                        concept_sim_lst.append(sim)
                    concept_mean = sum(concept_sim_lst) / len(concept_sim_lst)
                max_sim_dict[model][l1][l2] = concept_mean

    fig = plt.figure(figsize=(20, 15))
    
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
                sim = sim_dict[lang1][lang2]
                line.append(sim)
                if sim >= 0:
                    same_d.append(sim)
                else:
                    no_same_d.append(sim)
            data.append(line)
        data = np.array(data)
        data_no_center = np.array(data_no_center)

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
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path+".jpg")
    plt.savefig(save_path+".svg")
    print(save_path+".jpg")

def draw_sim_heatmap_7B(langs, model_concept_sim_dict, save_dir, save_name):
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
                        sim = max(list(sim_dict[l1][l2].values()))
                        concept_sim_lst.append(sim)
                    concept_mean = sum(concept_sim_lst) / len(concept_sim_lst)
                max_sim_dict[model][l1][l2] = concept_mean

    fig = plt.figure(figsize=(20, 6))
    
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

def linguistic_relationships(langs, model_concept_sim_dict):
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
                        sim = max(list(sim_dict[l1][l2].values()))
                        concept_sim_lst.append(sim)
                    concept_mean = sum(concept_sim_lst) / len(concept_sim_lst)
                max_sim_dict[model][l1][l2] = concept_mean

    
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
    pearson_corr_dict1 = {}
    pearson_corr_dict2 = {}
    for idx, model in enumerate(max_sim_dict):
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
            high, high_data, low, low_data = [],[],[],[]
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
            high_correlation_matrix = np.corrcoef(high, high_data)
            low_correlation_matrix = np.corrcoef(low, low_data)
            pearson_corr_dict2[t][model] = (high_correlation_matrix[0, 1] + low_correlation_matrix[0, 1]) / 2


    new_type_lst = ["genetic", "syntactic", "geographic", "phonological"]
    lines = []
    lines.append(r"\begin{tabular}{l|" + "cc|" * 4 + "}")
    lines.append(r"\toprule")
    strr ="{} "
    for new_type in new_type_lst:
        strr += r"& \multicolumn{2}{c|}{" + new_type.capitalize() + "} "
    lines.append(strr + r"\\")
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

def draw_sim_layers(model_sim_dict, save_name):
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
    size = 15
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

def compute_sim(langs, layers, reader_dict):
    sim_dict = {}
    sim_dict2 = {} # 
    for i, layer_idx in enumerate(list(reader_dict["en"].directions.keys())):
        sim_dict2[layer_idx] = {} # 
        for l1 in langs:
            if l1 not in sim_dict:
                sim_dict[l1] = {}
            sim_dict2[layer_idx][l1] = {} # 
            for l2 in langs:
                if l2 not in sim_dict[l1]:
                    sim_dict[l1][l2] = {}
                if l1 == l2:
                    continue
                v1 = reader_dict[l1].directions[layer_idx].squeeze() * reader_dict[l1].direction_signs[layer_idx][0]
                v2 = reader_dict[l2].directions[layer_idx].squeeze() * reader_dict[l2].direction_signs[layer_idx][0]
                cosine_similarity = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                sim_dict[l1][l2][layer_idx] = cosine_similarity
                sim_dict2[layer_idx][l1][l2] = cosine_similarity # 
    

    shared_langs_sim = [] # [layer1, layer2, ..]
    for i, layer_idx in enumerate(layers):
        layer_sim = [] # len(langs) ^ 2 
        for l1 in shared_langs:
            lang_sim = [] # 1 target lang with all source langs
            for l2 in shared_langs:
                if l1 == l2:
                    continue
                cosine_similarity = sim_dict2[layer_idx][l1][l2] #
                lang_sim.append(cosine_similarity)
            layer_sim.extend(lang_sim)
        shared_langs_sim.append(layer_sim)
    
    return sim_dict, shared_langs_sim

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
    all_sim_dict = {}
    shared_lang_sim_dict = {}
    for concept in concepts:
        for model_name in model_name_lst:
            if model_name not in shared_lang_sim_dict:
                shared_lang_sim_dict[model_name] = {}
            if model_name not in all_sim_dict:
                all_sim_dict[model_name] = {}
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
            sim_dict, shared_langs_sim = compute_sim(langs, hidden_layers, reader_dict)
            all_sim_dict[model_name][concept] = sim_dict
            shared_lang_sim_dict[model_name][concept] = shared_langs_sim
        key = "|".join(model_name_lst)
        if split == 0.8:
            save_dir = os.path.join("res", key)
        else:
            save_dir = os.path.join("res", f"{key}-{split}")
        os.makedirs(save_dir, exist_ok=True)

    draw_save_name = os.path.join(save_dir, f"cross_sim_across_layers")
    draw_sim_layers(shared_lang_sim_dict, draw_save_name)

    if len(all_sim_dict) == 3:
        draw_sim_heatmap_7B(langs, all_sim_dict, save_dir,  f"cross_sim_7b")
    else:   
        draw_sim_heatmap(langs, all_sim_dict, save_dir,  f"cross_sim_all")
        linguistic_relationships(langs, all_sim_dict) # the results are slightly different from the results in the paper