from experiments.summarize import main as summarize_run

import json
import os
from pprint import pprint
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple
from matplotlib.ticker import MaxNLocator

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def error(dic):
	dic["post_score"] = list(dic["post_score"])
	dic["post_score"][1]=dic["post_score"][0]**2/3*(dic["post_neighborhood_success"][1]/dic["post_neighborhood_success"][0]**2+
	dic["post_paraphrase_success"][1]/dic["post_paraphrase_success"][0]**2+ 
	dic["post_rewrite_success"][1]/dic["post_rewrite_success"][0]**2)
	return dic

i = 0
pairs = []

for alpha in [1.,5., 10., 12.0, 15.]:
    for top_num_heads in [1,2,5,10]:
        pairs.append((alpha, top_num_heads))


summaries = []

print("WARNING! You need to make the experiments first")
for run_id in [f"run_0{i}" for i in range(0,20)]:
    path_at = Path("./results")/"MEMIT_ITI"/run_id
    summaries.append(summarize_run(path_at,["ITI"], apply_std_norm=True, number_norm=1000,abs_path=True)[0])

pprint(summaries)


summaries = [error(summary) for summary in summaries]

path_mce = Path("./results")/"MEMIT_all"
path_mec = Path("./results")/"MEMIT_all"
path_mcc = Path("./results")/"MEMIT_all"
path_mee = Path("./results")/"MEMIT_all"


baseline_ce = error(summarize_run(path_mce, ["run_002"],apply_std_norm=True, number_norm=7000, abs_path=True)[0])
baseline_ec = error(summarize_run(path_mec, ["run_003"],apply_std_norm=True, number_norm=7000, abs_path=True)[0])

baseline_cc = error(summarize_run(path_mcc, ["run_000"],apply_std_norm=True, number_norm=7000, abs_path=True)[0])
baseline_ee = error(summarize_run(path_mee, ["run_001"],apply_std_norm=True, number_norm=7000, abs_path=True)[0])

rewrite_success = {}

keys = list(summaries[0].keys())
rewrite_success = {key:[] for key in keys}
rewrite_success["pairs"]=pairs

for el in summaries:
    for key in keys:
        rewrite_success[key].append(el[key])

print(rewrite_success)

mapping = {"post_neighborhood_acc":"Neighbourhood Accuracy (NA)",
		   'post_neighborhood_diff': "Neighbourhood Magnitude (NM)",
		   "post_neighborhood_success": "Neighborhood Success (NS)",
		   "post_paraphrase_acc":"Paraphrase Accuracy (PA)",
		   'post_paraphrase_diff': "Paraphrase Magnitude (PM)",
		   "post_paraphrase_success": "Paraphrase Success (PS)",
		   "post_rewrite_acc":"Efficacy Accuracy (EA)",
		   'post_rewrite_diff': "Efficacy Magnitude (EM)",
		   "post_rewrite_success": "Efficacy Success (ES)",
		   "time":"time",
		   "post_score":"Score",
		   }

mapping_2 = {"Neighbourhood Accuracy (NA)":(3,2),
	"Neighbourhood Magnitude (NM)":(3,3),
	"Neighborhood Success (NS)":(3,1),
	"Paraphrase Accuracy (PA)":(2,2),
	"Paraphrase Magnitude (PM)":(2,3),
	"Paraphrase Success (PS)":(2,1),
	"Efficacy Accuracy (EA)":(1,2),
	"Efficacy Magnitude (EM)":(1,3),
	"Efficacy Success (ES)":(1,1)}

colors = ['red', 'green', 'blue', 'purple']
legend_labels = ['num_heads = 1', 'num_heads = 2', 'num_heads = 5', 'num_heads = 10']

font= 18
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 16))

for i, key in enumerate(keys):
    pair = pairs[i]
    if key not in ["pairs", "run_dir", "num_cases"]:
        if mapping[key] in mapping_2.keys():
            ax = axes[mapping_2[mapping[key]][0]-1, mapping_2[mapping[key]][1]-1]
            for j in range(4):
                ax.plot([el[0] for el in pairs[j:j]], [el[0] for el in rewrite_success[key][j:j]], color=colors[j], label = legend_labels[j], marker="s",linestyle="solid")
            arr_x = [el[0] for el in pairs]
            arr_y = [el[0] for el in rewrite_success[key]]
            err = [el[1]/1.96 for el in rewrite_success[key]]  # The error is computed with 95% interval of confidence. This make it 68%
            for line in range(4):
                ax.plot(arr_x[line::4], arr_y[line::4], color=colors[line], marker="s", linestyle="solid")
                ax.fill_between(arr_x[line::4], np.subtract(arr_y[line::4], err[line::4]), np.add(arr_y[line::4], err[line::4]), color=colors[line], alpha=0.1)
            print(key, baseline_ec[key][0])
            ax.plot([el[0] for el in pairs], [baseline_ce[key][0] for el in pairs], linestyle="dashed", label="Baseline CAT-ENG")
            ax.set_xlabel("Alpha parameter",fontsize=font)
            ax.set_title(mapping[key],fontsize=font)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            ax.tick_params(axis='both', which='major', labelsize=font-2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))


handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.07), borderaxespad=0., ncols=3,fontsize=font)

fig.tight_layout(rect=[0, 0.08, 1, 1])

plt.savefig('./figures/Attention_fail/Att.pdf')
plt.close(fig)
