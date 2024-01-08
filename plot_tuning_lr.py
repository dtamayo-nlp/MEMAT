from experiments.summarize import main as summarize
import matplotlib.pyplot as plt
from collections import defaultdict

els = ["run_000","run_001","run_002","run_003"]
vals = [0.05,0.1,0.2,0.5]
values = []

def error(dic):
	dic["post_score"] = list(dic["post_score"])
	dic["post_score"][1]=dic["post_score"][0]**2/3*(dic["post_neighborhood_success"][1]/dic["post_neighborhood_success"][0]**2+
	dic["post_paraphrase_success"][1]/dic["post_paraphrase_success"][0]**2+ 
	dic["post_rewrite_success"][1]/dic["post_rewrite_success"][0]**2)
	return dic


for el in els:
    values.append(error(summarize("Tuning_LR",[el],apply_std_norm=True, number_norm=1000)[0]))

keys = ['post_rewrite_success', 'post_rewrite_acc', 'post_paraphrase_success', 'post_paraphrase_acc', 'post_neighborhood_success', 'post_neighborhood_acc', 'post_score']
arrays = defaultdict(list)
errors = defaultdict(list)
for el in values:    
    for key in keys:
        arrays[key].append(el[key][0])
        errors[key].append(el[key][1]/1.96)

accuracies = ["post_rewrite_acc","post_paraphrase_acc","post_neighborhood_acc"]
success = ["post_rewrite_success","post_paraphrase_success","post_neighborhood_success"]

mapping ={
     "rewrite":"Efficacy",
     "paraphrase":"Generalization",
     "neighborhood":"Specificity"
}

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,6))
font = 18
import numpy as np

colors = ["tab:blue","tab:orange","tab:green"]
for j,key in enumerate(success):
    label_name = mapping[key[len("post_"):-len("_success")]]
    axes[0].plot(vals, arrays[key], label=label_name, marker="s")
    axes[0].fill_between(vals, np.subtract(arrays[key], errors[key]), np.add(arrays[key], errors[key]), alpha=0.1, color=colors[j])

axes[0].set_xlabel("learning rate",fontsize=font)
axes[0].set_ylabel("ES",fontsize=font)
# axes[0].legend(fontsize=font)
axes[0].set_title("Success",fontsize=font)
axes[0].tick_params(axis='both', which='major', labelsize=font-2)


for j,key in enumerate(accuracies):
    label_name = mapping[key[len("post_"):-len("_acc")]]
    axes[1].plot(vals, arrays[key], label=label_name, marker="s")
    axes[1].fill_between(vals, np.subtract(arrays[key], errors[key]), np.add(arrays[key], errors[key]), alpha=0.1, color=colors[j])

axes[1].set_xlabel("learning rate",fontsize=font)
axes[1].set_ylabel("EA",fontsize=font)
axes[1].set_ylim(top=101)
# axes[1].legend(loc="upper left",fontsize=font)
axes[1].set_title("Accuracy",fontsize=font)
axes[1].tick_params(axis='both', which='major', labelsize=font-2)


axes[2].plot(vals, arrays["post_score"], marker="s")
key="post_score"
axes[2].fill_between(vals, np.subtract(arrays[key], errors[key]), np.add(arrays[key], errors[key]), alpha=0.1, color=colors[0])
axes[2].set_xlabel("learning rate",fontsize=font)
axes[2].set_ylabel("Score",fontsize=font)
axes[2].set_title("Score",fontsize=font)
axes[2].tick_params(axis='both', which='major', labelsize=font-2)


handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.13), borderaxespad=0., ncols=4,fontsize=font)
fig.tight_layout(rect=[0, 0.14, 1, 1])
# plt.tight_layout()

plt.savefig("./figures/Tuning_LR/subplots.pdf")

plt.close()
