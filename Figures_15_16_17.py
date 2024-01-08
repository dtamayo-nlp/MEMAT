from experiments.summarize import main
from pprint import pprint
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from matplotlib.ticker import MaxNLocator

def error(dic):
	dic["post_score"] = list(dic["post_score"])
	dic["post_score"][1]=dic["post_score"][0]**2/3*(dic["post_neighborhood_success"][1]/dic["post_neighborhood_success"][0]**2+
	dic["post_paraphrase_success"][1]/dic["post_paraphrase_success"][0]**2+ 
	dic["post_rewrite_success"][1]/dic["post_rewrite_success"][0]**2)
	return dic

relations = {"CE":["run_0{}".format(el) for el in range(0,4)],"EC":["run_0{}".format(el) for el in range(0,4)],
		"CC":["run_0{}".format(el) for el in range(0,4)], "EE":["run_0{}".format(el) for el in range(0,4)]}

path_mce = Path("./results")/"MEMIT_all"
path_mec = Path("./results")/"MEMIT_all"
path_mcc = Path("./results")/"MEMIT_all"
path_mee = Path("./results")/"MEMIT_all"


baseline_ce = error(main(path_mce, ["run_002"],apply_std_norm=True, number_norm=7000, abs_path=True)[0])
baseline_ec = error(main(path_mec, ["run_003"],apply_std_norm=True, number_norm=7000, abs_path=True)[0])

baseline_cc = error(main(path_mcc, ["run_000"],apply_std_norm=True, number_norm=7000, abs_path=True)[0])
baseline_ee = error(main(path_mee, ["run_001"],apply_std_norm=True, number_norm=7000, abs_path=True)[0])


keys = list(baseline_ce.keys())
num_heads = [1,2,5,8,10,15,30]
summaries_cat = defaultdict(lambda:[])
summaries_eng = defaultdict(lambda:[])
font=16
for (folder, runs) in relations.items():
	for run in runs:
		path_at = Path("./results")/folder/run/"attn_training"
		summaries_cat[folder].append(error(main(path_at,["catalan"], apply_std_norm=True, number_norm=3000,abs_path=True)[0]))
		summaries_eng[folder].append(error(main(path_at,["english"], apply_std_norm=True, number_norm=3000,abs_path=True)[0]))

	
folders = summaries_cat.keys()


rewrite_success_cat = defaultdict(lambda:defaultdict(lambda:[]))
rewrite_success_eng = defaultdict(lambda:defaultdict(lambda:[]))

for (folder,summaries) in summaries_cat.items():
	for el in summaries:
		for key in keys:
			rewrite_success_cat[folder][key].append(el[key])


for (folder,summaries) in summaries_eng.items():
	for el in summaries:
		for key in keys:
			rewrite_success_eng[folder][key].append(el[key])		
		
colors = ['red', 'green', 'blue', 'purple']
legend_labels = folders


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
num_heads=[8, 16, 32, 48]

mapping_3 = {
	"CC":"CAT-CAT",
	"EE":"ENG-ENG",
	"CE":"CAT-ENG",
	"EC":"ENG-CAT"
}

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# Iterate over keys and subplots
for i, key in enumerate(keys):
    if key not in ["pairs", "run_dir", "num_cases"]:
        if mapping[key] in mapping_2.keys():
            ax = axes[mapping_2[mapping[key]][0]-1, mapping_2[mapping[key]][1]-1]
            for j, (folder, dic) in enumerate(rewrite_success_eng.items()):
                arr_y = [el[0] for el in dic[key]]
                err = [el[1] / 1.96 for el in dic[key]]
                ax.plot(num_heads, arr_y, color=colors[j], marker="s", linestyle="solid", label=mapping_3[folder])
                ax.fill_between(num_heads, np.subtract(arr_y, err), np.add(arr_y, err), alpha=0.1, color=colors[j])
            ax.plot(num_heads, [baseline_ce[key][0] for el in num_heads], linestyle="dashed", label="Baseline CAT Training")
            ax.plot(num_heads, [baseline_ee[key][0] for el in num_heads], linestyle="dashed", label="Baseline ENG Training")
            ax.set_xlabel("Number of Heads",fontsize=font)
            ax.set_title(mapping[key],fontsize=font)
            ax.tick_params(axis='both', which='major', labelsize=font-2)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Create a common legend outside of the subplots and position it manually
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), borderaxespad=0., ncols=6,fontsize=font)

# Adjust layout to prevent clipping of titles and labels
fig.tight_layout(rect=[0, 0.05, 1, 1])

# Save the entire figure (all subplots) to the PDF
plt.savefig('./figures/MEMAT/English_plots.pdf')
plt.close(fig)


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))

# Iterate over keys and subplots
for i, key in enumerate(keys):
    if key not in ["pairs", "run_dir", "num_cases"]:
        if mapping[key] in mapping_2.keys():
            ax = axes[mapping_2[mapping[key]][0]-1, mapping_2[mapping[key]][1]-1]
            for j, (folder, dic) in enumerate(rewrite_success_cat.items()):
                arr_y = [el[0] for el in dic[key]]
                err = [el[1] / 1.96 for el in dic[key]]
                ax.plot(num_heads, arr_y, color=colors[j], marker="s", linestyle="solid", label=mapping_3[folder])
                ax.fill_between(num_heads, np.subtract(arr_y, err), np.add(arr_y, err), alpha=0.1, color=colors[j])
            ax.plot(num_heads, [baseline_cc[key][0] for el in num_heads], linestyle="dashed", label="Baseline CAT Training")
            ax.plot(num_heads, [baseline_ec[key][0] for el in num_heads], linestyle="dashed", label="Baseline ENG Training")
            ax.set_xlabel("Number of Heads",fontsize=font)
            ax.set_title(mapping[key],fontsize=font)
            ax.tick_params(axis='both', which='major', labelsize=font-2)
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Create a common legend outside of the subplots and position it manually
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.05), borderaxespad=0., ncols=6,fontsize=font)

# Adjust layout to prevent clipping of titles and labels
fig.tight_layout(rect=[0, 0.05, 1, 1])

# Save the entire figure (all subplots) to the PDF
plt.savefig('./figures/MEMAT/Catalan_plots.pdf')
plt.close(fig)



mapping_3 = {
	"CC":"CAT-CAT",
	"EE":"ENG-ENG",
	"CE":"CAT-ENG",
	"EC":"ENG-CAT"
}

font=22
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 8))
print(axes.shape)
ax = axes[0]
key = "post_score"
for j, (folder, dic) in enumerate(rewrite_success_eng.items()):
	arr_y = [el[0] for el in dic[key]]
	err = [el[1] / 1.96 for el in dic[key]]
	ax.plot(num_heads, arr_y, color=colors[j], marker="s", linestyle="solid", label=mapping_3[folder])
	ax.fill_between(num_heads, np.subtract(arr_y, err), np.add(arr_y, err), alpha=0.1,color=colors[j])
	ax.set_xlabel("Number of Heads",fontsize=font)
	ax.set_title("English "+ mapping[key],fontsize=font)
	ax.tick_params(axis='both', which='major', labelsize=font-2)


ax.plot(num_heads, [baseline_ce[key][0] for el in num_heads], linestyle="dashed", label="Baseline CAT Training")
ax.plot(num_heads, [baseline_ee[key][0] for el in num_heads], linestyle="dashed", label="Baseline ENG Training")
ax = axes[1]
for j, (folder, dic) in enumerate(rewrite_success_cat.items()):
	arr_y = [el[0] for el in dic[key]]
	err = [el[1] / 1.96 for el in dic[key]]
	ax.plot(num_heads, arr_y, color=colors[j], marker="s", linestyle="solid")
	ax.fill_between(num_heads, np.subtract(arr_y, err), np.add(arr_y, err), alpha=0.1,color=colors[j])
	ax.plot(num_heads, [baseline_cc[key][0] for el in num_heads], linestyle="dashed", c='#1f77b4')
	ax.plot(num_heads, [baseline_ec[key][0] for el in num_heads], linestyle="dashed", c = "tab:orange")
	ax.set_xlabel("Number of Heads",fontsize=font)
	ax.set_title("Catalan "+ mapping[key],fontsize=font)
	ax.tick_params(axis='both', which='major', labelsize=font-2)

# Create a common legend outside of the subplots and position it manually
handles, labels = axes[0].get_legend_handles_labels()


fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.13), borderaxespad=0., ncols=4,fontsize=font)

# Adjust layout to prevent clipping of titles and labels
fig.tight_layout(rect=[0, 0.14, 1, 1])

# Save the entire figure (all subplots) to the PDF
plt.savefig('./figures/MEMAT/Compare_Scores.pdf')
plt.close(fig)
