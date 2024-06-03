import json
import pprint 
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import os

from util.globals import AGUILA_MODEL

with open("./data/catalan_CF.json", "r") as f:
    data_ca = json.load(f)

with open("./data/english_CF.json", "r") as f:
    data_en = json.load(f)

subjects_en = [el["requested_rewrite"]["subject"] for el in data_en]
subjects_ca = [el["eval_target_new"]["requested_rewrite"]["subject"] for el in data_ca]

rel_en = [prom["requested_rewrite"]["prompt"].replace("{}","") for prom in data_en]
rel_ca = [prom["eval_target_new"]["requested_rewrite"]["prompt"].replace("{}","") for prom in data_ca]

tar_en = [el["requested_rewrite"]["target_new"]["str"] for el in data_en]
tar_ca = [el["eval_target_new"]["requested_rewrite"]["target_new"]["str"] for el in data_ca]

tok = AutoTokenizer.from_pretrained(AGUILA_MODEL)

len_analysis = 11229
sim_subjects = []
sim_relations = []
sim_tar = []

print(len(data_ca), len(data_en))
for i in range(len_analysis):

    sub_cat = tok.encode(subjects_ca[i])
    sub_eng = tok.encode(subjects_en[i])

    sub_inter = len(set(sub_cat).intersection(set(sub_eng)))/len(set(sub_cat).union(set(sub_eng)))
    sim_subjects.append(sub_inter)

    tar_cat = tok.encode(tar_ca[i])
    tar_eng = tok.encode(tar_en[i])
    
    tar_inter = len(set(tar_cat).intersection(set(tar_eng)))/len(set(tar_cat).union(set(tar_eng)))
    sim_tar.append(tar_inter)

    rel_cat = tok.encode(rel_ca[i])
    rel_eng = tok.encode(rel_en[i])
    
    rel_inter = len(set(rel_cat).intersection(set(rel_eng)))/len(set(rel_cat).union(set(rel_eng)))
    sim_relations.append(rel_inter)


def create_histogram(data, xlabel, ylabel, title, save_filename):
    font = 20
    plt.hist(data, bins=10, color='#008B8B', edgecolor='black')  
    plt.xlabel(xlabel, fontsize=font)
    plt.ylabel(ylabel, fontsize=font)
    # plt.title(title, fontsize=font)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tick_params(axis='both', which='major', labelsize=font-4)
    plt.tight_layout()
    plt.savefig(save_filename, dpi=300)
    plt.close()

create_histogram(sim_relations, 'Jaccard index: $J(r_{eng},r_{cat})$', 'Frequency', '', './figures/analysis_data/sim_rel_new.pdf')
create_histogram(sim_subjects, 'Jaccard index: $J(s_{eng},s_{cat})$', 'Frequency', '', './figures/analysis_data/sim_subj_new.pdf')
create_histogram(sim_tar, 'Jaccard index: $J(o_{eng}^*,o_{cat}^*)$', 'Frequency', '', './figures/analysis_data/sim_tar_new.pdf')


# # Create dataset
# catalan_sub_chosen, english_sub_chosen = zip(*[(data_ca[i], data_en[i]) for i, el in enumerate(sim_subjects) if el < 0.5])

# with open("./data/data_cat_dif_subj.json", "w") as f:
#     f.write(json.dumps(catalan_sub_chosen, indent = 4))

# with open("./data/data_eng_dif_subj.json", "w") as f:
#     f.write(json.dumps(english_sub_chosen, indent = 4))



# catalan_sub_chosen, english_sub_chosen = zip(*[(data_ca[i], data_en[i]) for i, el in enumerate(sim_subjects) if el == 1])

# with open("./data/data_cat_eq_subj.json", "w") as f:
#     f.write(json.dumps(catalan_sub_chosen, indent = 4))

# with open("./data/data_eng_eq_subj.json", "w") as f:
#     f.write(json.dumps(english_sub_chosen, indent = 4))