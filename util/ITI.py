import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle
from .nethook import TraceDict
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from einops import rearrange

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    Some models have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )

def tokenization_english(dataset, tokenizer):
    """
    This function takes a dataset in list of dicts format (CounterFact) and applies the tokenizer
    Outputs:
        - all_prompts: tokenized prompts with the form 'prompt.format(sub)+target'
        - all_labels: labels that indicate the 'new' targets as 1 and the 'old' targets as 0
    """

    all_prompts = []
    all_labels = []

    for el in dataset:
        prompt = el["requested_rewrite"]["prompt"].format(el["requested_rewrite"]["subject"])
        target_new = el["requested_rewrite"]["target_new"]["str"]
        target_old = el["requested_rewrite"]["target_true"]["str"]

        sentence_new = tokenizer(prompt + " " + target_new,return_tensors="pt")["input_ids"]
        sentence_old = tokenizer(prompt + " " + target_old,return_tensors="pt")["input_ids"]

        all_prompts += [sentence_new, sentence_old]
        all_labels += [1,0] 
    
    return all_prompts, all_labels

def tokenization_catalan(dataset, tokenizer):
    """
    This function takes a dataset in list of dicts format (CounterFact) and applies the tokenizer
    Outputs:
        - all_prompts: tokenized prompts with the form 'prompt.format(sub)+target'
        - all_labels: labels that indicate the 'new' targets as 1 and the 'old' targets as 0
    """

    all_prompts = []
    all_labels = []

    for el in dataset:

        prompt_new = el["eval_target_new"]["requested_rewrite"]["prompt"].format(el["eval_target_new"]["requested_rewrite"]["subject"])
        target_new = el["eval_target_new"]["requested_rewrite"]["target_new"]["str"]

        prompt_old = el["eval_target_true"]["requested_rewrite"]["prompt"].format(el["eval_target_true"]["requested_rewrite"]["subject"])
        target_old = el["eval_target_true"]["requested_rewrite"]["target_true"]["str"]

        sentence_new = tokenizer(prompt_new + " " + target_new,return_tensors="pt")["input_ids"]
        sentence_old = tokenizer(prompt_old + " " + target_old,return_tensors="pt")["input_ids"]

        all_prompts += [sentence_new, sentence_old]
        all_labels += [1,0]
    
    return all_prompts, all_labels

def get_falcon_activations_bau(model, prompt, device):

    model.eval()
    HEADS = [f"transformer.h.{i}.self_attention.head_out" for i in range(model.config.num_hidden_layers)]

    with torch.no_grad():
        prompt = prompt.to(device)
        with TraceDict(model, HEADS) as ret:
            output = model(prompt, output_hidden_states = True)
        hidden_states = output.hidden_states
        hidden_states = torch.stack(hidden_states, dim = 0).squeeze()
        hidden_states = hidden_states.detach().cpu().numpy()
        head_wise_hidden_states = [ret[head].output.squeeze().detach().cpu() for head in HEADS]
        head_wise_hidden_states = torch.stack(head_wise_hidden_states, dim = 0).squeeze().numpy()

    return hidden_states, head_wise_hidden_states

def flattened_idx_to_layer_head(flattened_idx, num_heads):
    return flattened_idx // num_heads, flattened_idx % num_heads

def layer_head_to_flattened_idx(layer, head, num_heads):
    return layer * num_heads + head

def get_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass, use_random_dir, com_directions): 
    interventions = {}
    for layer, head in top_heads: 
        interventions[f"transformer.h.{layer}.self_attention.head_out"] = []
    for layer, head in top_heads:
        if use_center_of_mass: 
            direction = com_directions[layer_head_to_flattened_idx(layer, head, num_heads)]
        elif use_random_dir: 
            direction = np.random.normal(size=(64,))
        else: 
            direction = probes[layer_head_to_flattened_idx(layer, head, num_heads)].coef_

        direction = direction / np.linalg.norm(direction)
        activations = tuning_activations[:,layer,head,:] # batch x 64
        proj_vals = activations @ direction.T
        proj_val_std = np.std(proj_vals)
        interventions[f"transformer.h.{layer}.self_attention.head_out"].append((head, direction.squeeze(), proj_val_std))
    for layer, head in top_heads: 
        interventions[f"transformer.h.{layer}.self_attention.head_out"] = sorted(interventions[f"transformer.h.{layer}.self_attention.head_out"], key = lambda x: x[0])

    return interventions

def get_separated_activations(labels, head_wise_activations): 
    # separate activations by example
    idxs_to_split_at = np.cumsum([2 for i in range(len(labels)//2)])
    labels = list(labels)
    separated_labels = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)
    return separated_head_wise_activations, separated_labels, idxs_to_split_at

def get_top_heads(train_idxs, val_idxs, separated_activations, separated_labels, num_layers, num_heads, seed, num_to_intervene, use_random_dir=False, save_figure=None):
    probes, all_head_accs_np = train_probes(seed, train_idxs, val_idxs, separated_activations, separated_labels, num_layers=num_layers, num_heads=num_heads)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)
    top_heads = []

    if save_figure is not None:
        fig, ax = plt.subplots()
        matrix = np.sort(np.array(all_head_accs_np.copy()).reshape(32,71),axis=1)[:,::-1]
        im = ax.imshow(matrix * 100, cmap='Blues')

        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        colorbar = plt.colorbar(im, cax=cax)

        colorbar.locator = ticker.MaxNLocator(nbins=5, integer=True)
        colorbar.update_ticks()
        colorbar.ax.tick_params(labelsize=14)

        yticks = range(0, 32, 5)
        ax.set_yticks(yticks)
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlabel('Heads (sorted)',fontsize=16)
        ax.set_ylabel('Layers',fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=14)

        plt.savefig(save_figure)
        plt.close(fig)

    print("The top percentages are:", np.sort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:10])
    top_accs = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene]

    print("The top 20 optimal heads are: ", np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:20])
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]
    if use_random_dir: 
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(num_heads*num_layers, num_heads*num_layers, replace=False)
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]
    return top_heads, probes

def train_probes(seed, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads):
    all_head_accs = []
    probes = []
    all_X_train = np.concatenate([separated_head_wise_activations[i] for i in train_set_idxs], axis = 0)
    all_X_val = np.concatenate([separated_head_wise_activations[i] for i in val_set_idxs], axis = 0)
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis = 0)
    for layer in tqdm(range(num_layers)): 
        for head in range(num_heads): 
            X_train = all_X_train[:,layer,head,:]
            X_val = all_X_val[:,layer,head,:]
            clf = LogisticRegression(random_state=seed, max_iter=1000).fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)
    all_head_accs_np = np.array(all_head_accs)
    return probes, all_head_accs_np

def save_probes(probes, path): 
    """takes in a list of sklearn lr probes and saves them to path"""
    with open(path, 'wb') as f: 
        pickle.dump(probes, f)

def get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels): 
    """
    This function computes, for each head, the mean direction of the "true" and "false" values
    """
    com_directions = []
    for layer in range(num_layers): 
        for head in range(num_heads): 
            usable_idxs = np.concatenate([train_set_idxs, val_set_idxs], axis=0)
            usable_head_wise_activations = np.concatenate([separated_head_wise_activations[i][:,layer,head,:] for i in usable_idxs], axis=0)
            usable_labels = np.concatenate([separated_labels[i] for i in usable_idxs], axis=0)
            true_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 1], axis=0)
            false_mass_mean = np.mean(usable_head_wise_activations[usable_labels == 0], axis=0)
            com_directions.append(true_mass_mean - false_mass_mean)
    com_directions = np.array(com_directions)
    return com_directions


def add_attention_direction(
    model, 
    tok, 
    head_wise_activations, 
    labels,
    top_num_heads,  
    alpha, 
    device, 
    seed, 
    train_set_idxs, 
    val_set_idxs, 
    use_center_of_mass = True,
    use_random_dir = False,
    ):
    start = time.time()

    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    # We separate the head activations in a specific format to deal with it better
    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)

    # If we want to use the mean difference of attention heads as a direction, we can use the `use_center_of_mass` parameter (this will be given by default)
    if use_center_of_mass:
        com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
    else:
        com_directions = None

    # We train the orobes to determine which "top heads" have valuable information for "language independence directions" 
    top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, seed, top_num_heads, use_random_dir)
    # This function returns a dict with the layers to be edited as key, and the tuple (head, direction, std) as value.
    # I don't understand why all the heads (wouldn't be better to do it with training only?) are used to compute the std (TO DO: CHANGE THIS!!!)
    interventions = get_interventions_dict(top_heads, probes, head_wise_activations, num_heads, use_center_of_mass, use_random_dir, com_directions)

    # We insert the new "language-indep" directions in the model
    for head_out_name, list_int_vec in interventions.items():
        layer_no = int(head_out_name.split('.')[2])
        displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
        for head_no, head_vec, std in list_int_vec:
            displacement[head_no] = alpha * std * head_vec
        displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
        bias_tobe = F.linear(displacement.to(torch.float32), model.transformer.h[layer_no].self_attention.dense.weight).to(device)
        model.transformer.h[layer_no].self_attention.dense.bias = nn.parameter.Parameter(bias_tobe)

    print(f"Adding attention took {int(time.time()-start)} seconds")
    
    return model


def add_attention_direction_second(
    model, 
    tok, 
    head_wise_activations, 
    labels,
    top_num_heads,  
    alpha, 
    device, 
    seed, 
    ids_top_accs,
    val_per,
    use_center_of_mass = True,
    use_random_dir = False,
    ):
    start = time.time()

    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    # We separate the head activations in a specific format to deal with it better
    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)
    separated_head_wise_activations_topacc = [separated_head_wise_activations[i] for i in ids_top_accs]
    separated_labels_topacc = [separated_labels[i] for i in ids_top_accs]

    train_set_idxs = [i for i in range(int(len(ids_top_accs)*(1-val_per)))]
    val_set_idxs = [i for i in range(int(len(ids_top_accs)*(1-val_per)),len(ids_top_accs))]

    # If we want to use the mean difference of attention heads as a direction, we can use the `use_center_of_mass` parameter (this will be given by default)
    if use_center_of_mass:
        com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations_topacc, separated_labels_topacc)
    else:
        com_directions = None

    # We train the orobes to determine which "top heads" have valuable information for "language independence directions" 
    top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations_topacc, separated_labels_topacc, num_layers, num_heads, seed, top_num_heads, use_random_dir)

    # This function returns a dict with the layers to be edited as key, and the tuple (head, direction, std) as value.
    interventions = get_interventions_dict(top_heads, probes, head_wise_activations, num_heads, use_center_of_mass, use_random_dir, com_directions)

    # We insert the new "language-indep" directions in the model
    for head_out_name, list_int_vec in interventions.items():
        layer_no = int(head_out_name.split('.')[2])
        displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
        for head_no, head_vec, std in list_int_vec:
            displacement[head_no] = alpha * std * head_vec
        displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
        bias_tobe = F.linear(displacement.to(torch.float32), model.transformer.h[layer_no].self_attention.dense.weight).to(device)
        model.transformer.h[layer_no].self_attention.dense.bias = nn.parameter.Parameter(bias_tobe)

    print(f"Adding attention took {int(time.time()-start)} seconds")
    
    return model


def determine_interventions(
    model, 
    head_wise_activations, 
    labels,
    top_num_heads,  
    seed, 
    ids_top_accs,
    val_per,
    use_random_dir = False,
    save_figure = None,
    ):

    num_layers = model.config.n_layer
    num_heads = model.config.n_head

    # We separate the head activations in a specific format to deal with it better
    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)

    separated_head_wise_activations_topacc = [separated_head_wise_activations[i] for i in ids_top_accs]
    separated_labels_topacc = [separated_labels[i] for i in ids_top_accs]

    random.seed(123)
    random.shuffle(separated_head_wise_activations_topacc)
    random.shuffle(separated_labels_topacc)

    train_set_idxs = [i for i in range(int(len(ids_top_accs)*(1-val_per)))]
    val_set_idxs = [i for i in range(int(len(ids_top_accs)*(1-val_per)),len(ids_top_accs))]

    # We train the orobes to determine which "top heads" have valuable information for "language independence directions" 
    top_heads, _ = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations_topacc, separated_labels_topacc, num_layers, num_heads, seed, top_num_heads, use_random_dir, save_figure= save_figure)
    print(top_heads)
    return top_heads