import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast, generate_mine, load_pipeline
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx, compute_delta_both_languages
from .memit_hparams import MEMITHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

# ----------------------------------------------------
# Code for dealing with zeroing in layers
def make_null_i(matrix, i):
    new_matrix = matrix.clone()
    new_matrix[:,i] = new_matrix[:,i]*0
    new_matrix[i,:] = new_matrix[i,:]*0
    return new_matrix

def identify_null_cols(matrix):
    # Check if all elements in each row are zero
    row_sums = matrix.clone().sum(dim=1)
    zero_rows = torch.nonzero(row_sums == 0).squeeze()
    return zero_rows.numel(), zero_rows.tolist()

def remove_column(matrix, i):
    new_matrix = matrix.clone()
    new_matrix = torch.cat((new_matrix[:i], new_matrix[i+1:]), dim=0)
    new_matrix = torch.cat((new_matrix[:, :i], new_matrix[:, i+1:]), dim=1)
    return new_matrix

def add_zero_column(matrix, i):
    new_matrix = matrix.clone()
    new_row = torch.zeros(1, matrix.shape[1], device=matrix.device, dtype=matrix.dtype)
    new_col = torch.zeros(matrix.shape[0] + 1, 1, device=matrix.device, dtype=matrix.dtype)
    new_matrix = torch.cat((new_matrix[:i], new_row, new_matrix[i:]), dim=0)
    new_matrix = torch.cat((new_matrix[:, :i], new_col, new_matrix[:, i:]), dim=1)
    return new_matrix

def compute_pseudoinverse_matrix(matrix):
    # Note that this pseudoinverse is not related to the Moore Penrose Inverse
    n, ids = identify_null_cols(matrix)
    print(f"There are {n} columns with zeros")
    if n==0:
        return torch.linalg.inv(matrix)
    # Remove the zero columns that are causing our matrix to be singular
    new_matrix = matrix.clone()
    for id_ in ids[::-1]:
        new_matrix = remove_column(new_matrix, id_)
    # Computing inverse
    new_matrix = torch.linalg.inv(new_matrix)
    # Rescaling the matrix
    for id_ in ids:
        new_matrix = add_zero_column(new_matrix,id_)
    return new_matrix
# ----------------------------------------------------

def apply_memit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    save_delta_matrix: str,
    language: str,
    copy=False,
    return_orig_weights=False,
    device = "0",
    cache_template: Optional[str] = None,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    weights_copy = {}
    if copy:
        model = deepcopy(model)
    
    if language == "both_separated":
        # This case is a bit different since it performs two separated trainings
        both = ["catalan", "english"]
        deltas = execute_memit_both_separated(model, tok, requests, hparams, both, device, cache_template=cache_template)

        if save_delta_matrix is not None:
            for i,delta in enumerate(deltas):
                torch.save(delta, save_delta_matrix + f"_{len(requests[i])}_{both[i]}")

        with torch.no_grad():
            for delta in deltas:
                for w_name, (key_mat, val_mat) in delta.items():
                    key_mat, val_mat = key_mat.to(device), val_mat.to(device)
                    upd_matrix = key_mat @ val_mat.T
                    w = nethook.get_parameter(model, w_name)
                    upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                    if return_orig_weights and w_name not in weights_copy:
                        weights_copy[w_name] = w.detach().clone()

                    w[...] += upd_matrix.float()
        return model, weights_copy

    if language == "both": 
        # Computes the loss at the same time
        deltas = execute_memit_both(model, tok, requests, hparams, language, device, cache_template=cache_template)

    else:
        deltas = execute_memit(model, tok, requests, hparams, language, device, cache_template=cache_template)
    if save_delta_matrix is not None:
        torch.save(deltas, save_delta_matrix + f"_{len(requests)}_{language}")
    

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to(device), val_mat.to(device)
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    return model, weights_copy


def execute_memit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    language: str,
    device: str,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}
    requests = deepcopy(requests)

    # Update target and print info
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )


    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        ).detach().cpu()
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model, tok, language, device)
    
    z_layer = hparams.layers[-1]
    z_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, hparams.v_lr, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to(device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
                language,
                device,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T

        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        n_texts = hparams.mom2_n_samples
        

        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            device, 
            language,
            force_recompute=force_recompute,
        ).detach().cpu()

        layer_ks, targets = (
                layer_ks.double(),
                targets.double(),
            )
        r = torch.cuda.memory_reserved(0)

        # print("Memory all device 1!", torch.cuda.memory_allocated(1))

        matrix = hparams.mom2_update_weight * cov.double().cpu() + layer_ks.detach().cpu() @ layer_ks.T.detach().cpu()
        
        # print("Memory all device 1!", torch.cuda.memory_allocated(1))
        n_nul_cols,_ = identify_null_cols(matrix)

        # This computation allows to deal with zero columns / singular matrices. 
        # It is not a perfect approach and can make some norms too big.
        if n_nul_cols != 0:
            adj_k = compute_pseudoinverse_matrix(matrix) @ layer_ks.detach().cpu()
        else:
            adj_k = torch.linalg.solve(matrix,layer_ks.detach().cpu())
    
        cov.cpu()

        resid = targets.detach().cpu() / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name].cpu() + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        for x in [layer_ks, cur_zs, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    print("FINISHED")

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas

def execute_memit_both_separated(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    languages: str,
    device: str,
    cache_template: Optional[str] = None,
):
    deltas_lang1 = execute_memit(model,
                                 tok,
                                 requests[0],
                                 hparams,
                                 languages[0],
                                 device,
                                 cache_template[0])
    
    deltas_lang2 = execute_memit(model,
                                tok,
                                requests[1],
                                hparams,
                                languages[1],
                                device,
                                cache_template[1])
    
    return [deltas_lang1, deltas_lang2]
    

def execute_memit_both(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    language: str,
    device: str,
    cache_template: Optional[str] = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    deltas = {}
    requests = deepcopy(requests)

    for i, request in enumerate(requests):
        if request[0]["target_new"]["str"][0]!= " ":
            # Space required for correct tokenization
            requests[i][0]["target_new"]["str"] = " " + requests[i][0]["target_new"]["str"]
            requests[i][1]["target_new"]["str"] = " " + requests[i][1]["target_new"]["str"]

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        ).detach().cpu()
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    context_templates_cat = get_context_templates(model, tok, "catalan", device)
    context_templates_eng = get_context_templates(model, tok, "english", device)
    context_templates = (context_templates_cat,context_templates_eng)
    
    z_layer = hparams.layers[-1]
    delta_list = []

    for request in requests:
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, hparams.v_lr, str(request[0]["case_id"]) + "_both_languages_loss"
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                delta_list.append(torch.from_numpy(data["v_star"]).to(device))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_delta = compute_delta_both_languages(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
                language,
            )

            delta_list.append(cur_delta)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_delta.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")

    targets = torch.stack(delta_list, dim=1)

    # Delta multilingual, but we decide on which data we work
    # By both, we will be considering to train on Catalan, but with the loss function in English Also 

    # Insert
    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks_cat = compute_ks(model, tok, [request[0] for request in requests], hparams, layer, context_templates[0]).T
        layer_ks_eng = compute_ks(model, tok, [request[1] for request in requests], hparams, layer, context_templates[1]).T

        layer_ks = 1/2 * layer_ks_cat + 1/2 * layer_ks_eng

        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)

        # Load covariance matrix
        force_recompute = False
        n_texts = hparams.mom2_n_samples

        cov = 1/2 * get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            device, 
            "catalan",
            force_recompute=force_recompute,
        ).detach().cpu() + 1/2 * get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            device, 
            "english",
            force_recompute=force_recompute,
        ).detach().cpu()

        layer_ks, targets = (
                layer_ks.double(),
                targets.double(),
            )


        matrix = hparams.mom2_update_weight * cov.double().cpu() + layer_ks.detach().cpu() @ layer_ks.T.detach().cpu()
        
        n_nul_cols,_ = identify_null_cols(matrix)

        # This computation allows to deal with zero columns / singular matrices. 
        # It is not a perfect approach and can make some norms too big.
        if n_nul_cols != 0:
            adj_k = compute_pseudoinverse_matrix(matrix) @ layer_ks.detach().cpu()
        else:
            adj_k = torch.linalg.solve(matrix,layer_ks.detach().cpu())
    
        cov.cpu()

        resid = targets.detach().cpu() / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name].cpu() + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        for x in [layer_ks, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    print("FINISHED")

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    device: str,
    language: str, 
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name, language)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            language,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to(device)) if inv else COV_CACHE[key].to(device)
    )

def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
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


def get_context_templates(model, tok, language, device):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        if language == "english":
            prefixes = ["The", "Therefore", "Because", "She", "I love"]
        elif language == "catalan":
            prefixes = ["El", "Per tant", "Perqu\u00e8", "Ella", "M'encanta"]
        CONTEXT_TEMPLATES_CACHE = [["{}"]]
        for length, n_gen in [(10,5)]:
            pipeline_base = load_pipeline(model, tok, device, max_out_len=length)
            for f in generate_mine(model, tok, prefixes, pipeline_base):
                CONTEXT_TEMPLATES_CACHE.append([f.replace("{", " ").replace("}", " ") + ". {}"])

        print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

    return CONTEXT_TEMPLATES_CACHE