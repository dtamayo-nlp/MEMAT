import torch
from time import time
import shutil
import torch.nn as nn
from util import nethook
from typing import Any, Dict, List, Optional,Tuple, Union
from transformers import AutoTokenizer
from Falcon.modelling_RW import RWForCausalLM
import torch.nn.functional as F
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from memit.memit_main import get_context_templates
import copy
import numpy as np
from rome import repr_tools
from memit import MEMITHyperParams
from util.globals import *

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def heads_optimization(
    model: RWForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: MEMITHyperParams,
    language: str,
    device: str,
    top_heads: dict,
    batch_size = 32,
) -> Dict[str, Tuple[torch.Tensor]]:

    requests_copy = copy.deepcopy(requests)
    # Let us update the format of the prompts
    for i, request in enumerate(requests_copy):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests_copy[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    if language == "catalan":
        context_templates = get_context_templates(model, tok, language, device)
        
    elif language == "english":
        context_templates = get_context_templates(model, tok, language, device)


    deltas = [torch.zeros((model.config.hidden_size//model.config.n_head,), requires_grad=True, device=device) for el in top_heads]
    target_init_attn, kl_distr_init = [None for el in top_heads], None

    opt = torch.optim.Adam(deltas, lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)
    kl_factor = hparams.kl_factor

    def edit_output_fn(cur_out, cur_layer):

        nonlocal target_init_attn

        # OKAY, for the future, cur_out has lenght (# batch, # toks, # heads, dim_head)

        for pair_index, wname in enumerate([hparams.attn_module_tmp.format(pair[0]) for pair in top_heads]):

            # head_layer = top_heads[pair_index][0]
            head_number = top_heads[pair_index][1]

            if cur_layer == wname:

                if target_init_attn[pair_index] is None:
                    target_init_attn = cur_out[0,-1,head_number].detach().clone() # This will have length 64 

                # Add intervened delta
                for i, idx in enumerate(lookup_idxs):
                    cur_out[i, -1, head_number, :] += deltas[pair_index]

        return cur_out


    loss_layer = hparams.v_loss_layer
    num_epochs = 1
    for epoch in range(num_epochs):
        for batch in chunks(requests_copy, batch_size):
            
            loss = 0
            opt.zero_grad()

            for request in batch:

                # Tokenize target into list of int token IDs
                target_ids = tok(request["target_new"]["str"], return_tensors="pt").to(device)[
                    "input_ids"
                ][0]

                # Compile list of rewriting and KL x/y pairs
                rewriting_prompts, kl_prompts = [
                    context.format(request["prompt"]) + tok.decode(target_ids[:-1])
                    for context_types in context_templates
                    for context in context_types
                ], ["{} is a"]
                all_prompts = rewriting_prompts + kl_prompts

                input_tok = tok(
                    [prompt.format(request["subject"]) for prompt in all_prompts],
                    return_tensors="pt",
                    padding=True,
                ).to(device)

                # Compute rewriting targets
                rewriting_targets = torch.tensor(-100, device=device).repeat(
                    len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
                )
                for i in range(len(rewriting_prompts)):
                    ex_len = input_tok["attention_mask"][i].sum()
                    rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

                lookup_idxs = [
                    find_fact_lookup_idx(
                        prompt, request["subject"], tok, hparams.fact_token, verbose=False
                    )
                    for i, prompt in enumerate(all_prompts)
                ]
                # Forward propagation
                with nethook.TraceDict(
                    module=model,
                    layers=[hparams.layer_module_tmp.format(loss_layer)]+[hparams.attn_module_tmp.format(head_layer[0]) for head_layer in top_heads],
                    retain_input=False,
                    retain_output=True,
                    edit_output=edit_output_fn,
                ) as tr:
                    logits = model(**input_tok).logits

                    # Compute distribution for KL divergence
                    kl_logits = torch.stack(
                        [
                            logits[i - len(kl_prompts), idx, :]
                            for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                        ],
                        dim=0,
                    )
                    kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                    if kl_distr_init is None:
                        kl_distr_init = kl_log_probs.detach().clone()
                
                # Compute loss on rewriting targets
                full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
                    : len(rewriting_prompts)
                ]
                log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w + lm_b, dim=2)
                loss_i = torch.gather(
                    log_probs,
                    2,
                    torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
                ).squeeze(2)
                mask = (rewriting_targets != -100).float()

                # Aggregate total losses
                nll_loss_each = -(loss_i * mask).sum(1) / target_ids.size(0)
                nll_loss =  nll_loss_each.mean()
                kl_loss = kl_factor * torch.nn.functional.kl_div(
                    kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
                )

                weight_decay = 0
                for delta in deltas:
                    weight_decay += hparams.v_weight_decay *  (
                       torch.norm(delta) / torch.norm(target_init_attn) ** 2
                    )
                weight_decay = weight_decay / len(deltas)

                # print(nll_loss_each)

                loss = nll_loss + kl_loss + weight_decay
                # print(
                #     f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                #     f"avg prob of [{request['target_new']['str']}] "
                #     f"{torch.exp(-nll_loss_each).mean().item()}"
                # )
                loss = loss / batch_size
                loss.backward()
                
            # Execute optimization 
            torch.nn.utils.clip_grad_norm(model.parameters(),1)
            opt.step()

            for delta in deltas:
                max_norm = hparams.clamp_norm_factor * target_init_attn.norm()
                if delta.norm() > max_norm:
                    with torch.no_grad():
                        delta[...] = delta * max_norm / delta.norm()

    return deltas

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

def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret