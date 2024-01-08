from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook

from .memit_hparams import MEMITHyperParams

def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    language: str,
    device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    if language == "catalan":

        # Compile list of rewriting and KL x/y pairs
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ], ["{} \u00e9s"]

    elif language == "english":

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

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=device)
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
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
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target



def compute_delta_both_languages(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITHyperParams,
    layer: int,
    context_templates: List[str],
    language: str,
    device = "cuda:0",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")

    # Tokenize target into list of int token IDs
    target_ids_cat = tok(request[0]["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Tokenize target into list of int token IDs
    target_ids_eng = tok(request[1]["target_new"]["str"], return_tensors="pt").to("cuda")[
        "input_ids"
    ][0]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts_cat, kl_prompts_cat = [
        context.format(request[0]["prompt"]) + tok.decode(target_ids_cat[:-1])
        for context_types in context_templates[0]
        for context in context_types
    ], ["{} \u00e9s"]

    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts_eng, kl_prompts_eng = [
        context.format(request[1]["prompt"]) + tok.decode(target_ids_eng[:-1])
        for context_types in context_templates[1]
        for context in context_types
    ], ["{} is a"]

    all_prompts_cat = rewriting_prompts_cat + kl_prompts_cat
    all_prompts_eng = rewriting_prompts_eng + kl_prompts_eng

    input_tok_cat = tok(
        [prompt.format(request[0]["subject"]) for prompt in all_prompts_cat],
        return_tensors="pt",
        padding=True,
    ).to(device)

    input_tok_eng = tok(
        [prompt.format(request[1]["subject"]) for prompt in all_prompts_eng],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Compute rewriting targets
    rewriting_targets_cat = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts_cat), *input_tok_cat["input_ids"].shape[1:]
    )

    # Compute rewriting targets
    rewriting_targets_eng = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts_eng), *input_tok_eng["input_ids"].shape[1:]
    )

    for i in range(len(rewriting_prompts_cat)):
        ex_len_cat = input_tok_cat["attention_mask"][i].sum()
        rewriting_targets_cat[i, ex_len_cat - len(target_ids_cat) : ex_len_cat] = target_ids_cat

    for i in range(len(rewriting_prompts_eng)):
        ex_len_eng = input_tok_eng["attention_mask"][i].sum()
        rewriting_targets_eng[i, ex_len_eng - len(target_ids_eng) : ex_len_eng] = target_ids_eng

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs_cat = [
        find_fact_lookup_idx(
            prompt, request[0]["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts_cat)
    ]

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs_eng = [
        find_fact_lookup_idx(
            prompt, request[1]["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts_eng)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=device)

    target_init_cat, kl_distr_init_cat = None, None

    target_init_eng, kl_distr_init_eng = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn_cat(cur_out, cur_layer):
        nonlocal target_init_cat

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init_cat is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init_cat = cur_out[0][0, lookup_idxs_cat[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs_cat):
                cur_out[0][i, idx, :] += delta

        return cur_out

    def edit_output_fn_eng(cur_out, cur_layer):
        nonlocal target_init_eng

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init_eng is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init_eng = cur_out[0][0, lookup_idxs_eng[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs_eng):
                cur_out[0][i, idx, :] += delta

        return cur_out

    
    def compute_loss(
            model,
            hparams,
            loss_layer,
            layer,
            edit_output_fn,
            input_tok,
            kl_prompts,
            lookup_idxs,
            rewriting_prompts,
            rewriting_targets,
            kl_distr_init,
            ln_f,
            lm_w,
            lm_b,
            delta,
            target_ids,
            language
    ):
        nonlocal target_init_cat, target_init_eng
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
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
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        
        if language == "catalan":
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init_cat) ** 2
            )
        elif language == "english":
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init_eng) ** 2
            )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss + weight_decay

        print(
            f"loss {language} {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of  [{request[0]['target_new']['str'] if language == 'catalan' else request[1]['target_new']['str']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )
        return loss
    
    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    for it in range(hparams.v_num_grad_steps):
        loss_cat = compute_loss(model,
                                hparams,
                                loss_layer,
                                layer,
                                edit_output_fn_cat,
                                input_tok_cat,
                                kl_prompts_cat,
                                lookup_idxs_cat,
                                rewriting_prompts_cat,
                                rewriting_targets_cat,
                                kl_distr_init_cat,
                                ln_f,
                                lm_w,
                                lm_b,
                                delta,
                                target_ids_cat,
                                "catalan")
        
        loss_eng = compute_loss(model,
                                hparams,
                                loss_layer,
                                layer,
                                edit_output_fn_eng,
                                input_tok_eng,
                                kl_prompts_eng,
                                lookup_idxs_eng,
                                rewriting_prompts_eng,
                                rewriting_targets_eng,
                                kl_distr_init_eng,
                                ln_f,
                                lm_w,
                                lm_b,
                                delta,
                                target_ids_eng,
                                "english")
        
        loss = 1/2 * loss_cat + 1/2 * loss_eng
        
        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = min(hparams.clamp_norm_factor * target_init_cat.norm(), hparams.clamp_norm_factor * target_init_eng.norm())

        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target_cat = target_init_cat + delta
    target_eng = target_init_eng + delta
    print(
        f"Catalan: Init norm {target_init_cat.norm()} | Delta norm {delta.norm()} | Target norm {target_cat.norm()}\n"
        f"English: Init norm {target_init_eng.norm()} | Delta norm {delta.norm()} | Target norm {target_eng.norm()}"
    )

    return delta


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


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
