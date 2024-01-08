import os, re, json
import torch, numpy
from collections import defaultdict
from util import nethook
from util.globals import DATA_DIR
from experiments.causal_trace import (
    ModelAndTokenizer,
    layername,
    guess_subject,
    plot_trace_heatmap,
)
from experiments.causal_trace import (
    make_inputs,
    decode_tokens,
    find_token_range,
    predict_token,
    predict_from_input,
    collect_embedding_std,
)
import json 
from dsets import KnownsDataset
from tqdm import tqdm 
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from util.globals import *

# This is to avoid the computation of gradients, which will allow to 
# save resources
torch.set_grad_enabled(False)

model_name = AGUILA_MODEL

mt = ModelAndTokenizer(
    model_name,
    # low_cpu_mem_usage=True,
    torch_dtype=(torch.float16 if "20b" in model_name else None),
)
mt.tokenizer.pad_token = mt.tokenizer.eos_token

def trace_with_patch(
    model,  
    inp,  
    states_to_patch, 
    answers_t,  
    tokens_to_mix,  
    noise=0.1, 
    trace_layers=None,  
):
    prng = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)
    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                x[1:, b:e] += noise * torch.from_numpy(
                    prng.randn(x.shape[0] - 1, e - b, x.shape[2])
                ).to(x.device)
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            h[1:, t] = h[0, t]
        return x


    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def calculate_hidden_flow(
    mt, prompt, subject, samples=10, noise=0.1, window=10, kind=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = [d[0] for d in predict_from_input(mt.model, inp)]
    [answer] = decode_tokens(mt.tokenizer, [answer_t])
    e_range = find_token_range(mt.tokenizer, inp["input_ids"][0], subject)
    low_score = trace_with_patch(
        mt.model, inp, [], answer_t, e_range, noise=noise
    ).item()
    if not kind:
        differences = trace_important_states(
            mt.model, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt.model,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(model, num_layers, inp, e_range, answer_t, noise=0.1):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            r = trace_with_patch(
                model,
                inp,
                [(tnum, layername(model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    model, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                model, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def plot_hidden_flow(
    mt,
    prompt,
    subject=None,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
):
    if subject is None:
        subject = guess_subject(prompt)
    result = calculate_hidden_flow(
        mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
    )
    plot_trace_heatmap(result, savepdf, modelname=modelname)


def plot_all_flow(mt, prompt, subject=None, noise=0.1, modelname=None):
    current_path = os.getcwd()
    for kind in [None, "mlp", "self_attention"]:
        plot_hidden_flow(
            mt, prompt, subject, modelname=modelname, noise=noise, kind=kind, savepdf = current_path + f"/photo_{kind}.pdf"
        )


def plot_mean_heatmap(result, savepdf=None, title=None, xlabel=None, modelname=None):
    differences = result["scores"]
    low_score = result["low_score"]
    kind = (
        None
        if (not result["kind"] or result["kind"] == "None")
        else str(result["kind"])
    )
    window = result.get("window", 10)
    labels = list(result["input_tokens"])

    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(3.5, 2), dpi=200)
        h = ax.pcolor(
            differences,
            cmap={None: "Purples", "None": "Purples", "mlp": "Greens", "self_attention": "Reds"}[
                kind
            ],
            vmin=low_score,
        )
        ax.invert_yaxis()
        ax.set_yticks([0.5 + i for i in range(len(differences))])
        ax.set_xticks([0.5 + i for i in range(0, differences.shape[1] - 6, 5)])
        ax.set_xticklabels(list(range(0, differences.shape[1] - 6, 5)))
        ax.set_yticklabels(labels)
        if not modelname:
            modelname = "GPT"
        if not kind:
            ax.set_title("Avg Indirect Effect of $h_i^{(l)}$ over 100 prompts")
            ax.set_xlabel(f"single restored layer within {modelname}")
        else:
            kindname = "MLP" if kind == "mlp" else "self_attention"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input")
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
        cb = plt.colorbar(h)
        if title is not None:
            ax.set_title(title)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if savepdf:
            os.makedirs(os.path.dirname(savepdf), exist_ok=True)
            plt.savefig(savepdf, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_hidden_flows(
    mt,
    prompts,
    subjects,
    samples=10,
    noise=0.1,
    window=10,
    kind=None,
    modelname=None,
    savepdf=None,
):
    # THIS IS NOT OPTIMAL, you should've used sums instead of storing the tensors in memory, but like this we have the option of saving all the thing 
    # we want and then analyse then after

    # Important! The subjects and prompts must be passed in list format. If you don't know the subject, a pretty bad 
    # algorithm will try to predict it

    first_subj = []
    mid_subj = []
    last_subj = []
    first_after = []
    further = []
    last = []

    for i in tqdm(range(len(prompts))):
        prompt = prompts[i]
        if subjects[i] is None:
            subjects[i] = guess_subject(prompt)
        
        subject = subjects[i]
    
        result = calculate_hidden_flow(
            mt, prompt, subject, samples=samples, noise=noise, window=window, kind=kind
        )
        low_score = result["low_score"]

        # Number of tokens
        num_tok = result["scores"].shape[0]
        tok_ids = [i for i in range(num_tok)]

        # Let's store the scores of each prompt according to its position
        beg_subj, end_subj = result["subject_range"]
        end_subj -= 1
        if end_subj-beg_subj==0:
            last_subj.append(result["scores"][end_subj,:]-low_score)

        elif end_subj-beg_subj == 1:
            first_subj.append(result["scores"][beg_subj,:]-low_score)
            last_subj.append(result["scores"][end_subj,:]-low_score)
        else:
            first_subj.append(result["scores"][beg_subj,:]-low_score)
            last_subj.append(result["scores"][end_subj,:]-low_score)
            mid_subj += [result["scores"][beg_subj+i+1,:]-low_score for i in range(end_subj-beg_subj-1)]

        # We remove the elements of the subject   
        for el in range(end_subj-beg_subj+1):
            tok_ids.remove(beg_subj+el)

        # We add the final token elements
        last.append(result["scores"][-1,:]-low_score) # I am considering it can be overlap between subject tokens and "last token"
        # And also remove the token elements from the memory list
        if num_tok-1 in tok_ids:
            tok_ids.remove(num_tok-1)

        # If there's a subsequent token, we compute it
        if num_tok-end_subj-1>1:
            first_after.append(result["scores"][end_subj+1,:]-low_score)
            tok_ids.remove(end_subj+1)
        
        # We store the rest of elements in "further"
        for id_ in tok_ids:
            further.append(result["scores"][id_,:]-low_score)

    first_subj = torch.vstack(first_subj).to(torch.float)
    
    mid_subj = torch.vstack(mid_subj).to(torch.float)
    last_subj = torch.vstack(last_subj).to(torch.float)
    first_after = torch.vstack(first_after).to(torch.float)
    further = torch.vstack(further).to(torch.float)
    last = torch.vstack(last).to(torch.float)

    all_tensors = {'first_subj': first_subj, 
                'mid_subj': mid_subj,
                "last_subj": last_subj,
                "first_after": first_after, 
                "further": further,
                "last":last}
    
    torch.save(all_tensors, f"all_tensors_{kind}_{modelname}.pth")

    first_subj = first_subj.mean(dim=0)
    mid_subj = mid_subj.mean(dim=0)
    last_subj = last_subj.mean(dim=0)
    first_after = first_after.mean(dim=0)
    further = further.mean(dim=0)
    last = last.mean(dim=0)

    scores = torch.vstack([first_subj,mid_subj,last_subj, first_after, further, last])

    new_result = dict(
        scores = scores,    
        low_score = 0,
        input_tokens = ["First subject tokens", "Middle subject tokens", "Last subject token", "First subsequent token", "Further tokens", "Last token"],
        kind = kind
    )
    plot_mean_heatmap(new_result, savepdf, modelname=modelname)


def plot_all_flows(mt, prompts, subjects, noise = 0.1, modelname = None, index = 0):
    current_path = os.getcwd()
    for kind in ["self_attention","mlp",None]:
        plot_hidden_flows(
            mt, prompts, subjects, modelname=modelname, noise=noise, kind=kind, savepdf = f"./figures/causal_trace/photo_{kind}_{index}.pdf"
        )

# Read catalan files:

file = str(DATA_DIR / "catalan_CF.json")

with open(file, "r") as f:
    data = json.load(f)

# We did a selection of the top accuracy examples 
selection = np.load("./original_indxs.npy")

data = [data[i] for i in selection]

prompts_catalan = [el["eval_target_true"]["requested_rewrite"]["prompt"].format(el["eval_target_true"]["requested_rewrite"]["subject"]) for el in data]
subjects_catalan = [el["eval_target_true"]["requested_rewrite"]["subject"] for el in data]

noise_level = 0.20331135392189026


plot_all_flows(mt, prompts_catalan[:100], subjects_catalan[:100], noise=noise_level, index = "catalan_100", modelname = "catalan_100")