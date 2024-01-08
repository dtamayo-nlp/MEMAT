import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import argparse
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from Falcon.modelling_RW import RWForCausalLM

from transformers import AutoTokenizer

# from baselines.ft import FTHyperParams, apply_ft_to_model
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact, compute_rewrite_quality_counterfact_new
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from util import nethook
from util.ITI import get_falcon_activations_bau, determine_interventions
from util.globals import *

from attention_translation.multiple_head_optimization import heads_optimization

from experiments.summarize import summary

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    # "ROME": (ROMEHyperParams, apply_rome_to_model),
    # "FT": (FTHyperParams, apply_ft_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
}

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

print("Let's begin")

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    dataset_size_limit: int,
    continue_from_run: str,
    skip_generation_tests: bool,
    generation_test_interval: int,
    conserve_memory: bool,
    dataset_name: str,
    language: str,
    language_eval:str,
    device: str,
    save_delta_matrix: str,
    load_delta:str,
    add_attention: bool,
    save_attention_heads: str,
    load_attention: str,
    top_num_heads:int,
    head_ind,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    eval_on_others:bool = False,
    save_figure: str= None,
):  
    # Set algorithm-specific variables
    params_class, apply_algo = (MEMITHyperParams, apply_memit_to_model)
    # Determine run directory
    # Create new dir if not continuing from prev run OR prev run doesn't exist
    if (
        continue_from_run is None
        or not (run_dir := RESULTS_DIR / dir_name / continue_from_run).exists()
    ):
        continue_from_run = None
    if continue_from_run is None:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"

        memit_dir = run_dir / "memit_training"
        attn_dir = run_dir / "attn_training"

        memit_dir.mkdir(parents=True,exist_ok=True)
        attn_dir.mkdir(parents=True, exist_ok=True)

        for lan in ["catalan", "english"]:
            dir_lang = run_dir / "attn_training" / lan
            dir_lang.mkdir(parents=True, exist_ok=True)

        run_dir.mkdir(parents=True, exist_ok=True)
    else: 
        run_id = continue_from_run[len("run_"):]
        memit_dir = run_dir / "memit_training"
        attn_dir = run_dir / "attn_training"

    # This part of the code will allow you to don't waste extra space if you have already computed the MEMIT deltas
        
    # if language == "catalan" and language!=language_eval:
    #     memit_dir = RESULTS_DIR / "MEMIT_all"/"run_002"
    #     run_name = "run_002"
    # elif language == "english" and language!=language_eval:
    #     # memit_dir = RESULTS_DIR / "EC" /"run_000"/"memit_training"
    #     memit_dir = RESULTS_DIR / "MEMIT_all"/"run_003"
    #     run_name = "run_003"

    # elif language == language_eval and language=="catalan":
    #     memit_dir = RESULTS_DIR / "MEMIT_all"/"run_000"
    #     run_name = "run_000"

    # elif language == language_eval and language=="english":
    #     memit_dir = RESULTS_DIR / "MEMIT_all"/"run_001"
    #     run_name = "run_001"

    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    info_of_training = {"Training Language MEMIT":language, "Eval Language MEMIT - Training Att": language_eval, "# Examples": num_edits, "dataset size limit": dataset_size_limit}
    with open(run_dir/"info.json", "w") as f:
        f.write(json.dumps(info_of_training,indent= 4))

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        print(device)
        model = RWForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(torch.device(device))
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    # Load data
    print("The device of the original model is", model.device)
    print("Loading dataset, attribute snippets, tf-idf data")

    # We will not use this part due to not having translated all the dataset. 
    snips = None
    vec = None

    ds_name = "mcf"
    ds_class, ds_eval_method = DS_DICT[ds_name]
    print(dataset_size_limit)
    print(DATA_DIR, dataset_name)
    
    ds = ds_class(DATA_DIR, dataset_name, tok=tok, size=dataset_size_limit)

    # This is not optimal, but let us also put the other language of training
    if language_eval == "catalan":
        ds_eval = ds_class(DATA_DIR, "catalan_CF.json",tok=tok, size=dataset_size_limit)
        ds_other = ds_class(DATA_DIR, "english_CF.json",tok=tok, size=dataset_size_limit)
        language_other = "english"

    elif language_eval == "english":
        ds_eval = ds_class(DATA_DIR, "english_CF.json",tok=tok, size=dataset_size_limit)
        ds_other = ds_class(DATA_DIR, "catalan_CF.json",tok=tok, size=dataset_size_limit)
        language_other = "catalan"

    # if language_eval != None and language_eval == language:
    #     if language_eval == "catalan":
    #         ds_new = ds_class(DATA_DIR, "catalan_data_7102.json",tok=tok, size=dataset_size_limit)
    #     elif language_eval == "english":
    #         ds_new = ds_class(DATA_DIR, "english_filt.json",tok=tok, size=dataset_size_limit)
            
    # ds_new = ds_class(DATA_DIR, "combined_data.json", tok = tok, size = dataset_size_limit)

    # Get cache templates
    cache_template = None
    print(use_cache)
    if use_cache:
        print("I am using this")
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_vlr_{{}}_case_{{}}_{language[:3]}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    for num_c, (record_chunks, record_chunks_eval, record_chunks_other) in enumerate(chunk_both(ds, ds_eval,ds_other, n = num_edits)):
        if eval_on_others and num_c==0:
            continue
        # -------------------------- MEMIT PART --------------------------

        case_result_template = str(memit_dir / "{}_edits-case_{}.json")
        case_result_template_2 = str(attn_dir / f"{language_eval}/{{}}_edits-case_{{}}.json")
        case_result_template_3 = str(attn_dir / f"{language_other}/{{}}_edits-case_{{}}.json")

        # # Is the chunk already done?
        # already_finished = True
        # for record in record_chunks:
        #     if not Path(
        #         case_result_template.format(num_edits, record["case_id"])
        #     ).exists():
        #         already_finished = False
        #         break
        # if already_finished:
        #     continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else device))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        start = time()
        if language == "catalan":
            data_to_edit = [
                {"case_id": record["case_id"], **record["eval_target_new"]["requested_rewrite"]}
                for record in record_chunks
            ]
        elif language == "english":
            data_to_edit = [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks
            ]
        else:
            assert language not in ["catalan", "english"], "The only languages accepted are english and catalan"

        weights_copy = {}
        if load_delta == None:
            if save_delta_matrix is not None:
                save_delta_matrix_i = save_delta_matrix+ f"_iter_{num_c}"
            else:
                save_delta_matrix_i = None
            # We apply the algorithm if no precomputed delta matrix is given.
            model, weights_copy = apply_algo(
                model,
                tok,
                data_to_edit,
                hparams,
                save_delta_matrix_i,
                language,
                copy=False,
                return_orig_weights=True,
                device = device,
                **args_conserve_memory,
                **etc_args,
            )

            print("The device of the edited model is", model.device)
        elif load_delta == "original_model":
            print("Do not perform any training")
            
        elif "{}" in load_delta:
            load_delta_i = load_delta.format(num_c)
            deltas = torch.load(load_delta_i)
            with torch.no_grad():
                for w_name, (key_mat, val_mat) in deltas.items():
                    key_mat, val_mat = key_mat.to(device), val_mat.to(device)
                    upd_matrix = key_mat @ val_mat.T
                    w = nethook.get_parameter(model, w_name)
                    upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                    if w_name not in weights_copy:
                            weights_copy[w_name] = w.detach().clone() 
                    w[...] += upd_matrix.float()

        else:
            # Case in which we do not apply the algorithm, only we load the precomputed weights.
            deltas = torch.load(load_delta)
            with torch.no_grad():
                for w_name, (key_mat, val_mat) in deltas.items():
                    key_mat, val_mat = key_mat.to(device), val_mat.to(device)
                    upd_matrix = key_mat @ val_mat.T
                    w = nethook.get_parameter(model, w_name)
                    upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                    if w_name not in weights_copy:
                            weights_copy[w_name] = w.detach().clone() 
                    w[...] += upd_matrix.float()

        exec_time = time() - start
        print("Execution took", exec_time)

        # Evaluate new model
        start = time()
        gen_test_vars = [snips, vec]
        
        if language_eval is None:
            language_eval = language

        if head_ind == None:
            for idx, record in enumerate(record_chunks_eval):

                if language_eval == "catalan":
                    # We are selecting the "prompt" of the target new considering it is the same as the other.
                    # Don't worry, this will only appear in the results/MEMIT/run_id data. It does not have
                    # an effect over the evaluation
                    requested_rewrite_eval = {"relation_id":record["relation_id"],**record["eval_target_true"]["requested_rewrite"],**record["eval_target_new"]["requested_rewrite"]}

                elif language_eval == "english":
                    requested_rewrite_eval = record["requested_rewrite"]

                out_file = Path(case_result_template.format(num_edits, record["case_id"]))
                if out_file.exists():
                    print(f"Skipping {out_file}; already exists")
                    continue

                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": requested_rewrite_eval,
                    "time": exec_time,
                    "post": ds_eval_method(
                        model,
                        tok,
                        record,
                        language_eval,
                        device,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                    ),
                }

                # Dump metrics in .json
                with open(out_file, "w") as f:
                    json.dump(metrics, f, indent=1)


        # -------------------------- Attention part --------------------------
        tot_num_heads = 71
        if add_attention:
            
            case_ids = [record["case_id"] for record in record_chunks_eval]

            start = time()
            if language_eval == "catalan":
                data_to_edit = [
                    {"case_id": record["case_id"], **record["eval_target_new"]["requested_rewrite"]}
                    for record in record_chunks_eval
                ]
            elif language_eval == "english":
                data_to_edit = [
                    {"case_id": record["case_id"], **record["requested_rewrite"]}
                    for record in record_chunks_eval
                ]
            else:
                assert language_eval not in ["catalan", "english"], "The only languages accepted are english and catalan"

            if language_eval == "catalan":
                prompts, labels = tokenization_catalan(record_chunks_eval, tok)

            if language_eval == "english":
                prompts, labels = tokenization_english(record_chunks_eval, tok)

            if head_ind == None:

                all_head_wise_activations = []

                print("Getting activations")
                for prompt in tqdm(prompts):
                    _, head_wise_activations = get_falcon_activations_bau(model, prompt, device)
                    # all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
                    all_head_wise_activations.append(head_wise_activations[:,-1,:])

                values = summary(RESULTS_DIR/dir_name/f"run_{str(run_id).zfill(3)}", runs=["memit_training"],get_uncompressed=True, abs_path=True, offset=(record_chunks_eval[0]["case_id"], record_chunks_eval[-1]["case_id"]))[0]

                ids_top_accs = np.where(np.array(values["post_rewrite_acc"])==1)[0]

                val_per = 0.3  

                print("Computing top heads...")
                top_heads = determine_interventions(model, all_head_wise_activations, labels, top_num_heads, 123, ids_top_accs, val_per, False, save_figure=save_figure)                

            else:
                if "{}" in head_ind:
                    top_heads = np.load(head_ind.format(num_c))[:top_num_heads]
                else:
                    top_heads = np.load(head_ind)[:top_num_heads]
            print("The top heads are: ", top_heads)

            # # Is the chunk already done?
            # already_finished = True
            # for record in record_chunks_eval:
            #     if not Path(
            #         case_result_template_2.format(num_edits, record["case_id"])
            #     ).exists():
            #         already_finished = False
            #         break
            # if already_finished:
            #     continue

            # Compute weight changes + record weights that changed

            if load_attention:
                heads = torch.load(load_attention, map_location=device)
            else:
                print("Performing head optimization")
                heads = heads_optimization(model, tok, data_to_edit, hparams, language_eval, device, top_heads, batch_size = 32)
                if save_attention_heads:
                    if head_ind == None:
                        if language != language_eval:
                            np.save(f"./indices_iter_{num_c}_languageT_{language_eval}_numH_{top_num_heads}", top_heads)
                        else:
                            np.save(f"./indices_iter_{num_c}_languageT_{language_eval[0]+language[0]}_numH_{top_num_heads}", top_heads)
                    torch.save(heads, save_attention_heads+f"_iter_{num_c}_languageT_{language_eval}")

            # Heads addition 
            original_bias = []
            with torch.no_grad():
                for index, (head_layer, head_no) in enumerate(top_heads):
                    original_bias.append(model.transformer.h[head_layer].self_attention.dense.bias )
                    displacement = torch.zeros((tot_num_heads, int(model.config.hidden_size / tot_num_heads)))
                    displacement[head_no] = heads[index]
                    displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
                    bias_tobe = F.linear(displacement.to(torch.float32), model.transformer.h[head_layer].self_attention.dense.weight).to(device)
                    model.transformer.h[head_layer].self_attention.dense.bias = nn.parameter.Parameter(bias_tobe)

            for record in record_chunks_eval:
                if language_eval== "catalan":
                    # We are selecting the "prompt" of the target new considering it is the same as the other.
                    # Don't worry, this will only appear in the results/MEMIT/run_id data. It does not have
                    # an effect over the evaluation
                    requested_rewrite_eval = {"relation_id":record["relation_id"],**record["eval_target_true"]["requested_rewrite"],**record["eval_target_new"]["requested_rewrite"]}

                elif language_eval == "english":
                    requested_rewrite_eval = record["requested_rewrite"]

                out_file_2 = Path(case_result_template_2.format(num_edits, record["case_id"]))
                if out_file_2.exists():
                    print(f"Skipping {out_file_2}; already exists")
                    continue

                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": requested_rewrite_eval,
                    "time": exec_time,
                    "post": compute_rewrite_quality_counterfact(
                        model,
                        tok,
                        record,
                        language_eval,
                        device,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                    ),
                }

                # Dump metrics in .json
                with open(out_file_2, "w") as f:
                    json.dump(metrics, f, indent=1)

            for record in record_chunks_other:
                if language_other== "catalan":
                    # We are selecting the "prompt" of the target new considering it is the same as the other.
                    # Don't worry, this will only appear in the results/MEMIT/run_id data. It does not have
                    # an effect over the evaluation
                    requested_rewrite_eval = {"relation_id":record["relation_id"],**record["eval_target_true"]["requested_rewrite"],**record["eval_target_new"]["requested_rewrite"]}

                elif language_other == "english":
                    requested_rewrite_eval = record["requested_rewrite"]

                out_file_3 = Path(case_result_template_3.format(num_edits, record["case_id"]))
                if out_file_3.exists():
                    print(f"Skipping {out_file_3}; already exists")
                    continue

                metrics = {
                    "case_id": record["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": requested_rewrite_eval,
                    "time": exec_time,
                    "post": compute_rewrite_quality_counterfact(
                        model,
                        tok,
                        record,
                        language_other,
                        device,
                        *(
                            gen_test_vars
                            if record["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                    ),
                }

                # Dump metrics in .json
                with open(out_file_3, "w") as f:
                    json.dump(metrics, f, indent=1)

        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to(device)

            for index, (head_layer, head_no) in enumerate(top_heads):
                model.transformer.h[head_layer].self_attention.dense.bias = None

# model = RWForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(torch.device(device))
        # model = RWForCausalLM.from_pretrained(model_name).to(torch.device(device))
        print("Evaluation took", time() - start)
        


def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]

def chunk_both(*arrays,n=10):
    for i in range(0,len(arrays[0]),n):
        yield tuple(arr[i:i+n] for arr in arrays)


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME"],
        default="MEMIT",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        choices=[AGUILA_MODEL],
        default=AGUILA_MODEL,
        help="Model to edit.",
    )

    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="aguila.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=None,
        help="Truncate CounterFact to first n records.",
    )

    parser.add_argument(
        "--generation_test_interval",
        type=int,
        default=1,
        help="One generation test is performed every [flag_value] iterations. If -1, generation tests are skipped.",
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        choices=["catalan_CF.json", "english_CF.json", "data_cat_dif_subj.json", "data_eng_dif_subj.json"],
        default="english_CF.json",
        help="Name of the dataset that will be used to train the model.",
    )

    parser.add_argument(
        "--language",
        type=str,
        choices = ["catalan", "english"],
        default = "catalan",
        help="String that indicates which language is used.",
    )

    parser.add_argument(
        "--save_delta_matrix",
        type=str,
        # choices = ["./deltas_diff_subj", "./deltas", "./deltas_new_set"],
        default = None,
        help="Name that will have the delta matrices. It will also be combined with #num_edits and language",
    )
    
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )

    parser.add_argument(
        "--num_edits",
        type=int,
        default=1,
        help="Number of rewrites to perform simultaneously.",
    )

    parser.add_argument(
        "--load_delta",
        default=None,
        help="If this argument is different than None, there will be no training, the script will only update the model with the delta matrix of the path given.",
    )

    parser.add_argument(
        "--use_cache",
        dest="use_cache",
        action="store_true",
        help="Use cached k/v pairs",
    )
    parser.add_argument(
        "--set_device",
        choices=["cuda:0", "cuda:1","cpu"],
        default="cuda:0",
        help="Device to choose. Some computations will be always used in CPU.",
    )

    parser.add_argument(
        "--dir_name",
        default="MEMIT",
        help="Device to choose. Some computations will be always used in CPU.",
    )

    parser.add_argument(
        "--language_eval",
        default=None,
        help="Language used for evaluation",
    )

    parser.add_argument(
        "--load_attention",
        default=None,
        help="If some attention heads had already been computed, we can load them",
    )

    parser.add_argument(
        "--top_num_heads",
        default=5,
        type=int,
        help = "The number of heads to be changed is"
    )
    parser.add_argument(
        "--save_attention_heads",
        default=None,
        help="Path to save the attention heads computed",
    )

    parser.add_argument(
        "--add_attention",
        action="store_true",
        help="Whether or not adding attention correction"
    )

    parser.add_argument(
        "--head_ind",
        help= "List of heads positions where performing the edition",
        default=None,
    )
    
    parser.add_argument(
        "--save_figure",
        help= "Save the figures related to the top heads",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--eval_on_others",
        action="store_true",
        default=False,
        help="Evaluate the effect of adding the heads found in one training to the rest of the examples"
    )

    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.generation_test_interval,
        args.conserve_memory,
        args.dataset_name,
        args.language,
        args.language_eval,
        args.set_device,
        args.save_delta_matrix,
        args.load_delta,
        args.add_attention,
        args.save_attention_heads,
        args.load_attention,
        args.top_num_heads,
        args.head_ind,
        dir_name=args.dir_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        eval_on_others=args.eval_on_others,
        save_figure = args.save_figure,
    )
