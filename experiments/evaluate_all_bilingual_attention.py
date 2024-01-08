import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import random
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

from attention_translation.bilingual_multihead_optimization import heads_optimization

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
    data_format: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    dataset_size_limit: int,
    continue_from_run: str,
    generation_test_interval: int,
    conserve_memory: bool,
    device: str,
    save_delta_matrix: str,
    load_delta:str,
    add_attention: bool,
    save_attention_heads: str,
    load_attention: str,
    top_num_heads:int,
    dir_name: str = "Bilingual_Attention",
    num_edits: int = 1,
    use_cache: bool = False,
    eval_on_others: bool = False,
):  
    print(AGUILA_MODEL)
    print(device)
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
        if add_attention:
            attn_dir.mkdir(parents=True, exist_ok=True)


        run_dir.mkdir(parents=True, exist_ok=True)

        for lan in ["catalan", "english"]:
            if add_attention:
                dir_lang = run_dir / "attn_training" / lan
                dir_lang.mkdir(parents=True, exist_ok=True)

            dir_lang = run_dir / "memit_training" / lan
            dir_lang.mkdir(parents=True, exist_ok=True)

    else: 
        run_id = continue_from_run[len("run_"):]
        memit_dir = run_dir / "memit_training"
        if add_attention:
            attn_dir = run_dir / "attn_training"
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

    info_of_training = {"Training Language MEMIT":"both", "Eval Language MEMIT - Training Att": "both", "# Examples": num_edits, "dataset size limit": dataset_size_limit}
    with open(run_dir/"info.json", "w") as f:
        f.write(json.dumps(info_of_training,indent= 4))

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = RWForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        print(device)
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
    
    # This is not optimal, but let us also put the other language of training
    if data_format == "no restriction":
        ds_cat = ds_class(DATA_DIR, "catalan_CF.json",tok=tok, size=dataset_size_limit)
        ds_eng = ds_class(DATA_DIR, "english_CF.json",tok=tok, size=dataset_size_limit)

    else:
        ds_cat = ds_class(DATA_DIR, "data_cat_dif_subj.json",tok=tok, size=dataset_size_limit)
        ds_eng = ds_class(DATA_DIR, "data_eng_dif_subj.json",tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template_cat = None
    cache_template_eng = None
    
    if use_cache:

        cache_template_cat = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_vlr_{{}}_case_{{}}_cat.npz"
        )
        print(f"Will load cache from {cache_template_cat}")

        cache_template_eng = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_vlr_{{}}_case_{{}}_eng.npz"
        )
        print(f"Will load cache from {cache_template_eng}")

        
    # Iterate through dataset
    for num_c, (record_chunks_cat, record_chunks_eng) in enumerate(chunk_both(ds_cat,ds_eng, n = num_edits)):

        # -------------------------- MEMIT PART --------------------------

        case_result_template_cat = str(memit_dir /"catalan" /"{}_edits-case_{}.json")
        case_result_template_eng = str(memit_dir /"english" /"{}_edits-case_{}.json")

        if add_attention:
            case_result_template_attn_cat = str(attn_dir / "catalan" /f"{{}}_edits-case_{{}}.json")
            case_result_template_attn_eng = str(attn_dir / "english" / f"{{}}_edits-case_{{}}.json")

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks_cat]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )

        etc_args = dict(cache_template=[cache_template_cat,cache_template_eng]) 

        start = time()

        data_to_edit_cat = [
                {"case_id": record["case_id"], **record["eval_target_new"]["requested_rewrite"]}
                for record in record_chunks_cat
            ]
        
        data_to_edit_eng = [
                {"case_id": record["case_id"], **record["requested_rewrite"]}
                for record in record_chunks_eng
            ]

        weights_copy = {}
        if load_delta == None:
            
            # Compute deltas in the first language:

            if save_delta_matrix is not None:
                save_delta_matrix_i = save_delta_matrix+ f"_iter_{num_c}"
            else:
                save_delta_matrix_i = None

            # We apply the algorithm if no precomputed delta matrix is given.
            
            model, weights_copy = apply_algo(
                model,
                tok,
                [data_to_edit_cat,data_to_edit_eng],
                hparams,
                save_delta_matrix_i,
                "both_separated",
                copy=False,
                return_orig_weights=True,
                device = device,
                **args_conserve_memory,
                **etc_args,
            )

            print("The device of the edited model is", model.device)
        elif load_delta == "original_model":
            print("We do not perform any training")
            
        elif load_delta:

            cat = torch.load(load_delta+f"_iter_{num_c}_{len(data_to_edit_cat)}_catalan")
            eng = torch.load(load_delta+f"_iter_{num_c}_{len(data_to_edit_eng)}_english")

            with torch.no_grad():
                for w_name in cat.keys():
                    key_cat, val_cat = cat[w_name]
                    key_eng, val_eng = eng[w_name]
                    upd_matrix_cat = key_cat.to(device) @ val_cat.T.to(device)
                    upd_matrix_eng = key_eng.to(device) @ val_eng.T.to(device)
                    w = nethook.get_parameter(model, w_name)
                    upd_matrix_cat = upd_matrix_match_shape(upd_matrix_cat, w.shape)
                    upd_matrix_eng = upd_matrix_match_shape(upd_matrix_eng, w.shape)

                    if w_name not in weights_copy:
                            weights_copy[w_name] = w.detach().clone() 
                    w[...] += upd_matrix_cat.float() + upd_matrix_eng.float()

        exec_time = time() - start
        print("Execution took", exec_time)

        # Evaluate new model
        start = time()
        gen_test_vars = [snips, vec]

        
        for idx, record_cat in enumerate(record_chunks_cat):
            record_eng = record_chunks_eng[idx]
            requested_rewrite_cat = {"relation_id":record_cat["relation_id"],**record_cat["eval_target_true"]["requested_rewrite"],**record_cat["eval_target_new"]["requested_rewrite"]}
            requested_rewrite_eng = record_eng["requested_rewrite"]

            out_file_cat = Path(case_result_template_cat.format(num_edits, record_cat["case_id"]))
            out_file_eng = Path(case_result_template_eng.format(num_edits, record_cat["case_id"]))

            if out_file_cat.exists() and out_file_eng.exists():
                print(f"Skipping {out_file_cat}; already exists (same in English)")
                continue
            
            metrics_cat = {
                "case_id": record_cat["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": requested_rewrite_cat,
                "time": exec_time,
                "post": ds_eval_method(
                    model,
                    tok,
                    record_cat,
                    "catalan",
                    device,
                    *(
                        gen_test_vars
                        if record_cat["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                ),
            }

            # Dump metrics in .json
            with open(out_file_cat, "w") as f:
                json.dump(metrics_cat, f, indent=1)
                    
            metrics_eng = {
                "case_id": record_eng["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": requested_rewrite_eng,
                "time": exec_time,
                "post": ds_eval_method(
                    model,
                    tok,
                    record_eng,
                    "english",
                    device,
                    *(
                        gen_test_vars
                        if record_eng["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                ),
            }

            # Dump metrics in .json
            with open(out_file_eng, "w") as f:
                json.dump(metrics_eng, f, indent=1)

        # -------------------------- Attention part --------------------------

        if add_attention:

            case_ids = [record_cat["case_id"] for record_cat in record_chunks_cat] #It doesn't matter the language, both are 

            start = time()
            data_to_edit_cat = [
                    {"case_id": record_cat["case_id"], **record_cat["eval_target_new"]["requested_rewrite"]}
                    for record_cat in record_chunks_cat
                ]
        
            data_to_edit_eng = [
                    {"case_id": record_eng["case_id"], **record_eng["requested_rewrite"]}
                    for record_eng in record_chunks_eng
                ]
            
            data_to_edit = data_to_edit_cat + data_to_edit_eng

            # This part is only to compute WHERE to edit the information
            random.seed(123)
            random.shuffle(data_to_edit)
            print(data_to_edit[0:10])

            prompts_cat, labels_cat = tokenization_catalan(record_chunks_cat, tok)
            prompts_eng, labels_eng = tokenization_english(record_chunks_eng, tok)

            prompts = prompts_cat+prompts_eng
            labels = labels_cat+labels_eng

            all_head_wise_activations = []

            print("Getting activations")
            for prompt in tqdm(prompts):
                _, head_wise_activations = get_falcon_activations_bau(model, prompt, device)
                # all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
                all_head_wise_activations.append(head_wise_activations[:,-1,:])

            values = summary(RESULTS_DIR/dir_name/f"run_{str(run_id).zfill(3)}"/"memit_training", runs=["catalan"],get_uncompressed=True, abs_path=True, offset=(record_chunks_cat[0]["case_id"], record_chunks_cat[-1]["case_id"]))[0]
            values_2 = summary(RESULTS_DIR/dir_name/f"run_{str(run_id).zfill(3)}"/"memit_training", runs=["english"],get_uncompressed=True, abs_path=True, offset=(record_chunks_eng[0]["case_id"], record_chunks_eng[-1]["case_id"]))[0]
            ids_top_accs = np.concatenate((np.where(np.array(values["post_rewrite_acc"])==1)[0],np.where(np.array(values_2["post_rewrite_acc"])==1)[0]+num_edits))
            
            val_per = 0.3  
            tot_num_heads = 71

            print("Computing top heads...")

            top_heads = determine_interventions(model, all_head_wise_activations, labels, top_num_heads, 123, ids_top_accs, val_per, False)

            print("The top heads are: ", top_heads)

            # This part is where we find the directions to EDIT the knowledge
            if load_attention:
                heads = torch.load(load_attention)
            else:
                heads = heads_optimization(model, tok, data_to_edit, hparams, "both", device, top_heads, batch_size = 32)
                if save_attention_heads:
                    np.save(f"./indices_iter_{num_c}_languageT_both",top_heads)
                    torch.save(heads, save_attention_heads+f"_iter_{num_c}_languageT_both_{top_num_heads}")

            # Heads training 
            with torch.no_grad():
                for index, (head_layer, head_no) in enumerate(top_heads):
                    # head_layer = pair[0]
                    # head_no = pair[1]
                    displacement = torch.zeros((tot_num_heads, int(model.config.hidden_size / tot_num_heads)))
                    displacement[head_no] = heads[index]
                    displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
                    bias_tobe = F.linear(displacement.to(torch.float32), model.transformer.h[head_layer].self_attention.dense.weight).to(device)
                    model.transformer.h[head_layer].self_attention.dense.bias = nn.parameter.Parameter(bias_tobe)

            for record_cat in record_chunks_cat:

                requested_rewrite_cat = {"relation_id":record_cat["relation_id"],**record_cat["eval_target_true"]["requested_rewrite"],**record_cat["eval_target_new"]["requested_rewrite"]}

                out_file_attn_cat = Path(case_result_template_attn_cat.format(num_edits, record_cat["case_id"]))
                if out_file_attn_cat.exists():
                    print(f"Skipping {out_file_attn_cat}; already exists")
                    continue

                metrics = {
                    "case_id": record_cat["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": requested_rewrite_cat,
                    "time": exec_time,
                    "post": compute_rewrite_quality_counterfact(
                        model,
                        tok,
                        record_cat,
                        "catalan",
                        device,
                        *(
                            gen_test_vars
                            if record_cat["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                    ),
                }

                # Dump metrics in .json
                with open(out_file_attn_cat, "w") as f:
                    json.dump(metrics, f, indent=1)

                

            for record_eng in record_chunks_eng:

                requested_rewrite_eng = record_eng["requested_rewrite"]

                out_file_attn_eng = Path(case_result_template_attn_eng.format(num_edits, record_eng["case_id"]))
                if out_file_attn_eng.exists():
                    print(f"Skipping {out_file_attn_eng}; already exists")
                    continue

                metrics = {
                    "case_id": record_eng["case_id"],
                    "grouped_case_ids": case_ids,
                    "num_edits": num_edits,
                    "requested_rewrite": requested_rewrite_eng,
                    "time": exec_time,
                    "post": compute_rewrite_quality_counterfact(
                        model,
                        tok,
                        record_eng,
                        "english",
                        device,
                        *(
                            gen_test_vars
                            if record_eng["case_id"] % generation_test_interval == 0
                            else [None, None]
                        ),  # Only test generation every generation_test_interval cases
                    ),
                }

                # Dump metrics in .json
                with open(out_file_attn_eng, "w") as f:
                    json.dump(metrics, f, indent=1)

        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")
            if add_attention:
                for index, (head_layer, head_no) in enumerate(top_heads):
                    model.transformer.h[head_layer].self_attention.dense.bias = None

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

        # print(prompt_new, target_new)
        sentence_new = tokenizer(prompt_new + " " + target_new,return_tensors="pt")["input_ids"]
        sentence_old = tokenizer(prompt_old + " " + target_old,return_tensors="pt")["input_ids"]

        all_prompts += [sentence_new, sentence_old]
        all_labels += [1,0]
    
    return all_prompts, all_labels


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT"],
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
        "--data_format",
        choices = ["diff_subj", "no restriction"],
        default="no restriction"
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
        help="If this argument is 'original_model', there will be no training. If it is not used"
        "the script will compute both deltas, otherwise this argument allows to load pre-computed deltas.",
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
        "--eval_on_others",
        action="store_true",
        help="Skip first training"
    )


    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.data_format,
        str(args.model_name),
        args.hparams_fname,
        args.dataset_size_limit,
        args.continue_from_run,
        args.generation_test_interval,
        args.conserve_memory,
        args.set_device,
        args.save_delta_matrix,
        args.load_delta,
        args.add_attention,
        args.save_attention_heads,
        args.load_attention,
        args.top_num_heads,
        dir_name = args.dir_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        eval_on_others=args.eval_on_others
    )
