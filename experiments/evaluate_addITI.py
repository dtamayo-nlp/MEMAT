import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import argparse
from tqdm import tqdm 

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from Falcon.modelling_RW import RWForCausalLM

# from baselines.ft import FTHyperParams, apply_ft_to_model
from dsets import (
    AttributeSnippets,
    CounterFactDataset,
    MultiCounterFactDataset,
    get_tfidf_vectorizer,
)
from experiments.py.eval_utils_counterfact import compute_rewrite_quality_counterfact
from experiments.py.eval_utils_zsre import compute_rewrite_quality_zsre
from memit import MEMITHyperParams, apply_memit_to_model
from util import nethook
from util.globals import *
import numpy as np
from experiments.summarize import summary

from util.ITI import upd_matrix_match_shape, add_attention_direction_second, get_falcon_activations_bau

ALG_DICT = {
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    # "ROME": (ROMEHyperParams, apply_rome_to_model),
    # "FT": (FTHyperParams, apply_ft_to_model),
}

DS_DICT = {
    "mcf": (MultiCounterFactDataset, compute_rewrite_quality_counterfact),
    "cf": (CounterFactDataset, compute_rewrite_quality_counterfact),
}

print("Let's begin")


def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    dataset_size_limit: int,
    continue_from_run: str,
    generation_test_interval: int,
    conserve_memory: bool,
    device: str,
    save_delta_matrix: str,
    load_delta:str,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
    add_attention: bool = True,
    alpha = 2,
    top_num_heads: int = 5,
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
        attn_dir = run_dir / "ITI"

        memit_dir.mkdir(parents=True,exist_ok=True)
        attn_dir.mkdir(parents=True, exist_ok=True)


        run_dir.mkdir(parents=True, exist_ok=True)
    else: 
        run_id = continue_from_run[len("run_"):]
        memit_dir = run_dir / "memit_training"
        attn_dir = run_dir / "ITI"

    # Get run hyperparameters
    params_path = (
        run_dir / "params.json"
        if continue_from_run is not None
        else HPARAMS_DIR / alg_name / hparams_fname
    )
    hparams = params_class.from_json(params_path)
    info_of_training = {"alpha": alpha, "top_num_heads": top_num_heads}

    with open(run_dir/"alpha_and_top_k.json", "w") as f:
        f.write(json.dumps(info_of_training,indent= 4))

    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")

    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        # model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
        model = RWForCausalLM.from_pretrained(model_name).to(device)
        print("Model loaded")
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name
        model_name = model.config._name_or_path

    language = "catalan"
    language_eval = "english"
    # Load data
    print("The device of the original model is", model.device)
    print("Loading dataset, attribute snippets, tf-idf data")

    # We will not use this part due to not having translated all the dataset. 
    snips = None
    vec = None
    gen_test_vars = [snips, vec]
    ds_name = "mcf"
    ds_class, ds_eval_method = DS_DICT[ds_name]

    dataset_name = "catalan_CF.json"
    dataset_eval = "english_CF.json"
    ds = ds_class(DATA_DIR, dataset_name, tok=tok, size=dataset_size_limit)
    ds_eval = ds_class(DATA_DIR, dataset_eval, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_vlr{{}}_case_{{}}_{language[:3]}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through dataset
    for num_c, (record_chunks, record_chunks_eval) in enumerate(chunk_both(ds,ds_eval, n=num_edits)):
        # if num_c == 0:
        #     continue
        case_result_template = str(memit_dir / "{}_edits-case_{}.json")
        case_result_template_2 = str(attn_dir / f"{{}}_edits-case_{{}}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
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

        if load_delta == None:
            # We apply the algorithm if no precomputed delta matrix is given.
            model, weights_copy = apply_algo(
                model,
                tok,
                data_to_edit,
                hparams,
                save_delta_matrix,
                language,
                copy=False,
                return_orig_weights=True,
                device = device,
                **args_conserve_memory,
                **etc_args,
            )

            print("The device of the edited model is", model.device)

        elif load_delta == "original_model":
            print("We do not change the original model")

        else:
            # Case in which we do not apply the algorithm, only we load the precomputed weights.
            deltas = torch.load(load_delta)
            with torch.no_grad():
                for w_name, (key_mat, val_mat) in deltas.items():
                    key_mat, val_mat = key_mat.to(device), val_mat.to(device)
                    upd_matrix = key_mat @ val_mat.T
                    w = nethook.get_parameter(model, w_name)
                    upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)
                    w[...] += upd_matrix.float()

        exec_time = time() - start
        print("Execution took", exec_time)


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


        # Evaluate new model
        start = time()
        

        if add_attention == True:

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
            all_head_wise_activations = []

            print("Getting activations")
            for prompt in tqdm(prompts):
                _, head_wise_activations = get_falcon_activations_bau(model, prompt, device)
                # all_layer_wise_activations.append(layer_wise_activations[:,-1,:])
                all_head_wise_activations.append(head_wise_activations[:,-1,:])

            values = summary(RESULTS_DIR/dir_name/f"run_{str(run_id).zfill(3)}", runs=["memit_training"],get_uncompressed=True, abs_path=True, offset=(record_chunks_eval[0]["case_id"], record_chunks_eval[-1]["case_id"]))[0]

            ids_top_accs = np.where(np.array(values["post_rewrite_acc"])==1)[0]
            val_per = 0.3  
            all_head_wise_activations = np.array(all_head_wise_activations)
            
            model = add_attention_direction_second(
                model, 
                tok, 
                all_head_wise_activations, 
                labels,
                top_num_heads,  
                alpha, 
                device, 
                123, 
                ids_top_accs,
                val_per)


        for record in record_chunks:
            if language == "catalan":
                # We are selecting the "prompt" of the target new considering it is the same as the other.
                # Don't worry, this will only appear in the results/MEMIT/run_id data. It does not have
                # an effect over the evaluation
                requested_rewrite_eval = {"relation_id":record["relation_id"],**record["eval_target_true"]["requested_rewrite"],**record["eval_target_new"]["requested_rewrite"]}

            elif language == "english":
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
                "post": ds_eval_method(
                    model,
                    tok,
                    record,
                    language,
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


        # Restore original weights
        with torch.no_grad():
            for k, v in weights_copy.items():
                nethook.get_parameter(model, k)[...] = v.to("cuda")

        print("Evaluation took", time() - start)
        
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

def chunk_both(*arrays,n=10):
    for i in range(0,len(arrays[0]),n):
        yield tuple(arr[i:i+n] for arr in arrays)

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["MEMIT", "ROME"],
        default="MEMIT",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
    )

    parser.add_argument(
        "--dir_name",
        default="MEMIT_ITI",
        help="This sets the directory where the results will be stored"
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
        choices = ["./deltas_diff_subj", "./deltas"],
        default = "deltas",
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
        "--top_num_heads",
        type=int,
        default=40,
        help="Number of heads to use for training.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=15.,
        help="Alpha parameter (strength) of the vector used.",
    )

    parser.add_argument(
        "--add_attention",
        action = "store_true",
        default = False,
        help="If it is set to true, this script will add the mean over the heads according to the other parameters",
    )

    parser.set_defaults(conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.dataset_size_limit,
        args.continue_from_run,
        args.generation_test_interval,
        args.conserve_memory,
        args.set_device,
        args.save_delta_matrix,
        args.load_delta,
        dir_name=args.dir_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
        add_attention = args.add_attention,
        alpha = args.alpha,
        top_num_heads = args.top_num_heads,
    )
