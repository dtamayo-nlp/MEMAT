import json
import shutil
from itertools import islice
from time import time
from typing import Tuple, Union
import argparse
import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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
from util.globals import *


# THIS EVALUATION TRAINS THE MODEL ON BOTH LANGUAGES 

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
    generation_test_interval: int,
    conserve_memory: bool,
    dataset_name: str,
    language: str,
    language_eval: str,
    device: str,
    save_delta_matrix: str,
    load_delta:str,
    dir_name: str,
    num_edits: int = 1,
    use_cache: bool = False,
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
        run_dir.mkdir(parents=True, exist_ok=True)
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

    # Instantiate vanilla model
    if type(model_name) is str:
        print("Instantiating model")
        model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
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

    ds = ds_class(DATA_DIR, dataset_name, tok=tok, size=dataset_size_limit)

    # Get cache templates
    cache_template = None
    if use_cache:
        cache_template = (
            KV_DIR
            / f"{model_name.replace('/', '_')}_{alg_name}"
            / f"{ds_name}_layer_{{}}_clamp_{{}}_vlr_{{}}_case_{{}}_{language[:3]}.npz"
        )
        print(f"Will load cache from {cache_template}")

    # Iterate through datas
    for num_c, record_chunks in enumerate(chunks(ds, num_edits)):

        case_result_template = str(run_dir / "{}_edits-case_{}.json")

        # Is the chunk already done?
        already_finished = True
        for record in record_chunks:
            if not Path(
                case_result_template.format(num_edits, record["english"]["case_id"])
            ).exists():
                already_finished = False
                break
        if already_finished:
            continue

        # Compute weight changes + record weights that changed
        case_ids = [record["english"]["case_id"] for record in record_chunks]
        args_conserve_memory = (
            dict(return_orig_weights_device=("cpu" if conserve_memory else "cuda"))
            if conserve_memory
            else dict()
        )
        etc_args = dict(cache_template=cache_template) if any(alg in alg_name for alg in ["ROME", "MEMIT"]) else dict()

        start = time()

        
        data_to_edit = [
            [
                {"case_id": record["catalan"]["case_id"], **record["catalan"]["eval_target_new"]["requested_rewrite"]},
                {"case_id": record["english"]["case_id"], **record["english"]["requested_rewrite"]}
            ]
            for record in record_chunks
        ]

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
            print("Do not perform any training")

        else:
            weights_copy = {}

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
        

        # Since we want to separate the evaluation in Eng-Cat. It is only going to be performed in one language
        for record in record_chunks:
            if language_eval == "catalan":
                # We are selecting the "prompt" of the target new considering it is the same as the other.
                # Don't worry, this will only appear in the results/MEMIT/run_id data. It does not have
                # an effect over the evaluation
                requested_rewrite_eval = {"relation_id":record["catalan"]["relation_id"],**record["catalan"]["eval_target_true"]["requested_rewrite"],**record["catalan"]["eval_target_new"]["requested_rewrite"]}

            elif language_eval == "english":
                requested_rewrite_eval = record["english"]["requested_rewrite"]

            out_file = Path(case_result_template.format(num_edits, record[language_eval]["case_id"]))
            if out_file.exists():
                print(f"Skipping {out_file}; already exists")
                continue

            metrics = {
                "case_id": record[language_eval]["case_id"],
                "grouped_case_ids": case_ids,
                "num_edits": num_edits,
                "requested_rewrite": requested_rewrite_eval,
                "time": exec_time,
                "post": ds_eval_method(
                    model,
                    tok,
                    record[language_eval],
                    language_eval,
                    device,
                    *(
                        gen_test_vars
                        if record[language_eval]["case_id"] % generation_test_interval == 0
                        else [None, None]
                    ),  # Only test generation every generation_test_interval cases
                ),
            }

            # Dump metrics in .json
            with open(out_file, "w") as f:
                json.dump(metrics, f, indent=1)

        if load_delta != "original_model":
            # Restore original weights
            with torch.no_grad():
                for k, v in weights_copy.items():
                    nethook.get_parameter(model, k)[...] = v.to("cuda")

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
        choices=["combined_data.json"],
        default="combined_data.json",
        help="Name of the dataset that will be used to train the model.",
    )

    parser.add_argument(
        "--language",
        type=str,
        choices = ["both"],
        default = "both",
        help="String that indicates which language is used.",
    )

    parser.add_argument(
        "--language_eval",
        type=str,
        choices=["catalan", "english"],
        default="catalan",
        help="Language used for evaluation",
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
        "--dir_name",
        default="MEMIT",
        help="Device to choose. Some computations will be always used in CPU.",
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
        help="Device to choose. Regardless of this argument, some computations will be performed in CPU.",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.dataset_size_limit,
        args.continue_from_run,
        args.generation_test_interval,
        args.conserve_memory,
        args.dataset_name,
        args.language,
        args.language_eval,
        args.set_device,
        args.save_delta_matrix,
        args.load_delta,
        dir_name=args.dir_name,
        num_edits=args.num_edits,
        use_cache=args.use_cache,
    )