import os
from pathlib import Path

import torch
from datasets import load_dataset, Dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.globals import *
from util.nethook import Trace, set_requires_grad
from util.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally

from .tok_dataset import (
    TokenizedDataset,
    dict_to_,
    flatten_masked_batch,
    length_collation,
)

STAT_TYPES = {
    "mom2": SecondMoment,
    "mean": Mean,
    "norm_mean": NormMean,
}


def main():
    """
    Command-line utility to precompute cached stats.
    """
    import argparse

    parser = argparse.ArgumentParser(description="ROME Statistics Collector")

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--model_name", default="aguila", choices=["gpt2-xl", "EleutherAI/gpt-j-6B","aguila"])
    aa("--dataset", default="wikipedia", choices=["wikitext", "wikipedia"])
    aa("--layers", default=[17], type=lambda x: list(map(int, x.split(","))))
    aa("--to_collect", default=["mom2"], type=lambda x: x.split(","))
    aa("--sample_size", default=10000, type=lambda x: None if x == "all" else int(x))
    aa("--batch_tokens", default=None, type=lambda x: None if x == "any" else int(x))
    aa("--precision", default="float32", choices=["float64", "float32", "float16"])
    aa("--stats_dir", default=STATS_DIR)
    aa("--download", default=1, type=int, choices=[0, 1])
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name).eval().cuda()
    set_requires_grad(False, model)

    for layer_num in args.layers:
        print(
            f"Computing stats for layer {layer_num} of {args.model_name} "
            f'over {args.sample_size or "all"} samples of {args.dataset}. '
            "Note, the statistics are collected over the inputs to the second MLP layer, "
            "or equivalently the outputs of the first MLP layer."
        )
        proj_layer_name = "c_proj" if "gpt2" in args.model_name else "fc_out"
        layer_name = f"transformer.h.{layer_num}.mlp.{proj_layer_name}"

        layer_stats(
            model,
            tokenizer,
            layer_name,
            args.stats_dir,
            args.dataset,
            args.to_collect,
            sample_size=args.sample_size,
            precision=args.precision,
            batch_tokens=args.batch_tokens,
            download=args.download,
        )

def layer_stats(
    model,
    tokenizer,
    layer_name,
    stats_dir,
    ds_name,
    language,
    to_collect,
    model_name=None,
    sample_size=None,
    precision=None,
    batch_tokens=None,
    download=True,
    progress=tqdm,
    force_recompute=False,
):
    """
    Function to load or compute cached stats.
    """

    def get_ds():
        raw_ds = load_dataset(
            ds_name,
            dict(wikitext="wikitext-103-raw-v1", wikipedia="20220301.en")[ds_name],
        )
        maxlen = model.config.max_position_embeddings
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

    def get_ds_2(language ="en", home = ""):

        # If exists a dataset, it takes it. Otherwise, it reads wikipedia, and clean it according to the needs... 
        try:
            print("Trying to take dataset")
            new_dataset = load_from_disk(f"./save_{language[:2]}")
            new_dataset = new_dataset.shuffle(seed = 123).select(range(10000))
        except:
            print("Dataset doesn't exist")
            ROOT_PATH = f"INSERT PATH HERE" # We did the computations with a dataset of wikipedia stored in local. Change this if you want to do additional tests.

            if language == "en":
                DATASETS = ["english_wikipedia.txt"]
            elif language == "ca":
                DATASETS = ["catalan_wikipedia.txt"]

            SEPARATOR = ["\n"]

            new_dataset = {}
            
            indices = []
            texts = []

            for idx, dataset in enumerate(DATASETS):
                dataset_name = "wikipedia"
                print(f'Processing {dataset_name}.\n')
                os.makedirs(f'splits/{dataset_name}', exist_ok=True)

                index = 0
                chunk_size = 1024**3

                with open(os.path.join(ROOT_PATH, dataset), 'r') as file:
                    while True:

                        fr = file.read(chunk_size)
                        if not fr:
                            break  
                        file_split = fr.split(f'{SEPARATOR[idx]}\n')
                        
                        for doc_content in tqdm(file_split):
                            index+=1
                            if doc_content == "Texto no disponible. Consulte el documento PDF de esta disposición.\n" or doc_content == "[('Código Universitario de\\nDerecho del Trabajo', {})]\n" or doc_content == "":
                                print(f'Documento {index} estaba vacio.')
                            else:
                                texts.append(doc_content)
                                indices.append(index)
                                # I could separate this by title also, but it does not really matter

            new_dataset = Dataset.from_dict({"text":texts,"indices":indices})
            new_dataset.save_to_disk(f"./save_{language}")
        maxlen = model.config.max_position_embeddings
        if batch_tokens is not None and batch_tokens < maxlen:
            maxlen = batch_tokens
        return TokenizedDataset(new_dataset, tokenizer, maxlen=1000)

    # Continue with computation of statistics
    batch_size = 1  # Examine this many dataset texts at once
    npos = model.config.max_position_embeddings
    if batch_tokens is None:
        batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
    if precision is None:
        precision = "float64"
    dtype = getattr(torch, precision)
    size_suffix = "" if sample_size is None else f"_{sample_size}"
    if batch_tokens < npos:
        size_suffix = "_t{batch_tokens}" + size_suffix
    if model_name is None:
        model_name = model.config._name_or_path.replace("/", "_")

    stats_dir = Path(stats_dir)
    if language[:2] == "en":
        file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
    else:
        file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}_cat.npz"
    filename = stats_dir / file_extension

    ds = get_ds_2(language=language[:2]) if not filename.exists() else None
    if progress is None:
        progress = lambda x: x

    stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
    loader = tally(
        stat,
        ds,
        cache=(filename if not force_recompute else None),
        sample_size=sample_size,
        batch_size=batch_size,
        collate_fn=length_collation(batch_tokens),
        pin_memory=True,
        random_sample=1,
        num_workers=2,
    )
    batch_count = -(-(sample_size or len(ds)) // batch_size)
    print("Computing the Covariance matrix")

    # You may need to change this
    device = "cuda:0"
    # else:
    #     device = "cpu"
    with torch.no_grad():
        for batch_group in progress(loader, total=batch_count):
            for batch in batch_group:
                batch = dict_to_(batch,device)
                with Trace(
                    model, layer_name, retain_input=True, retain_output=False, stop=True
                ) as tr:
                    model(**batch)
                feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                feats = feats.to(dtype=dtype)
                stat.add(feats)
    return stat


if __name__ == "__main__":
    main()
