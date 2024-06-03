# MEMAT: Mass-Editing Memory with Attention

Improving the edition of thousands of facts into a transformer memory at once.

## Table of Contents

- [Installation](#installation)
- [Experiments](#experiments-and-figures)
- [Acknowledgments](#acknowledgments)

## Installation

We utilized Python 3.9 with library versions specified in `requirements.txt`.

To replicate our experiments, download the necessary data and covariance matrices from this [``link``](https://drive.google.com/drive/folders/1Ey11xG6KR6tgn0zdCva4Nawz9kxwIrfe?usp=sharing). Without this data, the experiments will not function correctly.

## Experiments

### MEMIT
- To reproduce the training with MEMIT using different datasets, and evaluating in English and Catalan you just need to execute:
```
bash cross_linguality.sh
```
which calls the command: `python -m experiments.evaluate --alg_name=MEMIT --dir_name=MEMIT_all --hparams_fname=aguila.json --dataset_name={lang}_CF.json --language={lang} --language_eval={lang} --num_edits=1000 --dataset_size_limit=7000 --use_cache --continue_from_run=run_000` and saves the results of each experiment in `results/MEMIT_all/run_000`.

### MEMAT
- To reproduce the training with MEMAT and the attention head relevance, you just need to execute:
```
MEMAT_hyperparam_selection.sh
```

For particular experiments with MEMAT you just need to execute:
```
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name={dir_name} --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name={lang1}_CF.json --language={lang1} --language_eval={lang2} --num_edits=1000 --use_cache --save_attention_heads="./{name_id}_heads_$top_num_heads_i" --top_num_heads=$top_num_heads_i --add_attention --save_figure="./figures/head_relevance/{lang1}{lang2}.pdf"
```

After running the experiments, summarize the content that is present in `./results` using the main function of `experiments.summarize`. Refer to the provided script `Figure_9.py` for guidance.

- To obtain the data from Figure you need to use and load the head positions and the head correction vectors obtained from a first block of 1000 examples using `--head_ind` and `--load_attention` (you will also need to remove the data used for the further experiments). Once you run bash `MEMAT_hyperparam_selection.sh`, you will obtain the following files:
```
["./catalan_CC_heads_16_iter_0_languageT_catalan", "./catalan_CE_heads_16_iter_0_languageT_english", "./english_EC_heads_16_iter_0_languageT_catalan", "./english_EE_heads_16_iter_0_languageT_english"]
["./indices_iter_0_languageT_catalan_numH_16.npy", "./indices_iter_0_languageT_ce_numH_16.npy", "./indices_iter_0_languageT_ec_numH_16.npy","./indices_iter_0_languageT_english_numH_16.npy"]
```
What you need to do is loading each of them and use:
```
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=CC --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --use_cache --load_attention="./catalan_CC_heads_16_iter_0_languageT_catalan" head_ind="./indices_iter_0_languageT_catalan_numH_16.npy" --top_num_heads=16 --add_attention 
```

### Data

- To explore the tokenization of the subjects we also leave the following piece of code that generate some figures in ./figures/analysis_data:
```
python -m analysis_data.analysis  
```

### ITI experiment
- To explore how ITI fails in improving performance, execute:
```
bash evaluate_ITI_fails.sh
```

### Multilingual introduction
- To explore the different approaches to introduce both languages at the same time execute:
```
bash bilingual_approach_section.sh
```

EXTRA NOTE: If you want to avoid some extra time in computing the inverses implied in matrices $C$, you could save and load the $\Delta$ matrices using `--save_delta_matrix` and `--load_delta`, but this will imply an extra requirement of space. 

## Acknowledgments

Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git) and [``ITI``](https://github.com/likenneth/honest_llama.git). 

The model used is [``Aguila-7b``](https://huggingface.co/projecte-aina/aguila-7b) which is based on [``Falcon-7b``](https://huggingface.co/tiiuae/falcon-7b).
