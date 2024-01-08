# MEMAT: Mass-Editing Memory with Attention

Improving the edition of thousands of facts into a transformer memory at once.

## Table of Contents

- [Installation](#installation)
- [Dataset Translation](#dataset-translation)
- [Experiments and Figures](#experiments-and-figures)
- [Acknowledgments](#acknowledgments)

## Installation

We utilized Python 3.9 with library versions specified in `requirements.txt`.

To replicate our experiments, download the necessary data and covariance matrices from this [``link``](https://drive.google.com/drive/folders/1Ey11xG6KR6tgn0zdCva4Nawz9kxwIrfe?usp=sharing). Without this data, the experiments will not function correctly.

## Dataset Translation

If you download the data, you will already have the dataset `catalan_CF.json`. However, to bring a clearer view of the main procedure followed to translate from `english_CF.json` to `catalan_CF.json`, we provide the file `./Translator/Translation_FactualAssociations.ipynb`. To perform the translation you just need to copy the file `Translator` in your google colab, add `english_CF.json` and translate.

## Experiments and Figures

This section outlines the procedures for generating various figures and conducting experiments.

- Subfigures 2(e), 2(f) and 2(g): Execute the following command to generate different figures in ./figures/individual:
```
python causal_tracing.py
```

- Figure 3: Execute the following command to generate the figure in ./figures/causal_trace:
```
python -m experiments.computing_all 
```

- Figure 7. Use the command below to generate the figure in ./figures/Tuning_LR:
```
bash tuning_LR.sh
```

- Figure 8: Generate the figure in ./figures/analysis_data using:
```
python -m analysis_data.analysis  
```

To obtain each of the six radar charts mentioned in the document (located in ./radar_charts/figures), execute:
```
python -m radar_charts.scripts.create_{chart_number}
```

Note that these radar charts use compacted results found in the Appendix of our Document. To access the uncompacted data, reproduce the experiments, but we want to warn that they are resource-intensive.

After running the experiments, summarize the content that is present in `./results` using the main function of `experiments.summarize`. Refer to the provided scripts (`ITI_results.py`, `plot_tuning.py`, and `Figures_15_16_17.py`) for guidance.

For obtaining uncompacted results, use the following scripts:

- Figure 9 data: Execute:
```
bash cross_linguality.sh
```

- Figure 10 data: Execute:
```
bash bilingual_approach_section.sh
```

- Figures 12, 15, 16, 17 data: Execute:
```
bash MEMAT_hyperparam_selection.sh
```
(The results of this code are also used for Figure 18)

- Figure 13 data: Execute:
```
bash evaluate_ITI_fails.sh
```

- Figure 19 data: Execute:
```
bash bilingual_attention_correction_FIG19.sh
```

- To obtain the data from Figure 20 you need to use `--eval_on_others` and load the head positions and the head correction vectors obtained from a first block of 1000 examples using `--head_ind` and `--load_attention` . Once you run bash `MEMAT_hyperparam_selection.sh`, you will obtain the following files:
```
["./catalan_CC_heads_16_iter_0_languageT_catalan", "./catalan_CE_heads_16_iter_0_languageT_english", "./english_EC_heads_16_iter_0_languageT_catalan", "./english_EE_heads_16_iter_0_languageT_english"]
["./indices_iter_0_languageT_catalan_numH_16.npy", "./indices_iter_0_languageT_ce_numH_16.npy", "./indices_iter_0_languageT_ec_numH_16.npy","./indices_iter_0_languageT_english_numH_16.npy"]
```
What you need to do is loading each of them and use `eval_on_others`:
```
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=CC --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --use_cache --load_attention="./catalan_CC_heads_16_iter_0_languageT_catalan" head_ind="./indices_iter_0_languageT_catalan_numH_16.npy" --top_num_heads=16 --add_attention --save_figure="./figures/head_relevance/CC.pdf" --eval_on_others
```
- Figure 21 data: Execute:
```
bash random_heads.sh
```


## Acknowledgments

Our code is based on  [``MEMIT``](https://github.com/kmeng01/memit.git) and [``ITI``](https://github.com/likenneth/honest_llama.git). 

The model used is [``Aguila-7b``](https://huggingface.co/projecte-aina/aguila-7b) which is based on [``Falcon-7b``](https://huggingface.co/tiiuae/falcon-7b).
