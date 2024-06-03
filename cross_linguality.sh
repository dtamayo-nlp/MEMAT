# Experiments of adding 1000 examples using MEMIT repeated 7 times to reduce error

python -m experiments.evaluate --alg_name=MEMIT --dir_name=MEMIT_all --hparams_fname=aguila.json --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --dataset_size_limit=7000 --use_cache --continue_from_run=run_000

python -m experiments.evaluate --alg_name=MEMIT --dir_name=MEMIT_all --hparams_fname=aguila.json --dataset_name=english_CF.json --language=english --language_eval=english --num_edits=1000 --dataset_size_limit=7000 --use_cache --continue_from_run=run_001

python -m experiments.evaluate --alg_name=MEMIT --dir_name=MEMIT_all --hparams_fname=aguila.json --dataset_name=catalan_CF.json --language=catalan --language_eval=english --num_edits=1000 --dataset_size_limit=7000 --use_cache --continue_from_run=run_002

python -m experiments.evaluate --alg_name=MEMIT --dir_name=MEMIT_all --hparams_fname=aguila.json --dataset_name=english_CF.json --language=english --language_eval=catalan --num_edits=1000 --dataset_size_limit=7000 --use_cache --continue_from_run=run_003


# Experiments 
python -m experiments.evaluate --alg_name=MEMIT --dir_name=MEMIT_diff --hparams_fname=aguila.json --dataset_name=data_cat_dif_subj.json --language=catalan --language_eval=catalan --num_edits=1000 --dataset_size_limit=1000 --use_cache --continue_from_run=run_000

python -m experiments.evaluate --alg_name=MEMIT --dir_name=MEMIT_diff --hparams_fname=aguila.json --dataset_name=data_eng_dif_subj.json --language=english --language_eval=english --num_edits=1000 --dataset_size_limit=1000 --use_cache --continue_from_run=run_001

python -m experiments.evaluate --alg_name=MEMIT --dir_name=MEMIT_diff --hparams_fname=aguila.json --dataset_name=data_cat_dif_subj.json --language=catalan --language_eval=english --num_edits=1000 --dataset_size_limit=1000 --use_cache --continue_from_run=run_002

python -m experiments.evaluate --alg_name=MEMIT --dir_name=MEMIT_diff --hparams_fname=aguila.json --dataset_name=data_eng_dif_subj.json --language=english --language_eval=catalan --num_edits=1000 --dataset_size_limit=1000 --use_cache --continue_from_run=run_003