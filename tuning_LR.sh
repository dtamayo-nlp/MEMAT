python -m experiments.evaluate --alg_name=MEMIT --dir_name=Tuning_LR --hparams_fname=aguila3.json --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --dataset_size_limit=1000 --use_cache --continue_from_run=run_000

python -m experiments.evaluate --alg_name=MEMIT --dir_name=Tuning_LR --hparams_fname=aguila4.json --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --dataset_size_limit=1000 --use_cache --continue_from_run=run_001

python -m experiments.evaluate --alg_name=MEMIT --dir_name=Tuning_LR --hparams_fname=aguila.json --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --dataset_size_limit=1000 --use_cache --continue_from_run=run_002

python -m experiments.evaluate --alg_name=MEMIT --dir_name=Tuning_LR --hparams_fname=aguila2.json --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --dataset_size_limit=1000 --use_cache --continue_from_run=run_003

python plot_tuning_lr.py