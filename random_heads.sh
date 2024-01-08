
# Define the values for alpha_i and top_num_heads_i
top_num_heads_i=16


# Construct the Python command with the current values
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=EC_rand --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=english_CF.json --language=english --language_eval=catalan --num_edits=1000 --use_cache --top_num_heads=$top_num_heads_i --add_attention --head_ind="./random_ind.npy" > "logs_EC_${top_num_heads_i}_rand.txt" 2>&1
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=CC_rand --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --use_cache --top_num_heads=$top_num_heads_i --add_attention --head_ind="./random_ind.npy" > "logs_CC_${top_num_heads_i}_rand.txt" 2>&1
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=CE_rand --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=catalan_CF.json --language=catalan --language_eval=english --num_edits=1000 --use_cache --top_num_heads=$top_num_heads_i --add_attention --head_ind="./random_ind.npy" > "logs_CE_${top_num_heads_i}_rand.txt" 2>&1
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=EE_rand --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=english_CF.json --language=english --language_eval=english --num_edits=1000 --use_cache --top_num_heads=$top_num_heads_i --add_attention --head_ind="./random_ind.npy" > "logs_EE_${top_num_heads_i}_rand.txt" 2>&1