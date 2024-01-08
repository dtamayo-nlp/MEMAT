echo "We separate the hyperparameter search to just save one plot (look at --save_figure)"

top_num_heads_i=8
# L_1=Catalan L_2=Catalan
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=CC --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --use_cache --save_attention_heads="./catalan_CC_heads_$top_num_heads_i" --top_num_heads=$top_num_heads_i --add_attention --save_figure="./figures/head_relevance/CC.pdf" > "logs_CC_${top_num_heads_i}.txt" 2>&1
	
# L_1=English L_2=English
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=EE --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=english_CF.json --language=english --language_eval=english --num_edits=1000 --use_cache --save_attention_heads="./english_EE_heads_$top_num_heads_i" --top_num_heads=$top_num_heads_i --add_attention --save_figure="./figures/head_relevance/EE.pdf" > "logs_EE_${top_num_heads_i}.txt" 2>&1

# L_1=Catalan L_2=English
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=CE --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=catalan_CF.json --language=catalan --language_eval=english --num_edits=1000 --use_cache --save_attention_heads="./english_CE_heads_$top_num_heads_i" --top_num_heads=$top_num_heads_i --add_attention --save_figure="./figures/head_relevance/CE.pdf" > "logs_CE_${top_num_heads_i}.txt" 2>&1

# L_1=English L_2=Catalan
python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=EC --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=english_CF.json --language=english --language_eval=catalan --num_edits=1000 --use_cache --save_attention_heads="./catalan_EC_heads_$top_num_heads_i" --top_num_heads=$top_num_heads_i --add_attention --save_figure="./figures/head_relevance/EC.pdf" > "logs_EC_${top_num_heads_i}.txt" 2>&1


# Define the values for alpha_i and top_num_heads_i
top_num_heads_values=(16 32 48)

# Loop over top_num_heads_i values
for top_num_heads_i in "${top_num_heads_values[@]}"; do
# Construct the Python command with the current values
	# L_1=Catalan L_2=Catalan
	python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=CC --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=catalan_CF.json --language=catalan --language_eval=catalan --num_edits=1000 --use_cache --save_attention_heads="./catalan_CC_heads_$top_num_heads_i" --top_num_heads=$top_num_heads_i --add_attention > "logs_CC_${top_num_heads_i}.txt" 2>&1
	
	# L_1=English L_2=English
	python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=EE --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=english_CF.json --language=english --language_eval=english --num_edits=1000 --use_cache --save_attention_heads="./english_EE_heads_$top_num_heads_i" --top_num_heads=$top_num_heads_i --add_attention > "logs_EE_${top_num_heads_i}.txt" 2>&1

	# L_1=Catalan L_2=English
	python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=CE --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=catalan_CF.json --language=catalan --language_eval=english --num_edits=1000 --use_cache --save_attention_heads="./english_CE_heads_$top_num_heads_i" --top_num_heads=$top_num_heads_i --add_attention > "logs_CE_${top_num_heads_i}.txt" 2>&1
	
	# L_1=English L_2=Catalan
	python -m experiments.experiments_all_in_one --alg_name=MEMIT --hparams=aguila.json --dir_name=EC --dataset_size_limit=3000 --generation_test_interval=-1 --dataset_name=english_CF.json --language=english --language_eval=catalan --num_edits=1000 --use_cache --save_attention_heads="./catalan_EC_heads_$top_num_heads_i" --top_num_heads=$top_num_heads_i --add_attention > "logs_EC_${top_num_heads_i}.txt" 2>&1
done

python Figures_15_16_17.py