# To evaluate the sum of Delta_cat + Delta_eng we can use the following 

# - Adding 1000 samples 
python -m experiments.evaluate_all_bilingual_attention --alg_name=MEMIT --hparams=aguila.json --dir_name=Loss_Separated --dataset_size_limit=3000 --generation_test_interval=-1 --num_edits=1000 --use_cache

# Add attention
python -m experiments.evaluate_all_bilingual_attention --alg_name=MEMIT --hparams=aguila.json --dir_name=Bilingual_Attention --dataset_size_limit=3000 --generation_test_interval=-1 --num_edits=1000 --use_cache --top_num_heads=16 --add_attention


# To evaluate the sum of Delta_(cat+eng) we can use the following

# - Evaluate in Catalan
python -m experiments.evaluate_combination --alg_name=MEMIT --dir_name=Combined_loss --hparams_fname=aguila.json --dataset_size_limit=3000 --dataset_name=combined_data.json --language=both --language_eval=catalan --num_edits=1000 --use_cache

# - Evaluate in English
python -m experiments.evaluate_combination --alg_name=MEMIT --dir_name=Combined_loss --hparams_fname=aguila.json --dataset_size_limit=3000 --dataset_name=combined_data.json --language=both --language_eval=english --num_edits=1000 --use_cache
