#!/bin/bash

# Define the values for alpha_i and top_num_heads_i
alpha_values=(1.0 5.0 10.0 12.0 15.0)
top_num_heads_values=(1 2 5 10)

# Loop over alpha_i values
# This is only implemented for CE
for alpha_i in "${alpha_values[@]}"; do
  # Loop over top_num_heads_i values
  for top_num_heads_i in "${top_num_heads_values[@]}"; do
    # Construct the Python command with the current values
    python -m experiments.evaluate_addITI --alg_name=MEMIT --dir_name=MEMIT_ITI --dataset_size_limit=1000 --num_edits=1000 --use_cache --alpha="$alpha_i" --top_num_heads="$top_num_heads_i" --add_attention > "logs_${alpha_i}_${top_num_heads_i}.txt" 2>&1
  done
done

python ITI_results.py