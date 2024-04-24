#!/bin/bash

source "batch_jobs/_job_script.sh"

declare -A time=(
  # ["vicuna-7b-v1.5"]="??:??:??" 
    ["vicuna-7b-v1.5"]="06:00:00"
  # ["Llama-2-7b-chat-hf"]="??:??:??"
    ["Llama-2-7b-chat-hf"]="06:00:00"
)

for dataset in "behaviour" "string"; do
    for model in "${!time[@]}"; do
        for seed in 0 1 2; do
            submit_seed "${time[${model}]}" "$(job_script)" \
                python3 experiments/single.py \
                    --data_file_path "${PROJECT_DIR}/data/${dataset}.jsonl" \
                    --model_name_or_path "${PROJECT_DIR}/checkpoint/${model}" \
                    --generation_config_file_path "${PROJECT_DIR}/config/greedy.json" \
                    --num_optimization_steps 500 \
                    --num_triggers 512 \
                    --k 256 \
                    --batch_size 256 \
                    --num_trigger_tokens 20 \
                    --num_examples 25 \
                    --logging_steps 1 \
                    --hash_model_name \
                    --seed "${seed}"
        done
    done
done
