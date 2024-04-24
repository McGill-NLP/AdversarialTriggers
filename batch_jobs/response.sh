#!/bin/bash

source "batch_jobs/_job_script.sh"

declare -A time=(
  # ["Llama-2-7b-chat-hf"]="??:??:??"
    ["Llama-2-7b-chat-hf"]="16:00:00"
)

for model in "${!time[@]}"; do
    for dataset in "sharegpt"; do
        for seed in 0; do
            submit_seed "${time[${model}]}" "$(job_script)" \
                python3 experiments/response.py \
                    --data_file_path "${PROJECT_DIR}/data/${dataset}.jsonl" \
                    --model_name_or_path "${PROJECT_DIR}/checkpoint/${model}" \
                    --generation_config_file_path "${PROJECT_DIR}/config/sample.json" \
                    --max_new_tokens 1024 \
                    --seed "${seed}"
        done
    done
done
