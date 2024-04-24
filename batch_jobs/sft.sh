#!/bin/bash

source "batch_jobs/_job_script.sh"

declare -A time=(
  # ["Llama-2-7b-chat-hf"]="??:??:??"
    ["Llama-2-7b-chat-hf"]="16:00:00"
)

for model in "${!time[@]}"; do
    for dataset in "dromedary" "lima" "saferpaca20000" "sharegpt"; do
        submit_seed "${time[${model}]}" "$(job_script)" \
            deepspeed experiments/sft.py \
                --data_file_path "${PROJECT_DIR}/data/distil_m-${model}_d-${dataset}.jsonl" \
                --eval_data_file_path "${PROJECT_DIR}/data/eval_distil_m-${model}_d-${dataset}.jsonl" \
                --model_name_or_path "${PROJECT_DIR}/checkpoint/Llama-2-7b-hf" \
                --bf16 \
                --learning_rate 2e-5 \
                --num_train_epochs 1 \
                --per_device_train_batch_size 4 \
                --per_device_eval_batch_size 4 \
                --gradient_accumulation_steps 4 \
                --logging_steps 32 \
                --deepspeed \
                --deepspeed_config "${PROJECT_DIR}/config/ds_config.json" \
                --seed 0
    done
done
