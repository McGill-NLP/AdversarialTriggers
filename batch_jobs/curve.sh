#!/bin/bash

source "batch_jobs/_job_script.sh"

declare -A time=(
  # ["gemma-1.1-2b-it"]="??:??:??"
    ["gemma-1.1-2b-it"]="06:00:00"
  # ["gemma-1.1-7b-it"]="??:??:??"
    ["gemma-1.1-7b-it"]="06:00:00"
  # ["koala-7B-HF"]="??:??:??"
    ["koala-7B-HF"]="06:00:00"
  # ["Llama-2-7b-chat-hf"]="??:??:??"
    ["Llama-2-7b-chat-hf"]="06:00:00"
  # ["Llama-2-13b-chat-hf"]="??:??:??"
    ["Llama-2-13b-chat-hf"]="06:00:00"
  # ["Starling-LM-7B-alpha"]="??:??:??"
    ["Starling-LM-7B-alpha"]="06:00:00"
  # ["Starling-LM-7B-beta"]="??:??:??"
    ["Starling-LM-7B-beta"]="06:00:00"
  # ["vicuna-7b-v1.5"]="??:??:??" 
    ["vicuna-7b-v1.5"]="06:00:00"
  # ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-dromedary_l-2e-05_s-0"]="??:??:??"
    ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-dromedary_l-2e-05_s-0"]="06:00:00"
  # ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-lima_l-2e-05_s-0"]="??:??:??"
    ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-lima_l-2e-05_s-0"]="06:00:00"
  # ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-saferpaca2000_l-2e-05_s-0"]="??:??:??"
    ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-saferpaca2000_l-2e-05_s-0"]="06:00:00"
  # ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-sharegpt_l-2e-05_s-0"]="??:??:??"
    ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-sharegpt_l-2e-05_s-0"]="06:00:00"
)

for model in "${!time[@]}"; do
    for dataset in "behaviour"; do
        submit_seed "${time[${model}]}" "$(job_script)" \
            python3 experiments/curve.py \
                --data_file_path "${PROJECT_DIR}/data/${dataset}.jsonl" \
                --model_name_or_path "${PROJECT_DIR}/checkpoint/${model}" \
                --generation_config_file_path "${PROJECT_DIR}/config/greedy.json" \
                --num_devices 1
    done
done
