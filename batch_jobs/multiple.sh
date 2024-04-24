#!/bin/bash

source "batch_jobs/_job_script.sh"

declare -A time=(
  # ["gemma-1.1-2b-it"]="??:??:??"
    ["gemma-1.1-2b-it"]="24:30:00"
  # ["gemma-1.1-7b-it"]="??:??:??"
    ["gemma-1.1-7b-it"]="24:30:00"
  # ["guanaco-7B-HF"]="??:??:??"
    ["guanaco-7B-HF"]="24:30:00"
  # ["guanaco-13B-HF"]="??:??:??"
    ["guanaco-13B-HF"]="24:30:00"
  # ["koala-7B-HF"]="??:??:??"
    ["koala-7B-HF"]="24:30:00"
  # ["Llama-2-7b-chat-hf"]="??:??:??"
    ["Llama-2-7b-chat-hf"]="24:30:00"
  # ["Llama-2-13b-chat-hf"]="??:??:??"
    ["Llama-2-13b-chat-hf"]="24:30:00"
  # ["openchat_3.5"]="??:??:??"
    ["openchat_3.5"]="24:30:00"
  # ["Starling-LM-7B-alpha"]="??:??:??"
    ["Starling-LM-7B-alpha"]="24:30:00"
  # ["Starling-LM-7B-beta"]="??:??:??"
    ["Starling-LM-7B-beta"]="24:30:00"
  # ["vicuna-7b-v1.5"]="??:??:??"
    ["vicuna-7b-v1.5"]="24:30:00"
  # ["vicuna-13b-v1.5"]="??:??:??"
    ["vicuna-13b-v1.5"]="24:30:00"
  # ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-dromedary_l-2e-05_s-0"]="??:??:??"
    ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-dromedary_l-2e-05_s-0"]="24:30:00"
  # ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-lima_l-2e-05_s-0"]="??:??:??"
    ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-lima_l-2e-05_s-0"]="24:30:00"
  # ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-sharegpt_l-2e-05_s-0"]="??:??:??"
    ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-sharegpt_l-2e-05_s-0"]="24:30:00"
  # ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-saferpaca2000_l-2e-05_s-0"]="??:??:??"
    ["sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-saferpaca2000_l-2e-05_s-0"]="24:30:00"
  # ["gemma-1.1-2b-it gemma-1.1-7b-it"]="??:??:??"
    ["gemma-1.1-2b-it gemma-1.1-7b-it"]="24:30:00"
  # ["Llama-2-7b-chat-hf Llama-2-13b-chat-hf"]="??:??:??"
    ["Llama-2-7b-chat-hf Llama-2-13b-chat-hf"]="24:30:00"
  # ["Starling-LM-7B-alpha Starling-LM-7B-beta"]="??:??:??"
    ["Starling-LM-7B-alpha Starling-LM-7B-beta"]="24:30:00"
  # ["vicuna-7b-v1.5 vicuna-13b-v1.5"]="??:??:??"
    ["vicuna-7b-v1.5 vicuna-13b-v1.5"]="24:30:00"
  # ["guanaco-7B-HF guanaco-13B-HF vicuna-7b-v1.5 vicuna-13b-v1.5"]="??:??:??"
    ["guanaco-7B-HF guanaco-13B-HF vicuna-7b-v1.5 vicuna-13b-v1.5"]="24:30:00"
)

for model in "${!time[@]}"; do
    for seed in 0 1 2; do
        model_name_or_path=$(join_by_with_prefix "${model}" "${PROJECT_DIR}/checkpoint")

        submit_seed "${time[${model}]}" "$(job_script)" \
            python3 experiments/multiple.py \
                --data_file_path "${PROJECT_DIR}/data/behaviour.jsonl" \
                --model_name_or_path ${model_name_or_path} \
                --generation_config_file_path "${PROJECT_DIR}/config/greedy.json" \
                --split 0 \
                --num_optimization_steps 25000 \
                --num_triggers 512 \
                --k 256 \
                --batch_size 256 \
                --num_trigger_tokens 20 \
                --num_examples 25 \
                --logging_steps 1 \
                --hash_model_name \
                --max_time 86400 \
                --seed "${seed}"
    done
done
