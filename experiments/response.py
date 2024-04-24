#!/usr/bin/env python3

import argparse
import json
import logging
import os
from timeit import default_timer as timer

from transformers import AutoTokenizer
from vllm import SamplingParams

from adversarial_triggers.experiment_utils import (
    generate_experiment_id,
    get_file_name,
    log_commandline_args,
)
from adversarial_triggers.generation_utils import generate_multi_gpu
from adversarial_triggers.template import load_chat_template, load_system_message

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Generates responses using VLLM.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data is stored.",
)
parser.add_argument(
    "--name",
    action="store",
    type=str,
    default=None,
    help="If used, overrides the default name of the experiment.",
)
parser.add_argument(
    "--data_file_path",
    action="store",
    required=True,
    type=str,
    help="Path to the JSONL data file for response generation.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    required=True,
    type=str,
    help="The model to generate responses with.",
)
parser.add_argument(
    "--max_new_tokens",
    action="store",
    type=int,
    default=512,
    help="Maximum number of new tokens to generate.",
)
parser.add_argument(
    "--generation_config_file_path",
    action="store",
    type=str,
    required=True,
    help="Path to the generation configuration file. The configuration file is "
    "mapped onto the VLLM SamplingParams class.",
)
parser.add_argument(
    "--num_devices",
    action="store",
    type=int,
    default=1,
    help="Number of devices to use for generation.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=0,
    help="Seed for RNG.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    setup_time_start = timer()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    name = args.name or "response"

    dataset_name = get_file_name(args.data_file_path)
    generation_config_name = get_file_name(args.generation_config_file_path)

    experiment_id = generate_experiment_id(
        name=name,
        model_name_or_path=args.model_name_or_path,
        dataset_name=dataset_name,
        generation_config_name=generation_config_name,
        seed=args.seed,
    )

    logging.info("Response generation:")
    log_commandline_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    chat_template = load_chat_template(args.model_name_or_path)
    system_message = load_system_message(args.model_name_or_path)
    tokenizer.chat_template = chat_template

    with open(args.data_file_path, "r") as f:
        observations = [json.loads(line) for line in f]

    def format_prompt(observation):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": observation["query"]},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        return prompt

    prompts = [format_prompt(observation) for observation in observations]
    logging.info(f"Formatted {len(prompts)} prompts.")

    with open(args.generation_config_file_path, "r") as f:
        generation_config = json.load(f)

    sampling_params = SamplingParams(
        top_p=generation_config.get("top_p", 1.0),
        temperature=generation_config.get("temperature", 1.0),
        max_tokens=args.max_new_tokens,
    )

    setup_time = timer() - setup_time_start
    eval_time_start = timer()

    responses = generate_multi_gpu(
        prompts=prompts,
        model_name_or_path=args.model_name_or_path,
        sampling_params=sampling_params,
        num_devices=args.num_devices,
    )

    eval_time = timer() - eval_time_start

    durations = {
        "setup": setup_time,
        "eval": eval_time,
        "total": setup_time + eval_time,
    }

    results = {
        "args": vars(args),
        "responses": responses,
        "durations": durations,
        "dataset_name": dataset_name,
    }

    os.makedirs(f"{args.persistent_dir}/results/{name}", exist_ok=True)
    with open(f"{args.persistent_dir}/results/{name}/{experiment_id}.json", "w") as f:
        json.dump(results, f)
