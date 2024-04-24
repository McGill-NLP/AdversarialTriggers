#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from rich import print as rprint
except ImportError:
    raise ImportError("Please install rich to run this script.")


from adversarial_triggers.experiment_utils import log_commandline_args
from adversarial_triggers.template import load_chat_template, load_system_message

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(
    description="Runs interactive single-turn chat with model. Used primarily for "
    "debugging."
)
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data is stored.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    required=True,
    help="The model to chat with.",
)
parser.add_argument(
    "--generation_config_file_path",
    action="store",
    type=str,
    default=None,
    help="Path to the generation configuration file.",
)
parser.add_argument(
    "--max_new_tokens",
    action="store",
    type=int,
    default=512,
    help="Maximum number of new tokens to generate.",
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

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    set_seed(args.seed)
    random.seed(args.seed)

    logging.info("Interactive chat:")
    log_commandline_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    chat_template = load_chat_template(args.model_name_or_path)
    system_message = load_system_message(args.model_name_or_path)
    tokenizer.chat_template = chat_template

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map=0 if torch.cuda.is_available() else None,
    )

    with open(args.generation_config_file_path, "r") as f:
        generation_config = json.load(f)

    unused_parameters = model.generation_config.update(**generation_config)
    if unused_parameters:
        logging.warning(f"Generation parameters {unused_parameters} are not used.")

    def format_inputs(query):
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )

        return input_ids

    while True:
        query = input("Query: ")
        if query == "exit":
            break

        input_ids = format_inputs(query).to(model.device)
        output_ids = model.generate(input_ids, max_new_tokens=args.max_new_tokens)[0]
        response = tokenizer.decode(
            output_ids[input_ids.shape[-1] :], skip_special_tokens=True
        )

        rprint("[white]Query:[/white]", end=" ")
        print(query)
        rprint("[white]Response:[/white]", end=" ")
        print(response)
        print()
