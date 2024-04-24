#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random

from transformers import AutoTokenizer

from adversarial_triggers.experiment_utils import (
    generate_experiment_id,
    log_commandline_args,
    parse_experiment_parameters_from_file_path,
)
from adversarial_triggers.template import load_chat_template, load_system_message

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Prepares a distillation dataset.")
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
    help="Prefix for name of the output file. Default is 'distil'.",
)
parser.add_argument(
    "--response_file_path",
    action="store",
    required=True,
    type=str,
    help="Path to the file containing the responses.",
)
parser.add_argument(
    "--data_file_path",
    action="store",
    required=True,
    type=str,
    help="Path to the JSONL file containing the contexts to append the responses to.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    default="meta-llama/Llama-2-7b-chat-hf",
    help="Specifies the tokenizer to use for filtering long examples.",
)
parser.add_argument(
    "--max_length",
    action="store",
    type=int,
    default=2048,
    help="Maximum length of the input sequence. Sequences longer will be filtered out.",
)
parser.add_argument(
    "--num_examples",
    action="store",
    type=int,
    default=None,
    help="Number of examples to use. We take the first examples from the dataset. "
    "If None, we use the entire dataset.",
)
parser.add_argument(
    "--eval_fraction",
    action="store",
    type=float,
    default=0.01,
    help="Fraction of examples to use for evaluation.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Preparing distillation dataset:")
    log_commandline_args(args)

    name = args.name or "distil"

    experiment_parameters = parse_experiment_parameters_from_file_path(
        args.response_file_path
    )

    experiment_id = generate_experiment_id(
        name=name,
        model_name_or_path=experiment_parameters["model_name_or_path"],
        dataset_name=experiment_parameters["dataset_name"],
    )

    with open(args.response_file_path, "r") as f:
        responses = json.load(f)["responses"]

    with open(args.data_file_path, "r") as f:
        observations = [json.loads(line) for line in f]

    if len(observations) != len(responses):
        raise ValueError(
            f"Number of responses ({len(responses)}) does not match number of contexts "
            f"({len(observations)})."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    system_message = load_system_message(args.model_name_or_path)
    chat_template = load_chat_template(args.model_name_or_path)
    tokenizer.chat_template = chat_template

    records = []
    for id_, (observation, target) in enumerate(zip(observations, responses)):
        query = observation["query"]

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
            {"role": "assistant", "content": target},
        ]

        num_tokens = len(tokenizer.apply_chat_template(messages))

        if num_tokens > args.max_length:
            continue

        # Remove extra whitespace from the query and target.
        query = query.strip()
        target = target.strip()

        records.append({"id_": id_, "query": query, "target": target})

    random.shuffle(records)

    if args.num_examples is not None:
        records = records[: args.num_examples]

    train_records = records[: int(len(records) * (1 - args.eval_fraction))]
    eval_records = records[int(len(records) * (1 - args.eval_fraction)) :]

    logging.info(
        f"Using {len(train_records)} examples for training and {len(eval_records)} "
        f"examples for evaluation."
    )

    os.makedirs(f"{args.persistent_dir}/data", exist_ok=True)
    with open(f"{args.persistent_dir}/data/{experiment_id}.jsonl", "w") as f:
        f.writelines([json.dumps(record) + "\n" for record in train_records])

    with open(f"{args.persistent_dir}/data/eval_{experiment_id}.jsonl", "w") as f:
        f.writelines([json.dumps(record) + "\n" for record in eval_records])
