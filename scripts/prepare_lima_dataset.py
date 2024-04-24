#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random

from transformers import AutoTokenizer

from adversarial_triggers.experiment_utils import log_commandline_args
from adversarial_triggers.template import load_chat_template, load_system_message

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Prepares the LIMA dataset.")
parser.add_argument(
    "--persistent_dir",
    action="store",
    type=str,
    default=os.path.realpath(os.path.join(thisdir, "..")),
    help="Directory where all persistent data is stored.",
)
parser.add_argument(
    "--data_file_path",
    action="store",
    required=True,
    type=str,
    help="Path to the JSONL file containing the raw examples.",
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
    default=4096,
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
    default=0.001,
    help="Fraction of examples to use for evaluation.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Preparing LIMA dataset:")
    log_commandline_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    system_message = load_system_message(args.model_name_or_path)
    chat_template = load_chat_template(args.model_name_or_path)
    tokenizer.chat_template = chat_template

    with open(args.data_file_path, "r") as f:
        dataset = [json.loads(line) for line in f]

    records = []
    for id_, example in enumerate(dataset):
        if len(example["conversations"]) != 2:
            # We skip multi-turn conversations.
            continue

        query = example["conversations"][0]
        target = example["conversations"][1]

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": query},
            {"role": "assistant", "content": target},
        ]

        num_tokens = len(tokenizer.apply_chat_template(messages))

        if num_tokens > args.max_length:
            continue

        records.append(
            {
                "id_": id_,
                "query": query,
                "target": target,
            }
        )

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
    with open(f"{args.persistent_dir}/data/lima.jsonl", "w") as f:
        f.writelines([json.dumps(record) + "\n" for record in train_records])

    with open(f"{args.persistent_dir}/data/eval_lima.jsonl", "w") as f:
        f.writelines([json.dumps(record) + "\n" for record in eval_records])
