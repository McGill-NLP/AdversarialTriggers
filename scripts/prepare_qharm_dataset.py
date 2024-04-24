#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random

from datasets import load_dataset
from transformers import AutoTokenizer

from adversarial_triggers.experiment_utils import log_commandline_args

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Prepares the Q-Harm dataset.")
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


_DATASET_NAME = "Anthropic/hh-rlhf"
_SPLIT = "train"
_DATA_DIR = "harmless-base"


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Preparing Q-Harm dataset:")
    log_commandline_args(args)

    dataset = load_dataset(_DATASET_NAME, split=_SPLIT, data_dir=_DATA_DIR)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    records = []
    for id_, observation in enumerate(dataset):
        # The text in each observation is formatted as follows:
        #   Human: <query>
        #   Assistant: <target>
        # The query and target are separated by a blank line. Both the query and target
        # can span multiple lines.
        text = observation["chosen"]
        text = text.strip()

        lines = text.split("\n\n")
        raw_query = lines[0]

        if not raw_query.startswith("Human:"):
            raise ValueError(
                f"Expected query to start with 'Human:', but got '{raw_query}'."
            )

        offset = 1
        while not lines[offset].startswith("Assistant:"):
            raw_query += "\n\n" + lines[offset]
            offset += 1

        raw_target = lines[offset]
        offset += 1
        while offset < len(lines) and not lines[offset].startswith("Human:"):
            raw_target += "\n\n" + lines[offset]
            offset += 1

        if not raw_target.startswith("Assistant:"):
            raise ValueError(
                f"Expected target to start with 'Assistant:', but got '{raw_target}'."
            )

        query = raw_query.replace("Human:", "").strip()
        target = raw_target.replace("Assistant:", "").strip()

        # Some examples in the dataset don't contain a target.
        if not query or not target:
            continue

        # Rough check for length. Examples will not be formatted like this during
        # training.
        num_tokens = len(tokenizer.encode(query + " " + target))
        if num_tokens > args.max_length:
            continue

        records.append(
            {
                "id_": id_,
                "query": query,
                "target": target,
            }
        )

    logging.info(
        f"Filtered {len(dataset) - len(records)} examples that had more than "
        f"{args.max_length} tokens."
    )

    random.shuffle(records)

    if args.num_examples is not None:
        records = records[: args.num_examples]

    os.makedirs(f"{args.persistent_dir}/data", exist_ok=True)
    with open(f"{args.persistent_dir}/data/qharm.jsonl", "w") as f:
        f.writelines([json.dumps(record) + "\n" for record in records])
