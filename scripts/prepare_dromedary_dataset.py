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
parser = argparse.ArgumentParser(description="Prepares the Dromedary dataset.")
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


_DATASET_NAME = "zhiqings/dromedary-2-70b-v2"
_SPLIT = "train"


# Taken from: https://github.com/IBM/Dromedary. These are used to format
# the user input. We still use the chat template for the model as well.
_PROMPT_TEMPLATE_WITH_INPUT = "{instruction}\n\n{input_}"
_PROMPT_TEMPLATE_WITHOUT_INPUT = "{instruction}"


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Preparing Dromedary dataset:")
    log_commandline_args(args)

    dataset = load_dataset(_DATASET_NAME, split=_SPLIT)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    records = []
    for id_, observation in enumerate(dataset):
        instruction = observation["instruction"]
        input_ = observation["input"]
        target = observation["output"]

        if input_:
            query = _PROMPT_TEMPLATE_WITH_INPUT.format(
                instruction=instruction, input_=input_
            )
        else:
            query = _PROMPT_TEMPLATE_WITHOUT_INPUT.format(instruction=instruction)

        # Rough check for length. Examples will not be formatted like this during
        # training.
        num_tokens = len(tokenizer.encode(query + " " + target))
        if num_tokens > args.max_length:
            continue

        # There are many partial responses in the dataset. We only add an EOS token
        # to targets containing `\n\nUser`.
        target = target.replace("\n\n### User", tokenizer.eos_token)

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

    train_records = records[: int(len(records) * (1 - args.eval_fraction))]
    eval_records = records[int(len(records) * (1 - args.eval_fraction)) :]

    logging.info(
        f"Using {len(train_records)} examples for training and {len(eval_records)} "
        f"examples for evaluation."
    )

    os.makedirs(f"{args.persistent_dir}/data", exist_ok=True)
    with open(f"{args.persistent_dir}/data/dromedary.jsonl", "w") as f:
        f.writelines([json.dumps(record) + "\n" for record in train_records])

    with open(f"{args.persistent_dir}/data/eval_dromedary.jsonl", "w") as f:
        f.writelines([json.dumps(record) + "\n" for record in eval_records])
