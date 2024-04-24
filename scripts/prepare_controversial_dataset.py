#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random

from adversarial_triggers.experiment_utils import log_commandline_args

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Prepares the Controversial dataset.")
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
    help="Path to the JSON data file containing the raw dataset.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Preparing Controversial dataset:")
    log_commandline_args(args)

    with open(args.data_file_path, "r") as f:
        dataset = json.load(f)["instructions"]

    records = []
    for query in dataset:
        records.append(
            {
                "query": query,
                "target": "",
            }
        )

    random.shuffle(records)

    # Add an autoincrement ID to each record.
    records = [{"id_": i, **record} for i, record in enumerate(records)]

    logging.info(
        f"Saving {len(records)} examples to {args.persistent_dir}/data/controversial.jsonl"
    )

    os.makedirs(f"{args.persistent_dir}/data", exist_ok=True)
    with open(f"{args.persistent_dir}/data/controversial.jsonl", "w") as f:
        f.writelines([json.dumps(record) + "\n" for record in records])
