#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import os
import random

from adversarial_triggers.experiment_utils import log_commandline_args

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Prepares the String dataset.")
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
    help="Path to the CSV data file containing the raw dataset.",
)
parser.add_argument(
    "--num_examples",
    action="store",
    type=int,
    default=None,
    help="Number of examples to use. We take the first examples from the dataset. "
    "If None, we use the entire dataset.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Preparing String dataset:")
    log_commandline_args(args)

    with open(args.data_file_path, "r") as f:
        reader = csv.DictReader(f)
        dataset = list(reader)

    records = []
    for observation in dataset:
        query = ""  # Query doesn't exist for this dataset.
        target = observation["target"]

        records.append(
            {
                "query": query,
                "target": target,
            }
        )

    random.shuffle(records)

    # Add an autoincrement ID to each record.
    records = [{"id": i, **record} for i, record in enumerate(records)]

    if args.num_examples is not None:
        records = records[: args.num_examples]

    logging.info(
        f"Saving {len(records)} examples to {args.persistent_dir}/data/string.jsonl"
    )

    os.makedirs(f"{args.persistent_dir}/data", exist_ok=True)
    with open(f"{args.persistent_dir}/data/string.jsonl", "w") as f:
        f.writelines([json.dumps(record) + "\n" for record in records])
