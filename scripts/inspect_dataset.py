#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random

try:
    from rich import print as rprint
except ImportError:
    raise ImportError("Please install rich to run this script.")

from adversarial_triggers.experiment_utils import log_commandline_args

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Prints examples from JSONL dataset.")
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
    help="Path to the JSONL file containing the dataset.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Dataset inspection:")
    log_commandline_args(args)

    with open(args.data_file_path, "r") as f:
        observations = [json.loads(line) for line in f]

    random.shuffle(observations)

    for i, observation in enumerate(observations):
        id_ = observation["id_"]
        query = observation["query"]
        target = observation["target"]

        rprint(f"[yellow]Observation {id_}[/yellow]")
        rprint("[white]Query:[/white]", end=" ")
        print(query)
        rprint("[white]Target:[/white]", end=" ")
        print(target)
        print()

        code = input("Continue (c) / Quit (q): ")
        if code == "q":
            break
