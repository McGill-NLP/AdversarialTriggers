#!/usr/bin/env python3

import argparse
import json
import logging
import os

from adversarial_triggers.experiment_utils import get_file_name, log_commandline_args
from adversarial_triggers.export import annotation

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Creates table with dataset examples.")
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
    nargs="+",
    help="The datasets to export examples for.",
)
parser.add_argument(
    "--num_examples",
    action="store",
    type=int,
    default=3,
    help="The number of examples to export for each dataset.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Creating dataset example table:")
    log_commandline_args(args)

    latex = (
        r"\begin{tabular}{lp{0.75\linewidth}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Dataset} & \textbf{Example Instructions} \\" + "\n"
        r"\midrule" + "\n"
    )

    dataset_names = [
        get_file_name(data_file_path) for data_file_path in args.data_file_path
    ]

    pretty_dataset_names = [
        annotation.dataset[get_file_name(data_file_path)]
        for data_file_path in args.data_file_path
    ]

    # Sort datasets by pretty name.
    sorted_data_file_paths = [
        data_file_path
        for _, data_file_path in sorted(
            zip(pretty_dataset_names, args.data_file_path), key=lambda pair: pair[0]
        )
    ]

    for data_file_path in sorted_data_file_paths:
        dataset_name = get_file_name(data_file_path)
        pretty_dataset_name = annotation.dataset[dataset_name]

        with open(data_file_path, "r") as f:
            observations = [json.loads(line) for line in f]

        examples = observations[: args.num_examples]

        latex += r"\texttt{" + pretty_dataset_name + r"}"
        for example in examples:
            latex += r" & \texttt{" + example["query"] + r"} \\[0.5em]"

    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}"

    os.makedirs(f"{args.persistent_dir}/table", exist_ok=True)
    with open(f"{args.persistent_dir}/table/dataset_example.tex", "w") as f:
        f.write(latex)
