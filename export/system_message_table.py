#!/usr/bin/env python3

import argparse
import logging
import os

from adversarial_triggers.experiment_utils import (
    get_short_model_name,
    log_commandline_args,
)
from adversarial_triggers.export import annotation
from adversarial_triggers.template import load_system_message

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Creates table with system messages.")
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
    required=True,
    nargs="+",
    help="The models to export system messages for.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Creating system message table:")
    log_commandline_args(args)

    model_family_mapping = {
        annotation.model_family.labeller(m): m
        for m in [get_short_model_name(m) for m in args.model_name_or_path]
    }

    model_families = sorted(model_family_mapping.keys())

    latex = (
        r"\begin{tabular}{lp{0.7\linewidth}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Model Family} & \textbf{System Message} \\" + "\n"
        r"\midrule" + "\n"
    )

    for model_family in model_families[:-1]:
        model_name = model_family_mapping[model_family]

        system_message = load_system_message(model_name)
        system_message = system_message.replace("\n", r"\newline ")

        latex += model_family + r" & \texttt{" + system_message + r"} \\" + "\n"
        latex += r"\\" + "\n"
    else:
        model_family = model_families[-1]
        model_name = model_family_mapping[model_family]

        system_message = load_system_message(model_name)
        system_message = system_message.replace("\n", r"\newline ")

        latex += model_family + r" & \texttt{" + system_message + r"} \\" + "\n"

    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}"

    os.makedirs(f"{args.persistent_dir}/table", exist_ok=True)
    with open(f"{args.persistent_dir}/table/system_message.tex", "w") as f:
        f.write(latex)
