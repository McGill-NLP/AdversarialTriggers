#!/usr/bin/env python3

import argparse
import logging
import os

from transformers import AutoTokenizer

from adversarial_triggers.experiment_utils import (
    get_short_model_name,
    log_commandline_args,
)
from adversarial_triggers.export import annotation
from adversarial_triggers.template import load_chat_template

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Creates table with chat templates.")
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
    help="The models to export chat templates for.",
)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("Creating chat template table:")
    log_commandline_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path[0])
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id

    model_family_mapping = {
        annotation.model_family.labeller(m): m
        for m in [get_short_model_name(m) for m in args.model_name_or_path]
    }

    model_families = sorted(model_family_mapping.keys())

    placeholder_messages = [
        {"role": "system", "content": r"${system_message}"},
        {"role": "user", "content": r"${user_message}"},
        {"role": "assistant", "content": r"${assistant_message}"},
    ]

    latex = (
        r"\begin{tabular}{lp{0.75\linewidth}}" + "\n"
        r"\toprule" + "\n"
        r"\textbf{Model Family} & \textbf{Chat Template} \\" + "\n"
        r"\midrule" + "\n"
    )

    def replace_special_chars(s):
        s = s.replace("<s>", "")
        s = s.replace("<bos>", "")
        s = s.replace("</s>", "")
        s = s.replace("<eos>", "")
        s = s.replace("\n", r"\n ")
        return s

    for model_family in model_families[:-1]:
        model_name = model_family_mapping[model_family]
        chat_template = load_chat_template(model_name)
        tokenizer.chat_template = chat_template

        chat = tokenizer.apply_chat_template(placeholder_messages, tokenize=False)
        chat = replace_special_chars(chat)

        chat = (
            r"\begin{minipage}[t]{\linewidth}\begin{verbatim}"
            + chat
            + r"\end{verbatim}\end{minipage}"
        )

        latex += model_family + r" & " + chat + r" \\" + "\n"
    else:
        model_family = model_families[-1]
        model_name = model_family_mapping[model_family]
        chat_template = load_chat_template(model_name)
        tokenizer.chat_template = chat_template

        chat = tokenizer.apply_chat_template(placeholder_messages, tokenize=False)
        chat = replace_special_chars(chat)

        chat = (
            r"\begin{minipage}[t]{\linewidth}\begin{verbatim}"
            + chat
            + r"\end{verbatim}\end{minipage}"
        )

        latex += model_family + r" & " + chat + r" \\" + "\n"

    latex += r"\bottomrule" + "\n"
    latex += r"\end{tabular}"

    os.makedirs(f"{args.persistent_dir}/table", exist_ok=True)
    with open(f"{args.persistent_dir}/table/chat_template.tex", "w") as f:
        f.write(latex)
