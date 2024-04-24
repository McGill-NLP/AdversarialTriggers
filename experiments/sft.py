#!/usr/bin/env python

import argparse
import logging
import os
import random

import deepspeed
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    set_seed,
)

from adversarial_triggers.dataset import DataCollatorForSFT, TriggerDataset
from adversarial_triggers.experiment_utils import (
    generate_experiment_id,
    get_file_name,
    log_commandline_args,
    parse_experiment_id,
)
from adversarial_triggers.template import load_chat_template, load_system_message
from adversarial_triggers.trainer import SFTTrainer

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(description="Runs supervised fine-tuning.")
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
    help="If used, overrides the default name of the experiment.",
)
parser.add_argument(
    "--data_file_path",
    action="store",
    required=True,
    type=str,
    help="Path to the JSONL data file containing the training data.",
)
parser.add_argument(
    "--eval_data_file_path",
    action="store",
    required=True,
    type=str,
    help="Path to the JSONL data file containing the evaluation data.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    type=str,
    help="The model to fine-tune.",
)
parser.add_argument(
    "--bf16",
    action="store_true",
    help="Use bfloat16 precision for training.",
)
parser.add_argument(
    "--learning_rate",
    action="store",
    type=float,
    default=2e-5,
    help="Learning rate for training.",
)
parser.add_argument(
    "--lr_scheduler_type",
    action="store",
    type=str,
    default="cosine",
    help="Learning rate scheduler type.",
)
parser.add_argument(
    "--gradient_accumulation_steps",
    action="store",
    type=int,
    default=1,
    help="How many gradients to accumulate before updating the model.",
)
parser.add_argument(
    "--warmup_ratio",
    action="store",
    type=float,
    default=0.03,
    help="Warmup ratio for training.",
)
parser.add_argument(
    "--weight_decay",
    action="store",
    type=float,
    default=0.0,
    help="Weight decay for training.",
)
parser.add_argument(
    "--max_grad_norm",
    action="store",
    type=float,
    default=1.0,
    help="Maximum gradient norm for training. Larger values will be clipped.",
)
parser.add_argument(
    "--num_train_epochs",
    action="store",
    type=int,
    default=1,
    help="Number of epochs to train for.",
)
parser.add_argument(
    "--max_steps",
    action="store",
    type=int,
    default=-1,
    help="The number of training steps. Overrides num_train_epochs.",
)
parser.add_argument(
    "--per_device_train_batch_size",
    action="store",
    type=int,
    default=4,
    help="Batch size for training.",
)
parser.add_argument(
    "--per_device_eval_batch_size",
    action="store",
    type=int,
    default=4,
    help="Batch size for evaluation.",
)
parser.add_argument(
    "--dataloader_num_workers",
    action="store",
    type=int,
    default=0,
    help="Number of workers for the dataloader.",
)
parser.add_argument(
    "--save_steps",
    action="store",
    type=int,
    default=4096,
    help="How often to save checkpoints during training.",
)
parser.add_argument(
    "--eval_steps",
    action="store",
    type=int,
    default=512,
    help="How often to evaluate the model during training.",
)
parser.add_argument(
    "--logging_steps",
    action="store",
    type=int,
    default=32,
    help="How often to log results during training.",
)
parser.add_argument(
    "--resume_from_checkpoint",
    action="store_true",
    help="If a checkpoint already exists in the output directory, resume from it.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=0,
    help="Seed for RNG.",
)
parser.add_argument(
    "--local_rank",
    action="store",
    type=int,
    default=-1,
    help="Local rank for distributed training.",
)
deepspeed.add_config_arguments(parser)


if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    set_seed(args.seed)
    random.seed(args.seed)
    name = args.name or "sft"

    dataset_file_name = get_file_name(args.data_file_path)
    dataset_parameters = parse_experiment_id(dataset_file_name)

    experiment_id = generate_experiment_id(
        name=name,
        model_name_or_path=args.model_name_or_path,
        teacher_model_name_or_path=dataset_parameters["model_name_or_path"],
        dataset_name=dataset_parameters["dataset_name"],
        learning_rate=args.learning_rate,
        seed=args.seed,
    )

    logging.info("Supervised fine-tuning:")
    log_commandline_args(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    tokenizer.padding_side = "right"

    logging.info(f"Using templates for {dataset_parameters['model_name_or_path']}.")
    chat_template = load_chat_template(dataset_parameters["model_name_or_path"])
    system_message = load_system_message(dataset_parameters["model_name_or_path"])
    tokenizer.chat_template = chat_template

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="flash_attention_2" if torch.cuda.is_available() else None,
    )

    train_dataset = TriggerDataset.from_file(
        data_file_path=args.data_file_path,
        tokenizer=tokenizer,
        system_message=system_message,
        triggers="",
        include_target=True,
        include_eos_token=True,
        return_masks=False,
        progress_bar=True,
    )
    logging.info(f"Loaded {len(train_dataset)} training examples.")

    eval_dataset = TriggerDataset.from_file(
        data_file_path=args.eval_data_file_path,
        tokenizer=tokenizer,
        system_message=system_message,
        triggers="",
        include_target=True,
        include_eos_token=True,
        return_masks=False,
        progress_bar=True,
    )
    logging.info(f"Loaded {len(eval_dataset)} evaluation examples.")

    data_collator = DataCollatorForSFT(tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        args=TrainingArguments(
            bf16=args.bf16,
            deepspeed=args.deepspeed_config,
            output_dir=f"{args.persistent_dir}/checkpoint/{experiment_id}",
            logging_dir=f"{args.persistent_dir}/tensorboard/{experiment_id}",
            learning_rate=args.learning_rate,
            lr_scheduler_type=args.lr_scheduler_type,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            per_device_train_batch_size=args.per_device_train_batch_size,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            max_grad_norm=args.max_grad_norm,
            num_train_epochs=args.num_train_epochs,
            max_steps=args.max_steps,
            dataloader_num_workers=args.dataloader_num_workers,
            save_steps=args.save_steps,
            eval_steps=args.eval_steps,
            logging_steps=args.logging_steps,
            save_strategy="steps",
            evaluation_strategy="steps",
            seed=args.seed,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    tokenizer.save_pretrained(f"{args.persistent_dir}/checkpoint/{experiment_id}")
    trainer.save_model(f"{args.persistent_dir}/checkpoint/{experiment_id}")
