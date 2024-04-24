#!/usr/bin/env python3

import argparse
import json
import logging
import os
import random
from timeit import default_timer as timer

import torch
from transformers import set_seed

from adversarial_triggers.dataset import (
    DataCollatorForTriggerOptimization,
    TriggerDataset,
    initialize_trigger,
)
from adversarial_triggers.experiment_utils import (
    generate_experiment_id,
    get_file_name,
    log_commandline_args,
)
from adversarial_triggers.metric import JailbreakContainsMetric
from adversarial_triggers.optimizer import ModelWorker
from adversarial_triggers.template import load_chat_template, load_system_message
from adversarial_triggers.trainer import TriggerTrainer

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(
    description="Runs trigger optimization for multiple targets."
)
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
    help="Path to the JSONL data file for trigger optimization.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    required=True,
    nargs="+",
    help="The model(s) to use for trigger optimization.",
)
parser.add_argument(
    "--num_optimization_steps",
    action="store",
    type=int,
    default=500,
    help="Number of steps to perform for trigger optimization.",
)
parser.add_argument(
    "--generation_config_file_path",
    action="store",
    type=str,
    required=True,
    help="Path to the generation configuration file.",
)
parser.add_argument(
    "--num_triggers",
    action="store",
    type=int,
    default=512,
    help="Number of candidate triggers to evaluate during GCG.",
)
parser.add_argument(
    "--k",
    action="store",
    type=int,
    default=256,
    help="The number of candidate tokens during trigger optimization.",
)
parser.add_argument(
    "--batch_size",
    action="store",
    type=int,
    default=32,
    help="Batch size during optimization. Note, this is not the GCG batch size.",
)
parser.add_argument(
    "--num_trigger_tokens",
    action="store",
    type=int,
    default=20,
    help="The number of tokens in the trigger.",
)
parser.add_argument(
    "--logging_steps",
    action="store",
    type=int,
    default=32,
    help="How often to log progress during trigger optimization.",
)
parser.add_argument(
    "--num_examples",
    action="store",
    type=int,
    default=25,
    help="Number of examples to use for trigger optimization.",
)
parser.add_argument(
    "--split",
    action="store",
    type=int,
    default=0,
    help="Which split to use for trigger optimization. This determines the offset "
    "in the dataset to use. Note, the dataset is already shuffled during preparation.",
)
parser.add_argument(
    "--hash_model_name",
    action="store_true",
    help="Whether to hash the model name. This is useful to avoid long experiment "
    "IDs.",
)
parser.add_argument(
    "--max_time",
    action="store",
    type=int,
    default=None,
    help="Maximum time to run the optimization in seconds.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=0,
    help="Seed for RNG.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    setup_time_start = timer()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    set_seed(args.seed)
    random.seed(args.seed)
    name = args.name or "multiple"

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    dataset_name = get_file_name(args.data_file_path)
    generation_config_name = get_file_name(args.generation_config_file_path)

    experiment_id = generate_experiment_id(
        name=name,
        model_name_or_path=args.model_name_or_path,
        dataset_name=dataset_name,
        num_optimization_steps=args.num_optimization_steps,
        num_triggers=args.num_triggers,
        k=args.k,
        num_trigger_tokens=args.num_trigger_tokens,
        generation_config_name=generation_config_name,
        seed=args.seed,
        hash_model_name=args.hash_model_name,
    )

    logging.info("Multiple target trigger optimization:")
    log_commandline_args(args)

    num_devices = torch.cuda.device_count()
    if num_devices > 0 and num_devices < len(args.model_name_or_path):
        raise ValueError(
            f"Number of devices ({num_devices}) does not match number of models "
            "({len(args.model_name_or_path)})."
        )
    logging.info(f"Using {num_devices} GPUs.")

    with open(args.generation_config_file_path, "r") as f:
        generation_config = json.load(f)

    workers, system_messages = [], []
    for i, model_name_or_path in enumerate(args.model_name_or_path):
        system_message = load_system_message(model_name_or_path)
        chat_template = load_chat_template(model_name_or_path)

        worker = ModelWorker(
            model_name_or_path=model_name_or_path,
            chat_template=chat_template,
            generation_config=generation_config,
            torch_dtype=torch.float16 if num_devices >= 1 else torch.float32,
            device_map=i if num_devices >= 1 else None,
            attn_implementation="flash_attention_2" if num_devices >= 1 else None,
        )

        system_messages.append(system_message)
        workers.append(worker)

    for worker in workers:
        worker.start()
        logging.info(f"Worker started with PID {worker.process.pid}.")

    device = workers[0].model.device
    logging.info(f"Loaded {args.model_name_or_path} models.")
    logging.info(f"Using device {device} for aggregation.")

    jailbreak_metric = JailbreakContainsMetric()

    trigger = initialize_trigger(args.num_trigger_tokens)
    logging.info(f"Initialized trigger: {trigger}")

    base_tokenizer = workers[0].tokenizer  # We use the first tokenizer for validation.
    data_collator = DataCollatorForTriggerOptimization(base_tokenizer)

    setup_time = timer() - setup_time_start
    optimize_time_start = timer()

    trainer = TriggerTrainer(
        workers=workers,
        tokenizer=base_tokenizer,
        device=device,
        data_collator=data_collator,
        jailbreak_metric=jailbreak_metric,
        num_triggers=args.num_triggers,
        k=args.k,
        logging_steps=args.logging_steps,
        max_time=args.max_time,
    )

    with open(args.data_file_path, "r") as f:
        observations = [json.loads(line) for line in f]

    load_indices = list(range(len(observations)))[
        args.num_examples * args.split : args.num_examples * (args.split + 1)
    ]

    # Load a dataset for each model. This is necessary because each model may
    # use a different system message and chat template.
    datasets = []
    for worker, system_message in zip(workers, system_messages):
        datasets.append(
            TriggerDataset.from_observations(
                observations,
                tokenizer=worker.tokenizer,
                system_message=system_message,
                triggers=trigger,
                load_indices=load_indices,
            )
        )

    history = trainer.fit(
        datasets=datasets, num_optimization_steps=args.num_optimization_steps
    )

    optimize_time = timer() - optimize_time_start
    eval_time_start = timer()

    eval_datasets = []
    for worker, system_message in zip(workers, system_messages):
        eval_datasets.append(
            TriggerDataset.from_observations(
                observations,
                tokenizer=worker.tokenizer,
                system_message=system_message,
                triggers=trigger,
                load_indices=load_indices,
                include_target=False,
            )
        )

    results = trainer.evaluate(eval_datasets)

    eval_time = timer() - eval_time_start
    total_time = timer() - setup_time_start

    for worker in workers:
        worker.stop()

    durations = {
        "setup": setup_time,
        "optimize": optimize_time,
        "eval": eval_time,
        "total": total_time,
    }

    os.makedirs(f"{args.persistent_dir}/results/{name}", exist_ok=True)
    with open(f"{args.persistent_dir}/results/{name}/{experiment_id}.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "results": results,
                "history": history,
                "durations": durations,
                "dataset_name": dataset_name,
                "load_indices": load_indices,
            },
            f,
        )
