#!/usr/bin/env python3

import argparse
import glob
import gc
import json
import logging
import os

import torch
from transformers import AutoTokenizer, set_seed
from vllm import SamplingParams

from adversarial_triggers.dataset import (
    DataCollatorForTriggerOptimization,
    TriggerDataset,
)
from adversarial_triggers.experiment_utils import (
    generate_experiment_id,
    get_file_name,
    get_short_model_name,
    log_commandline_args,
)
from adversarial_triggers.generation_utils import generate_multi_gpu
from adversarial_triggers.metric import JailbreakContainsMetric
from adversarial_triggers.optimizer import ModelWorker
from adversarial_triggers.template import load_chat_template, load_system_message
from adversarial_triggers.trainer import TriggerTrainer

thisdir = os.path.dirname(os.path.realpath(__file__))
parser = argparse.ArgumentParser(
    description="Evaluates ASR through N steps of trigger optimization. Note, this "
    "is nearly identical to the jailbreak experiment script. The only difference is "
    "we evaluate more triggers (not just the trigger with the minimum loss)."
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
    help="Path to the JSONL data file for jailbreak evaluation.",
)
parser.add_argument(
    "--model_name_or_path",
    action="store",
    required=True,
    type=str,
    help="The model to evaluate.",
)
parser.add_argument(
    "--max_new_tokens",
    action="store",
    type=int,
    default=64,
    help="Maximum number of new tokens to generate.",
)
parser.add_argument(
    "--generation_config_file_path",
    action="store",
    type=str,
    required=True,
    help="Path to the generation configuration file.",
)
parser.add_argument(
    "--num_devices",
    action="store",
    type=int,
    default=1,
    help="Number of devices to use for generation.",
)
parser.add_argument(
    "--seed",
    action="store",
    type=int,
    default=0,
    help="Seed for RNG.",
)


# Use greedy decoding for safety classification. We only need to generate a single token
# for each response.
_LLAMA_GUARD_SAMPLING_PARAMS = SamplingParams(
    top_p=1.0, temperature=1.0, top_k=1, max_tokens=12
)

# Number of optimization steps to evaluate.
_NUM_STEPS = 500

if __name__ == "__main__":
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    set_seed(args.seed)
    name = args.name or "curve"

    torch.multiprocessing.set_start_method("spawn")
    torch.multiprocessing.set_sharing_strategy("file_system")

    dataset_name = get_file_name(args.data_file_path)
    generation_config_name = get_file_name(args.generation_config_file_path)

    experiment_id = generate_experiment_id(
        name=name,
        model_name_or_path=args.model_name_or_path,
        dataset_name=dataset_name,
        generation_config_name=generation_config_name,
        seed=args.seed,
    )

    logging.info("ASR curve evaluation:")
    log_commandline_args(args)

    jailbreak_metric = JailbreakContainsMetric()

    with open(args.data_file_path, "r") as f:
        observations = [json.loads(line) for line in f]

    trigger_result_files = glob.glob(f"{args.persistent_dir}/results/multiple/*.json")
    if len(trigger_result_files) == 0:
        raise RuntimeError(
            f"No optimized triggers found in {args.persistent_dir}/results/multiple."
        )

    result_files = []
    for trigger_result_file in trigger_result_files:
        with open(trigger_result_file, "r") as f:
            data = json.load(f)

        short_model_name = get_short_model_name(data["args"]["model_name_or_path"])
        actual_model_name = get_short_model_name(args.model_name_or_path)

        if short_model_name == actual_model_name:
            result_files.append(trigger_result_file)

    if len(result_files) == 0:
        raise RuntimeError(
            f"No optimized triggers found for model {args.model_name_or_path}."
        )

    for i, result_file in enumerate(result_files):
        logging.info(f"Creating inputs for trigger {i + 1}/{len(result_files)}.")

        with open(args.generation_config_file_path, "r") as f:
            generation_config = json.load(f)

        chat_template = load_chat_template(args.model_name_or_path)
        system_message = load_system_message(args.model_name_or_path)

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, use_fast=True
        )
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.chat_template = chat_template

        worker = ModelWorker(
            model_name_or_path=args.model_name_or_path,
            chat_template=chat_template,
            generation_config=generation_config,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map=0 if torch.cuda.is_available() else None,
            attn_implementation=None,
        )

        worker.start()
        logging.info(f"Worker started with PID {worker.process.pid}.")

        device = worker.model.device
        logging.info(f"Using device: {device}.")

        data_collator = DataCollatorForTriggerOptimization(worker.tokenizer)

        trainer = TriggerTrainer(
            workers=worker,
            tokenizer=worker.tokenizer,
            device=device,
            data_collator=data_collator,
            jailbreak_metric=jailbreak_metric,
        )

        with open(result_file, "r") as f:
            data = json.load(f)

        inputs = []
        for step, trigger in zip(
            data["history"]["step"][: _NUM_STEPS + 1],
            data["history"]["trigger"][: _NUM_STEPS + 1],
        ):
            logging.info(f"Evaluating step {step}/{_NUM_STEPS}.")

            trigger_dataset = TriggerDataset.from_observations(
                observations,
                tokenizer=worker.tokenizer,
                system_message=system_message,
                triggers=trigger,
                include_target=False,
            )

            trigger_responses = trainer.generate(
                trigger_dataset, max_new_tokens=args.max_new_tokens
            )

            for response, observation in zip(trigger_responses, observations):
                inputs.append(
                    {
                        "args": vars(args),
                        "trigger_args": data["args"],
                        "mode": "trigger",
                        "step": step,
                        "id_": observation["id_"],
                        "query": observation["query"],
                        "trigger": trigger,
                        "response": response,
                    }
                )

        worker.stop()
        worker.model.cpu()
        del worker
        gc.collect()
        torch.cuda.empty_cache()

        tokenizer = AutoTokenizer.from_pretrained(
            f"{args.persistent_dir}/checkpoint/LlamaGuard-7b",
            use_fast=True,
        )

        def format_cls_prompt(query, response):
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]

            prompt = tokenizer.apply_chat_template(messages, tokenize=False)

            return prompt

        cls_prompts = [
            format_cls_prompt(input_["query"], input_["response"]) for input_ in inputs
        ]

        cls_responses = generate_multi_gpu(
            prompts=cls_prompts,
            model_name_or_path=f"{args.persistent_dir}/checkpoint/LlamaGuard-7b",
            sampling_params=_LLAMA_GUARD_SAMPLING_PARAMS,
            num_devices=args.num_devices,
        )

        cls_scores = []
        for response in cls_responses:
            response = response.strip()
            if response.startswith("safe"):
                cls_scores.append(0)
            elif response.startswith("unsafe"):
                cls_scores.append(1)
            else:
                raise ValueError(f"Invalid CLS response: {response}")

        str_scores = jailbreak_metric([input_["response"] for input_ in inputs])

        records = []
        for input_, cls_score, str_score in zip(inputs, cls_scores, str_scores):
            records.append(
                {
                    "args": input_["args"],
                    "trigger_args": input_["trigger_args"],
                    "dataset_name": dataset_name,
                    "mode": input_["mode"],
                    "step": input_["step"],
                    "id_": input_["id_"],
                    "query": input_["query"],
                    "trigger": input_["trigger"],
                    "response": input_["response"],
                    "cls_score": cls_score,
                    "str_score": str_score,
                }
            )

        # Get name of trigger file without extension.
        trigger_file_name = os.path.basename(result_file)
        trigger_file_name = os.path.splitext(trigger_file_name)[0]

        os.makedirs(f"{args.persistent_dir}/results/{name}", exist_ok=True)
        with open(
            f"{args.persistent_dir}/results/{name}/{trigger_file_name}.jsonl", "w"
        ) as f:
            f.writelines([json.dumps(record) + "\n" for record in records])
