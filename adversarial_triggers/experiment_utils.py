import argparse
import hashlib
import logging
import os
import re
from typing import Any, Dict, List, Optional, Union


def generate_experiment_id(
    name: Optional[str] = None,
    model_name_or_path: Optional[Union[str, List[str]]] = None,
    teacher_model_name_or_path: Optional[Union[str, List[str]]] = None,
    dataset_name: Optional[str] = None,
    num_optimization_steps: Optional[int] = None,
    num_triggers: Optional[int] = None,
    k: Optional[int] = None,
    num_trigger_tokens: Optional[int] = None,
    learning_rate: Optional[float] = None,
    generation_config_name: Optional[str] = None,
    seed: Optional[int] = None,
    hash_model_name: bool = False,
) -> str:
    """Generates a unique experiment ID.

    Args:
        name: The name of the experiment.
        model_name_or_path: The names of the models used.
        teacher_model_name_or_path: The names of the teacher models used.
        dataset_name: The name of the dataset.
        num_optimization_steps: The number of optimization steps.
        num_triggers: The number of candidate triggers.
        k: Number of tokens selected using gradient.
        num_trigger_tokens: The number of tokens in the trigger.
        learning_rate: The learning rate.
        generation_config_name: The name of the generation configuration.
        seed: Seed for RNG.
        hash_model_name: Whether to hash the model name.

    Returns:
        The experiment ID.

    Usage:
        >>> generate_experiment_id(
        ...     name="single",
        ...     model_name_or_path="meta-llama/Llama-2-7b-chat-hf",
        ...     dataset_name="behaviour",
        ...     num_optimization_steps=128,
        ...     num_triggers=128,
        ...     k=128,
        ...     num_trigger_tokens=20,
        ...     seed=0,
        ... )
        'single_m-Llama-2-7b-chat-hf_d-behaviour_n-128_c-128_k-128_t-20_s-0'
    """
    experiment_id = name

    if isinstance(model_name_or_path, (str, list)):
        short_model_name = get_short_model_name(model_name_or_path)
        if hash_model_name:
            experiment_id += f"_m-{hash_string(short_model_name)}"
        else:
            experiment_id += f"_m-{short_model_name}"
    if isinstance(teacher_model_name_or_path, (str, list)):
        short_teacher_model_name = get_short_model_name(teacher_model_name_or_path)
        if hash_model_name:
            experiment_id += f"_i-{hash_string(short_teacher_model_name)}"
        else:
            experiment_id += f"_i-{short_teacher_model_name}"

    if isinstance(dataset_name, str):
        experiment_id += f"_d-{dataset_name}"
    if isinstance(num_optimization_steps, int):
        experiment_id += f"_n-{num_optimization_steps}"
    if isinstance(num_triggers, int):
        experiment_id += f"_c-{num_triggers}"
    if isinstance(k, int):
        experiment_id += f"_k-{k}"
    if isinstance(num_trigger_tokens, int):
        experiment_id += f"_t-{num_trigger_tokens}"
    if isinstance(learning_rate, float):
        experiment_id += f"_l-{learning_rate}"
    if isinstance(generation_config_name, str):
        experiment_id += f"_g-{generation_config_name}"
    if isinstance(seed, int):
        experiment_id += f"_s-{seed}"

    return experiment_id


def parse_experiment_id(experiment_id: str) -> Dict[str, Any]:
    """Dynamically parses an experiment ID.

    Args:
        experiment_id: An experiment ID created using `generate_experiment_id`.

    Returns:
        A dictionary containing the experiment parameters.

    Usage:
        >>> parse_experiment_id(
        ...     "single_m-Llama-2-7b-chat-hf_d-behaviour_n-128_c-128_k-128_t-20_s-0"
        ... )
        {
            "name": "single",
            "model_name_or_path": ["meta-llama/Llama-2-7b-chat-hf"],
            "dataset_name": "behaviour",
            "num_optimization_steps": 128,
            "num_triggers": 128,
            "k": 128,
            "num_trigger_tokens": 20,
            "seed": 0,
        }
    """
    parameter_to_regex = {
        "model_name_or_path": "([A-Za-z0-9-_.]+)",
        "teacher_model_name_or_path": "([A-Za-z0-9-_.]+)",
        "dataset_name": "([A-Za-z0-9-_]+)",
        "num_optimization_steps": "([0-9]+)",
        "num_triggers": "([0-9]+)",
        "k": "([0-9]+)",
        "num_trigger_tokens": "([0-9]+)",
        "learning_rate": "([eE0-9-.]+)",
        "generation_config_name": "([A-Za-z0-9-]+)",
        "seed": "([0-9]+)",
    }

    parameter_to_code = {
        "model_name_or_path": "m",
        "teacher_model_name_or_path": "i",
        "dataset_name": "d",
        "num_optimization_steps": "n",
        "num_triggers": "c",
        "k": "k",
        "num_trigger_tokens": "t",
        "learning_rate": "l",
        "generation_config_name": "g",
        "seed": "s",
    }

    parameter_to_type = {
        "model_name_or_path": str,
        "teacher_model_name_or_path": str,
        "dataset_name": str,
        "num_optimization_steps": int,
        "num_triggers": int,
        "k": int,
        "num_trigger_tokens": int,
        "learning_rate": float,
        "generation_config_name": str,
        "seed": int,
    }

    # Check which parameters are in the experiment ID. This search is currently brittle
    # and can potentially return false positives.
    parameters_to_parse = []
    for parameter, code in parameter_to_code.items():
        if re.search(f"_{code}-", experiment_id):
            parameters_to_parse.append(parameter)

    # Build the regex. The experiment name is always included.
    regex = "([A-Za-z0-9-_.]+)"
    for parameter in parameters_to_parse:
        regex += f"_{parameter_to_code[parameter]}-{parameter_to_regex[parameter]}"

    parts = re.match(regex, experiment_id).groups()

    # Cast the parameters to the correct type.
    results = {"name": parts[0]}
    for parameter, part in zip(parameters_to_parse, parts[1:]):
        results[parameter] = parameter_to_type[parameter](part)

    # Convert to list of models.
    for key in ["model_name_or_path", "teacher_model_name_or_path"]:
        if key in parameters_to_parse:
            results[key] = results[key].split("--")

    return results


def log_commandline_args(args: argparse.Namespace) -> None:
    """Log the commandline arguments."""
    for arg in vars(args):
        logging.info(f" - {arg}: {getattr(args, arg)}")


def parse_experiment_parameters_from_file_path(file_path: str) -> Dict[str, Any]:
    """Gets the experiment parameters from a file path."""
    experiment_id = get_file_name(file_path)
    return parse_experiment_id(experiment_id)


def get_file_name(file_path: str) -> str:
    """Gets file name from a file path without extension."""
    file_name = os.path.basename(file_path)
    return os.path.splitext(file_name)[0]


def get_short_model_name(model_name_or_path: Union[str, List[str]]) -> str:
    """Gets short name for model(s). If multiple models are provided, the names are
    delimited by `--`.

    Args:
        model_name_or_path: The model name or path. Either a single string or a list of
            strings.

    Returns:
        The short name for the model(s).

    Usage:
        >>> get_short_model_name("meta-llama/Llama-2-7b-chat-hf")
        "Llama-2-7b-chat-hf"
        >>> get_short_model_name(
        ...     ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"]
        ... )
        "Llama-2-7b-chat-hf--Llama-2-13b-chat-hf"
    """
    if isinstance(model_name_or_path, str):
        model_name_or_path = [model_name_or_path]

    # Strip tailing slash.
    model_name_or_path = [m.rstrip("/") for m in model_name_or_path]

    return "--".join(m.split("/")[-1] for m in model_name_or_path)


def hash_string(string: str, n: int = 12) -> str:
    """Hashes a string using SHA256.

    Args:
        string: The string to hash.

    Returns:
        The hashed string.

    Usage:
        >>> hash_string("hello world", n=8)
        "2cf24dba"
    """
    return hashlib.sha256(string.encode()).hexdigest()[:n]


def tex_format_time(secs: float) -> str:
    """Gets pretty format for time in seconds.

    Args:
        secs: The time in seconds.

    Returns:
        The pretty format for the time.

    Usage:
        >>> tex_format_time(3600)
        "01:00"
        >>> tex_format_time(3661)
        "01:01"
    """
    hh, mm = divmod(secs // 60, 60)
    return f"{int(hh):02d}:{int(mm):02d}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prints experiment ID for given arguments without project dependencies."
    )
    parser.add_argument(
        "experiment_path",
        action="store",
        type=str,
        help="Path to the experiment script to generate the experiment ID for.",
    )
    parser.add_argument(
        "--data_file_path",
        action="store",
        required=True,
        type=str,
        help="Path to the JSONL data file used in the experiment.",
    )
    parser.add_argument(
        "--model_name_or_path",
        action="store",
        required=True,
        nargs="+",
        help="The model(s) to used in the experiment.",
    )
    parser.add_argument(
        "--num_optimization_steps",
        action="store",
        type=int,
        default=None,
        help="Number of steps to perform for trigger optimization.",
    )
    parser.add_argument(
        "--num_triggers",
        action="store",
        type=int,
        default=None,
        help="Number of candidate triggers to evaluate during GCG.",
    )
    parser.add_argument(
        "--k",
        action="store",
        type=int,
        default=None,
        help="The number of candidate tokens during trigger optimization.",
    )
    parser.add_argument(
        "--num_trigger_tokens",
        action="store",
        type=int,
        default=None,
        help="The number of tokens in the trigger.",
    )
    parser.add_argument(
        "--learning_rate",
        action="store",
        type=float,
        default=None,
        help="The learning rate used in the experiment.",
    )
    parser.add_argument(
        "--generation_config_file_path",
        action="store",
        type=str,
        default=None,
        help="Path to the generation configuration file.",
    )
    parser.add_argument(
        "--hash_model_name",
        action="store_true",
        help="Whether to hash the model name. This is useful for avoid long experiment "
        "IDs.",
    )
    parser.add_argument(
        "--seed",
        action="store",
        type=int,
        default=None,
        help="Seed for RNG.",
    )

    args, _ = parser.parse_known_args()

    name = get_file_name(args.experiment_path)

    dataset_file_name = get_file_name(args.data_file_path)
    dataset_parameters = parse_experiment_id(dataset_file_name)

    teacher_model_name_or_path = dataset_parameters.get("model_name_or_path")
    dataset_name = dataset_parameters.get("dataset_name") or dataset_file_name

    # Make sure extension is not included in experiment ID.
    generation_config_name = (
        get_file_name(args.generation_config_file_path)
        if args.generation_config_file_path
        else None
    )

    experiment_id = generate_experiment_id(
        name=name,
        model_name_or_path=args.model_name_or_path,
        teacher_model_name_or_path=teacher_model_name_or_path,
        dataset_name=dataset_name,
        num_optimization_steps=args.num_optimization_steps,
        num_triggers=args.num_triggers,
        k=args.k,
        num_trigger_tokens=args.num_trigger_tokens,
        learning_rate=args.learning_rate,
        generation_config_name=generation_config_name,
        seed=args.seed,
        hash_model_name=args.hash_model_name,
    )

    print(experiment_id)
