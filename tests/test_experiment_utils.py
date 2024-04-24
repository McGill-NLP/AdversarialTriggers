import pytest

from adversarial_triggers.experiment_utils import (
    generate_experiment_id,
    get_short_model_name,
    parse_experiment_id,
)


@pytest.mark.parametrize(
    "experiment_parameters, expected",
    [
        (
            {"name": "test"},
            "test",
        ),
        (
            {"name": "test", "model_name_or_path": ["meta-llama/Llama-2-7b-chat-hf"]},
            "test_m-Llama-2-7b-chat-hf",
        ),
        (
            {"name": "test", "generation_config_name": "greedy"},
            "test_g-greedy",
        ),
        (
            {"name": "test", "generation_config_name": "nucleus95"},
            "test_g-nucleus95",
        ),
        (
            {"name": "test", "model_name_or_path": ["meta-llama/Llama-2-7b-chat-hf"]},
            "test_m-Llama-2-7b-chat-hf",
        ),
    ],
)
def test_generate_experiment_id(experiment_parameters, expected):
    actual_experiment_id = generate_experiment_id(**experiment_parameters)
    assert actual_experiment_id == expected


@pytest.mark.parametrize(
    "experiment_id, expected",
    [
        (
            "test",
            {"name": "test"},
        ),
        (
            "test_m-Llama-2-7b-chat-hf",
            {"name": "test", "model_name_or_path": ["Llama-2-7b-chat-hf"]},
        ),
        (
            "test_g-greedy",
            {"name": "test", "generation_config_name": "greedy"},
        ),
        (
            "test_g-nucleus95",
            {"name": "test", "generation_config_name": "nucleus95"},
        ),
        (
            "test_m-Llama-2-7b-chat-hf",
            {"name": "test", "model_name_or_path": ["Llama-2-7b-chat-hf"]},
        ),
    ],
)
def test_parse_experiment_id(experiment_id, expected):
    actual_experiment_parameters = parse_experiment_id(experiment_id)
    assert actual_experiment_parameters == expected


@pytest.mark.parametrize(
    "model_name_or_path, expected",
    [
        (
            "meta-llama/Llama-2-7b-chat-hf",
            "Llama-2-7b-chat-hf",
        ),
        (
            "Llama-2-7b-chat-hf",
            "Llama-2-7b-chat-hf",
        ),
        (
            ["lmsys/vicuna-7b-v1.5", "meta-llama/Llama-2-7b-chat-hf"],
            "vicuna-7b-v1.5--Llama-2-7b-chat-hf",
        ),
    ],
)
def test_get_short_model_name(model_name_or_path, expected):
    actual_short_model_name = get_short_model_name(model_name_or_path)
    assert actual_short_model_name == expected
