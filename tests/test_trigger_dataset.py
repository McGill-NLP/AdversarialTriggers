import pytest
from transformers import AutoTokenizer

from adversarial_triggers.dataset.trigger_dataset import (
    TriggerDataset,
    create_token_mask,
)
from adversarial_triggers.template.system_message import DEFAULT_SYSTEM_MESSAGE


@pytest.fixture
def tokenizer():
    # Use Llama2 tokenizer for all tests. BOS and EOS tokens are <s> and </s>,
    # respectively.
    return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)


@pytest.fixture
def observations():
    return [
        {
            "id_": 0,
            "query": "This is the first query.",
            "target": "This is the first target.",
        },
        {
            "id_": 1,
            "query": "This is the second query.",
            "target": "This is the second target.",
        },
        {
            "id_": 2,
            "query": "This is the third query.",
            "target": "This is the third target.",
        },
        {
            "id_": 3,
            "query": "This is the fourth query.",
            "target": "This is the fourth target.",
        },
    ]


@pytest.fixture
def trigger():
    return "This is a trigger"


@pytest.mark.parametrize("num_triggers", [1, 2, 5, 10])
def test_from_observations_length(tokenizer, observations, trigger, num_triggers):
    triggers = [trigger] * num_triggers

    dataset = TriggerDataset.from_observations(
        observations=observations,
        tokenizer=tokenizer,
        system_message=DEFAULT_SYSTEM_MESSAGE,
        triggers=triggers,
        include_target=True,
    )

    assert len(dataset) == (len(observations) * num_triggers)


@pytest.mark.parametrize("load_indices", [[0], [1], [2], [3], [0, 1, 2, 3]])
def test_from_observations_load_indices_length(
    tokenizer, observations, trigger, load_indices
):
    dataset = TriggerDataset.from_observations(
        observations=observations,
        tokenizer=tokenizer,
        system_message=DEFAULT_SYSTEM_MESSAGE,
        triggers=trigger,
        include_target=True,
        load_indices=load_indices,
    )

    assert len(dataset) == len(load_indices)


@pytest.mark.parametrize("load_indices", [[0], [1], [2], [3], [0, 1, 2, 3]])
def test_from_observations_load_indices_values(
    tokenizer, observations, trigger, load_indices
):
    dataset = TriggerDataset.from_observations(
        observations=observations,
        tokenizer=tokenizer,
        system_message=DEFAULT_SYSTEM_MESSAGE,
        triggers=trigger,
        include_target=True,
        load_indices=load_indices,
    )

    actual_ids = [dataset.observations[i]["id_"] for i in range(len(dataset))]
    expected_ids = [observations[i]["id_"] for i in load_indices]

    assert actual_ids == expected_ids


def test_example_shapes(tokenizer, observations, trigger):
    dataset = TriggerDataset.from_observations(
        observations=observations,
        tokenizer=tokenizer,
        system_message=DEFAULT_SYSTEM_MESSAGE,
        triggers=trigger,
        include_target=True,
    )

    assert all(
        example["input_ids"].shape == example["target_mask"].shape
        and example["input_ids"].shape == example["trigger_mask"].shape
        for example in dataset
    )


def test_create_token_mask_partial_match():
    input_ids = [1, 2, 3, 4, 5]
    search_token_ids = [2, 3]

    expected_mask = [0, 1, 1, 0, 0]
    actual_mask = create_token_mask(input_ids, search_token_ids)

    assert actual_mask == expected_mask


def test_create_token_mask_no_search_tokens():
    input_ids = [1, 2, 3, 4, 5]
    search_token_ids = []

    expected_mask = [0, 0, 0, 0, 0]
    actual_mask = create_token_mask(input_ids, search_token_ids)

    assert actual_mask == expected_mask


def test_create_token_mask_no_match():
    input_ids = [1, 2, 3, 4, 5]
    search_token_ids = [6, 7]

    expected_mask = [0, 0, 0, 0, 0]
    actual_mask = create_token_mask(input_ids, search_token_ids)

    assert actual_mask == expected_mask


def test_create_token_mask_exact_match():
    input_ids = [1, 2, 3, 4, 5]
    search_token_ids = [1, 2, 3, 4, 5]

    expected_mask = [1, 1, 1, 1, 1]
    actual_mask = create_token_mask(input_ids, search_token_ids)

    assert actual_mask == expected_mask


def test_create_token_mask_longest_match():
    # When there are multiple matches, the longest match should be masked.
    input_ids = [2, 3, 1, 2, 3, 4, 5]
    search_token_ids = [2, 3, 4]

    expected_mask = [0, 0, 0, 1, 1, 1, 0]
    actual_mask = create_token_mask(input_ids, search_token_ids)

    assert actual_mask == expected_mask


def test_create_token_mask_offset():
    # We should be able to specify an offset to start the search from.
    input_ids = [1, 2, 3, 2, 5]
    search_token_ids = [2]
    offset = 2

    expected_mask = [0, 0, 0, 1, 0]
    actual_mask = create_token_mask(input_ids, search_token_ids, offset=offset)

    assert actual_mask == expected_mask
