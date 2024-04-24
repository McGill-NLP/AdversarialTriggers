import difflib
import json
import logging
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer


class TriggerDataset(Dataset):
    """Dataset containing examples for trigger optimization and evaluation."""

    def __init__(
        self,
        data: List[Dict[str, torch.Tensor]],
        observations: List[Dict[str, Union[str, int]]],
        tokenizer: PreTrainedTokenizer,
        system_message: str,
        triggers: Union[str, List[str]],
        include_target: bool = True,
        include_eos_token: bool = False,
        return_masks: bool = True,
        load_indices: Optional[List[int]] = None,
        progress_bar: bool = False,
    ):
        self.data = data
        self.observations = observations
        self.tokenizer = tokenizer
        self.system_message = system_message
        self.triggers = triggers
        self.include_target = include_target
        self.include_eos_token = include_eos_token
        self.return_masks = return_masks
        self.load_indices = load_indices

    @classmethod
    def from_file(
        cls,
        data_file_path: str,
        tokenizer: PreTrainedTokenizer,
        system_message: str,
        triggers: Union[str, List[str]],
        include_target: bool = True,
        include_eos_token: bool = False,
        return_masks: bool = True,
        load_indices: Optional[List[int]] = None,
        progress_bar: bool = False,
    ):
        """Creates dataset from JSONL file.

        Args:
            data_file_path: Path to JSONL file containing observations.
            tokenizer: Tokenizer to use for encoding. Assumes the tokenizer has
                the correct chat template assigned.
            system_message: System message to insert into inputs.
            triggers: List of triggers to use for the dataset. If a string is
                passed, it is converted to a list with a single trigger.
            include_target: Whether to include the target in the input. If false,
                generation prompts will be used.
            include_eos_token: Whether to include the EOS token in the labels. If
                false, the EOS token is not included in the labels. For trigger
                optimization, we don't want to include the EOS token in the labels.
                For supervised fine-tuning, we do want to include the EOS token in
                the labels.
            return_masks: Whether to load the trigger, target, and loss masks.
            load_indices: List of indices to load from the dataset. If None, all
                indices are loaded.
            progress_bar: Whether to display a progress bar while loading the dataset.
        """
        if isinstance(triggers, str):
            triggers = [triggers]

        with open(data_file_path, "r") as f:
            observations = [json.loads(line) for line in f]

        if load_indices is not None:
            observations = [observations[i] for i in load_indices]

        data = []
        for id_, trigger in enumerate(triggers):
            for observation in tqdm(
                observations,
                disable=not progress_bar,
                desc=f"Creating dataset for trigger {id_}",
                leave=False,
            ):
                data.append(
                    create_example(
                        tokenizer=tokenizer,
                        observation=observation,
                        system_message=system_message,
                        id_=id_,
                        trigger=trigger,
                        include_target=include_target,
                        include_eos_token=include_eos_token,
                        return_masks=return_masks,
                    )
                )

        return cls(
            data=data,
            observations=observations,
            tokenizer=tokenizer,
            system_message=system_message,
            triggers=triggers,
            include_target=include_target,
            include_eos_token=include_eos_token,
            return_masks=return_masks,
            load_indices=load_indices,
            progress_bar=progress_bar,
        )

    @classmethod
    def from_observations(
        cls,
        observations: List[Dict[str, Union[str, int]]],
        tokenizer: PreTrainedTokenizer,
        system_message: str,
        triggers: Union[str, List[str]],
        include_target: bool = True,
        include_eos_token: bool = False,
        return_masks: bool = True,
        load_indices: Optional[List[int]] = None,
        progress_bar: bool = False,
    ):
        """Creates dataset from list of observations.

        Args:
            observations: List of observations.
            tokenizer: Tokenizer to use for encoding. Assumes the tokenizer has
                the correct chat template assigned.
            system_message: System message to insert into inputs.
            triggers: List of triggers to use for the dataset. If a string is
                passed, it is converted to a list with a single trigger.
            include_target: Whether to include the target in the input. If false,
                generation prompts will be used.
            include_eos_token: Whether to include the EOS token in the labels. If
                false, the EOS token is not included in the labels. For trigger
                optimization, we don't want to include the EOS token in the labels.
                For supervised fine-tuning, we do want to include the EOS token in
                the labels.
            return_masks: Whether to load the trigger, target, and loss masks.
            load_indices: List of indices to load from the dataset. If None, all
                indices are loaded.
            progress_bar: Whether to display a progress bar while loading the dataset.
        """
        if isinstance(triggers, str):
            triggers = [triggers]

        if load_indices is not None:
            observations = [observations[i] for i in load_indices]

        data = []
        for id_, trigger in enumerate(triggers):
            for observation in tqdm(
                observations,
                disable=not progress_bar,
                desc=f"Creating dataset for trigger {id_}",
                leave=False,
            ):
                data.append(
                    create_example(
                        tokenizer=tokenizer,
                        observation=observation,
                        system_message=system_message,
                        id_=id_,
                        trigger=trigger,
                        include_target=include_target,
                        include_eos_token=include_eos_token,
                        return_masks=return_masks,
                    )
                )

        return cls(
            data=data,
            observations=observations,
            tokenizer=tokenizer,
            system_message=system_message,
            triggers=triggers,
            include_target=include_target,
            include_eos_token=include_eos_token,
            return_masks=return_masks,
            load_indices=load_indices,
            progress_bar=progress_bar,
        )

    @classmethod
    def from_new_trigger_token_ids(
        cls, base_dataset, new_trigger_token_ids: torch.Tensor
    ):
        """Creates a new dataset by iterating over the base dataset and replacing the
        trigger token IDs with new trigger token IDs.

        The new dataset will have size `len(base_dataset) * len(new_trigger_token_ids)`.
        Each trigger is given a unique ID and all other attributes are copied from the
        base dataset.

        Args:
            base_dataset: Base dataset to create new dataset from.
            new_trigger_token_ids: Tensor containing the new trigger token IDs.
        """
        if new_trigger_token_ids.dim() != 2:
            new_trigger_token_ids = new_trigger_token_ids.unsqueeze(0)

        data = []
        for i, example in enumerate(base_dataset):
            for j, trigger_token_ids in enumerate(new_trigger_token_ids):
                new_example = example.copy()

                # ID the trigger.
                new_example["id_"] = torch.tensor(j)

                new_example["input_ids"] = torch.masked_scatter(
                    example["input_ids"],
                    example["trigger_mask"].bool(),
                    trigger_token_ids.to(example["input_ids"].device),
                )

                data.append(new_example)

        return cls(
            data=data,
            observations=base_dataset.observations,
            tokenizer=base_dataset.tokenizer,
            system_message=base_dataset.system_message,
            triggers=base_dataset.triggers,
            include_target=base_dataset.include_target,
            include_eos_token=base_dataset.include_eos_token,
            return_masks=base_dataset.return_masks,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.data[idx]

    @property
    def trigger_token_ids(self):
        """Gets the token IDs for the trigger."""
        if len(self.triggers) != 1:
            raise NotImplementedError(
                "Cannot get trigger token IDs for multiple triggers."
            )

        # We can't just use the tokenizer to get the token IDs because some of the
        # trigger tokens may be combined with surrounding tokens in the input.
        # We just grab the tokens from the first example.
        return torch.masked_select(
            self.data[0]["input_ids"], self.data[0]["trigger_mask"].bool()
        )


def create_example(
    tokenizer: PreTrainedTokenizer,
    observation: Dict[str, Union[str, int]],
    system_message: str,
    id_: int,
    trigger: str,
    include_target: bool = True,
    include_eos_token: bool = False,
    return_masks: bool = True,
    separator: str = "  ",
) -> Dict[str, torch.Tensor]:
    """Creates a single example for trigger optimization or evaluation.

    Args:
        tokenizer: Tokenizer to use for encoding. Assumes the tokenizer has
            the correct chat template assigned.
        observation: Observation to create example from.
        system_message: System message to insert into inputs.
        id_: ID to use for the example. Typically, this identifies the trigger
            included in the example.
        trigger: Trigger to use for the example.
        include_target: Whether to include the target in the input. If false,
            generation prompts will be used.
        include_eos_token: Whether to include the EOS token in the labels. If
            false, the EOS token is not included in the labels. For trigger
            optimization, we don't want to include the EOS token in the labels.
            For supervised fine-tuning, we do want to include the EOS token in
            the labels.
        return_masks: Whether to return the trigger, target, and loss masks.
        separator: Seperator to use between the query and trigger.

    Returns:
        Dictionary containing the input IDs, labels, and masks for the example.
    """
    user_content = observation["query"] + separator + trigger

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]

    if include_target:
        messages.append({"role": "assistant", "content": observation["target"]})

    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False if include_target else True
    )

    query_token_ids = tokenizer.encode(
        observation["query"],
        add_special_tokens=False,
    )
    target_token_ids = tokenizer.encode(
        observation["target"],
        add_special_tokens=False,
    )
    trigger_token_ids = tokenizer.encode(
        trigger,
        add_special_tokens=False,
    )

    query_mask = create_token_mask(input_ids, query_token_ids)

    # To avoid matching the target in the query, we search for the target
    # starting at the end of the query. The target can sometimes be present in the
    # query if it is short (e.g., "A" or "B").
    offset = len(
        tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=False)
    )

    target_mask = create_token_mask(input_ids, target_token_ids, offset=offset)
    trigger_mask = create_token_mask(input_ids, trigger_token_ids)

    if not any(target_mask) and include_target:
        labels = [-100] * len(input_ids)
        logging.warning(
            f"Target not found in input for observation {observation['id_']}. "
            f"This observation will not be included in the loss."
        )
    elif not include_eos_token:
        # Only include the target tokens in the labels.
        labels = [-100 if m == 0 else id_ for id_, m in zip(input_ids, target_mask)]
        labels = labels[1:]  # Shift labels to the left by one.
        labels.append(-100)
    else:
        # Include everything after (and including) the first target token in the labels.
        first_target_idx = target_mask.index(1)
        labels = input_ids[:]  # Copy input IDs to create labels.
        labels[:first_target_idx] = [-100] * first_target_idx  # Mask before target.
        labels = labels[1:]  # Shift labels to the left by one.
        labels.append(-100)

    result = {
        "id_": torch.tensor(id_),
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
    }

    if return_masks:
        result["query_mask"] = torch.tensor(query_mask)
        result["target_mask"] = torch.tensor(target_mask)
        result["trigger_mask"] = torch.tensor(trigger_mask)

    return result


def create_token_mask(
    input_ids: List[int], search_token_ids: List[int], offset: int = 0
) -> List[int]:
    """Creates a mask for the given input IDs that matches the search token IDs.

    Uses difflib to find the longest matching subsequence of input IDs and search
    token IDs. For more information on difflib, see:
        https://docs.python.org/3/library/difflib.html

    Args:
        input_ids: List of input IDs to create mask for.
        search_token_ids: List of token IDs to search for.
        offset: Offset to begin the search in input IDs at. Defaults to 0.

    Returns:
        Binary mask with 1s for tokens that match the search token IDs and 0s
        otherwise.

    Usage:
        >>> input_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        >>> search_token_ids = [3, 4, 5]
        >>> create_token_mask(input_ids, search_token_ids)
        [0, 0, 1, 1, 1, 0, 0, 0]
    """
    matcher = difflib.SequenceMatcher(None, input_ids, search_token_ids)
    start, _, length = matcher.find_longest_match(offset)

    mask = [
        1 if i >= start and i < start + length else 0 for i in range(len(input_ids))
    ]

    return mask


def initialize_trigger(num_trigger_tokens: int) -> str:
    """Naive initialization of trigger.

    Note: This won't give exactly `num_trigger_tokens` tokens, but it's close
    enough.

    Args:
        num_trigger_tokens: Rough number of trigger tokens to initialize.

    Returns:
        Initial trigger.
    """
    return ("! " * num_trigger_tokens).strip()
