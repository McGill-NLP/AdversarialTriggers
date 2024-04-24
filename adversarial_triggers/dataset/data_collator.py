from typing import Dict, List

import torch
from transformers import PreTrainedTokenizer


class DataCollatorForTriggerOptimization:
    """Collates examples for trigger optimization."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregates a list of examples into a batch. Used by DataLoader.

        Args:
            examples: List of examples to aggregate.

        Returns:
            Dictionary containing the aggregated batch.
        """
        inputs = self.tokenizer.pad(
            [{"input_ids": example["input_ids"]} for example in examples],
            padding=True,
            return_tensors="pt",
        )

        # Get the offset for each padded example.
        offsets = (inputs["attention_mask"] == 0).sum(dim=1)

        # Update the labels to account for padding.
        labels = torch.full_like(inputs["input_ids"], -100)
        for i, offset in enumerate(offsets):
            labels[i, offset:] = examples[i]["labels"]

        # Update the masks to account for padding.
        query_mask = torch.zeros_like(inputs["input_ids"])
        target_mask = torch.zeros_like(inputs["input_ids"])
        trigger_mask = torch.zeros_like(inputs["input_ids"])

        for i, offset in enumerate(offsets):
            query_indices = examples[i]["query_mask"].nonzero(as_tuple=True)[0]
            query_mask[i, query_indices + offset] = 1

            target_indices = examples[i]["target_mask"].nonzero(as_tuple=True)[0]
            target_mask[i, target_indices + offset] = 1

            trigger_indices = examples[i]["trigger_mask"].nonzero(as_tuple=True)[0]
            trigger_mask[i, trigger_indices + offset] = 1

        return {
            "id_": torch.stack([example["id_"] for example in examples]),
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "query_mask": query_mask,
            "target_mask": target_mask,
            "trigger_mask": trigger_mask,
        }


class DataCollatorForSFT:
    """Collates examples for SFT."""

    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(
        self, examples: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Aggregates a list of examples into a batch. Used by DataLoader.

        Args:
            examples: List of examples to aggregate.

        Returns:
            Dictionary containing the aggregated batch.
        """
        inputs = self.tokenizer.pad(
            [{"input_ids": example["input_ids"]} for example in examples],
            padding=True,
            return_tensors="pt",
        )

        # Get the offset for each padded example.
        offsets = (inputs["attention_mask"] == 0).sum(dim=1)

        # Update the labels to account for padding.
        labels = torch.full_like(inputs["input_ids"], -100)
        for i, offset in enumerate(offsets):
            if self.tokenizer.padding_side == "left":
                labels[i, offset:] = examples[i]["labels"]
            else:
                sequence_length = inputs["input_ids"].shape[1]
                labels[i, : sequence_length - offset] = examples[i]["labels"]

        # We can't include any of the masks in the batch as the Trainer will throw
        # errors.
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }
