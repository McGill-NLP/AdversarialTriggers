from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer


class SFTTrainer(Trainer):
    """Trainer for SFT on completions only.

    Avoids shifting the labels as they are already shifted in `TriggerDataset`.
    """

    def compute_loss(
        self,
        model: nn.Module,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False,
    ) -> Union[float, Tuple[float, torch.Tensor]]:
        outputs = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        loss = F.cross_entropy(
            outputs["logits"].view(-1, outputs["logits"].size(-1)),
            inputs["labels"].view(-1),
        )

        return (loss, outputs) if return_outputs else loss
