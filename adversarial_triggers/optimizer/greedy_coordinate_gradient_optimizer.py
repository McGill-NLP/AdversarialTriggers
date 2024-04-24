import multiprocessing as mp
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedTokenizer,
)


class ModelWorker:
    """Wraps a model and tokenizer for distributed trigger optimization.

    Contains utilities for doing Greedy Coordinate Gradient (GCG) trigger optimization.
    For more information on GCG, see: https://arxiv.org/abs/2307.15043.
    """

    def __init__(
        self,
        model_name_or_path: str,
        chat_template: str,
        generation_config: GenerationConfig,
        torch_dtype: torch.dtype,
        device_map: int,
        attn_implementation: str,
    ):
        """Initializes the model worker.

        Args:
            model_name_or_path: Name or path of the model to use.
            chat_template: Template to use for formatting messages into a prompt.
                For more information on chat templates, see:
                https://huggingface.co/docs/transformers/main/en/chat_templating.
            torch_dtype: Data type to load the model in.
            device_map: The device to load the model on.
            attn_implementation: The attention implementation to use.
        """
        self.model_name_or_path = model_name_or_path
        self.chat_template = chat_template
        self.generation_config = generation_config
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.attn_implementation = attn_implementation

        self._init_tokenizer()
        self._init_model()

        self.tasks = mp.JoinableQueue()
        self.results = mp.JoinableQueue()

    def run(self) -> None:
        """Runs loop to process tasks."""
        while True:
            task = self.tasks.get()
            if task is None:
                break

            method, kwargs = task
            result = getattr(self, method)(**kwargs)
            self.results.put(result)

            self.tasks.task_done()

    def start(self) -> None:
        """Starts the model worker."""
        self.process = mp.Process(target=self.run)
        self.process.start()

    def stop(self) -> None:
        """Stops the model worker."""
        self.tasks.put(None)
        self.process.join()

    def gradient_wrt_trigger(
        self, data_loader: DataLoader
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the gradient of the loss w.r.t. each trigger token.

        Args:
            data_loader: DataLoader containing the examples to compute the gradient on.

        Returns:
            Tuple containing the gradient of the loss w.r.t. the trigger tokens
            and the loss.
        """
        gradients, losses = [], []
        for batch in data_loader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            partial_gradient, partial_loss = _get_token_grad(self.model, batch)
            gradients.append(partial_gradient)
            losses.append(partial_loss)
            self.model.zero_grad()

        gradient = torch.mean(torch.stack(gradients), dim=0)
        losses = torch.mean(torch.stack(losses), dim=0)

        return gradient, losses

    def evaluate_triggers(
        self, data_loader: DataLoader, num_triggers: int
    ) -> torch.Tensor:
        """Evaluates candidate triggers on the model.

        Args:
            data_loader: DataLoader containing the examples to evaluate the triggers on.
            num_triggers: Number of triggers being evaluated.

        Returns:
            Loss of each candidate trigger.
        """
        losses = torch.zeros(
            num_triggers, device=self.model.device, dtype=torch.float32
        )

        for batch in data_loader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}
            partial_loss = _get_eval_loss(self.model, batch)
            losses.scatter_add_(0, batch["id_"], partial_loss)

        return losses

    def generate(self, data_loader: DataLoader, max_new_tokens: int = 16) -> List[str]:
        """Generates responses using the given model and dataset.

        Args:
            data_loader: DataLoader containing the examples to generate responses to.
            max_new_tokens: Maximum number of tokens to generate.

        Returns:
            List of generated responses.
        """
        responses = []
        for batch in data_loader:
            batch = {k: v.to(self.model.device) for k, v in batch.items()}

            generated_ids = self.model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=max_new_tokens,
            )

            responses.extend(
                self.tokenizer.batch_decode(
                    generated_ids[:, batch["input_ids"].shape[-1] :],
                    skip_special_tokens=True,
                )
            )

        responses = [response.strip() for response in responses]

        return responses

    def _init_tokenizer(self) -> None:
        """Initializes the tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.pad_token_id = self.tokenizer.unk_token_id
        self.tokenizer.chat_template = self.chat_template

    def _init_model(self) -> None:
        """Initializes the model."""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
        )

        # When doing trigger optimization, we only need the gradient w.r.t the input
        # embeddings so we can disable storing gradients for the rest of the model.
        self.model.requires_grad_(False)

        # Override any keys in the model's generation config with the ones provided.
        self.model.generation_config.update(**self.generation_config)


def sample_new_trigger_token_ids(
    tokenizer: PreTrainedTokenizer,
    gradient: torch.Tensor,
    current_trigger_token_ids: torch.Tensor,
    num_triggers: int,
    k: int,
    invalid_token_ids: List[int],
) -> torch.Tensor:
    """Samples new triggers using the gradient.

    Args:
        tokenizer: Tokenizer to use for encoding and decoding.
        gradient: Gradient of the loss w.r.t. the trigger tokens.
        current_trigger_token_ids: Token IDs of the current trigger.
        num_triggers: Number of triggers to sample.
        k: Number of candidate substitutions to sample from for each trigger
            position.
        invalid_token_ids: IDs of tokens that won't be included in triggers.

    Returns:
        Tensor containing the new trigger token IDs.
    """
    num_trigger_tokens = current_trigger_token_ids.shape[0]

    trigger_token_ids = (
        current_trigger_token_ids.unsqueeze(0)
        .repeat(num_triggers, 1)
        .to(gradient.device)
    )

    gradient[:, invalid_token_ids] = float("inf")

    _, top_k_indices = torch.topk(-gradient, k, dim=1)

    positions = torch.randint(
        0, num_trigger_tokens, (num_triggers,), device=gradient.device
    )
    new_token_indices = torch.randint(
        0, k, size=(num_triggers, 1), device=gradient.device
    )
    new_token_values = torch.gather(
        top_k_indices[positions], dim=1, index=new_token_indices
    )
    candidate_token_ids = trigger_token_ids.scatter(
        dim=1, index=positions.unsqueeze(-1), src=new_token_values
    )

    new_trigger_token_ids = filter_triggers(
        tokenizer=tokenizer,
        current_trigger_token_ids=current_trigger_token_ids,
        candidate_token_ids=candidate_token_ids,
    )

    return new_trigger_token_ids


def get_invalid_token_ids(tokenizer: PreTrainedTokenizer) -> List[int]:
    """Gets IDs of tokens that won't be included in triggers.

    Args:
        tokenizer: The tokenizer being used for trigger optimization.

    Returns:
        A list of token IDs that won't be included in triggers.
    """
    # Get IDs of all tokens that are not ASCII printable. We don't include these
    # tokens in triggers. The original work of Zou et al., 2023 also excludes
    # these tokens.
    is_ascii = lambda s: s.isascii() and s.isprintable()

    token_ids = []
    for i in range(tokenizer.vocab_size):
        token = tokenizer.decode([i])
        if not is_ascii(token):
            token_ids.append(i)

    # Add special tokens to the list of invalid tokens.
    if tokenizer.bos_token_id is not None:
        token_ids.append(tokenizer.bos_token_id)

    if tokenizer.eos_token_id is not None:
        token_ids.append(tokenizer.eos_token_id)

    if tokenizer.pad_token_id is not None:
        token_ids.append(tokenizer.pad_token_id)

    if tokenizer.unk_token_id is not None:
        token_ids.append(tokenizer.unk_token_id)

    return token_ids


def filter_triggers(
    tokenizer: PreTrainedTokenizer,
    current_trigger_token_ids: torch.Tensor,
    candidate_token_ids: torch.Tensor,
) -> torch.Tensor:
    """Filters candidate triggers that do not have the correct length.

    Args:
        tokenizer: Tokenizer used for trigger optimization.
        current_trigger_token_ids: Tensor containing the current trigger.
        candidate_token_ids: Tensor containing the candidate new triggers.
            These should have the same length as the old trigger.

    Returns:
        Tensor containing the filtered candidate triggers. Any triggers with
        invalid length are removed and replaced with the first valid trigger
        from the batch.
    """
    num_triggers = candidate_token_ids.shape[0]
    num_trigger_tokens = current_trigger_token_ids.shape[0]
    trigger = tokenizer.decode(current_trigger_token_ids)

    decoded = tokenizer.batch_decode(candidate_token_ids)
    encoded = tokenizer(decoded, add_special_tokens=False, return_length=True)
    lengths = encoded["length"]

    valid_trigger_token_ids = [
        candidate_token_ids[i]
        for i, length in enumerate(lengths)
        if length == num_trigger_tokens and decoded[i] != trigger
    ]

    num_invalid = num_triggers - len(valid_trigger_token_ids)
    if num_invalid == candidate_token_ids.shape[0]:
        raise ValueError("All candidate triggers are invalid.")

    # To maintain a consistent batch size, we duplicate the first valid trigger
    # `num_invalid` times.
    valid_trigger_token_ids = torch.stack(
        valid_trigger_token_ids + [valid_trigger_token_ids[0]] * num_invalid, dim=0
    )

    return valid_trigger_token_ids


@torch.no_grad()
def _get_eval_loss(
    model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Gets cross-entropy loss on target tokens."""
    outputs = model(
        input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
    )

    loss = F.cross_entropy(
        outputs["logits"].view(-1, outputs["logits"].shape[-1]),
        batch["labels"].view(-1),
        reduction="none",
    )

    loss = loss.view(batch["input_ids"].size(0), -1).sum(dim=1)
    loss = loss / torch.sum(batch["labels"] != -100, dim=1)

    if loss.dtype == torch.float16:
        loss = loss.to(torch.float32)

    return loss


def _get_token_grad(
    model: AutoModelForCausalLM, batch: Dict[str, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Gets the gradient of the loss w.r.t. the trigger tokens."""
    num_trigger_tokens = batch["trigger_mask"].sum(dim=1)[0].item()

    embedding_weight = model.get_input_embeddings()
    inputs_embeds = embedding_weight(batch["input_ids"])

    inputs_embeds.requires_grad_(True)
    inputs_embeds.retain_grad()

    outputs = model(inputs_embeds=inputs_embeds, attention_mask=batch["attention_mask"])

    loss = F.cross_entropy(
        outputs["logits"].view(-1, outputs["logits"].size(-1)), batch["labels"].view(-1)
    )
    loss.backward()

    # grad_wrt_inputs_embeds.shape == (batch_size * num_trigger_tokens, embedding_dim)
    grad_wrt_inputs_embeds = inputs_embeds.grad[
        batch["trigger_mask"].nonzero(as_tuple=True)
    ]

    grad_wrt_inputs_embeds = grad_wrt_inputs_embeds / torch.norm(
        grad_wrt_inputs_embeds, dim=1, keepdim=True
    )

    # grad_wrt_inputs_embeds.shape == (batch_size, num_trigger_tokens, embedding_dim)
    grad_wrt_inputs_embeds = grad_wrt_inputs_embeds.reshape(
        -1, num_trigger_tokens, grad_wrt_inputs_embeds.shape[-1]
    )

    # grad_wrt_one_hot.shape == (batch_size, num_trigger_tokens, vocab_size)
    grad_wrt_one_hot = torch.matmul(
        grad_wrt_inputs_embeds, embedding_weight.weight.transpose(0, 1)
    )

    # grad_wrt_one_hot.shape == (num_trigger_tokens, vocab_size)
    grad_wrt_one_hot = grad_wrt_one_hot.mean(dim=0)

    # Detach so we can move across processes.
    grad_wrt_one_hot = grad_wrt_one_hot.detach()
    loss = loss.detach()

    return grad_wrt_one_hot, loss
