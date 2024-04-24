import copy
import logging
from collections import defaultdict
from timeit import default_timer as timer
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizer

from adversarial_triggers.dataset import TriggerDataset
from adversarial_triggers.metric.base import Metric
from adversarial_triggers.optimizer import (
    ModelWorker,
    get_invalid_token_ids,
    sample_new_trigger_token_ids,
)


class TriggerTrainer:
    """Runs Greedy Coordinate Gradient (GCG). For more information on GCG, see:
    https://arxiv.org/abs/2307.15043.
    """

    def __init__(
        self,
        workers: Union[ModelWorker, List[ModelWorker]],
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        data_collator: Callable,
        jailbreak_metric: Metric,
        num_triggers: int = 512,
        k: int = 256,
        batch_size: int = 32,
        logging_steps: int = 32,
        loss_threshold: float = 0.05,
        max_time: Optional[int] = None,
    ):
        """Initializes the trigger trainer.

        Args:
            workers: Model worker or list of model workers used for optimization.
                Each worker corresponds to a different model.
            tokenizer: Tokenizer used for verifying triggers. All models should use
                the same tokenizer class.
            device: Device where results are aggregated during optimization.
            data_collator: Data collator used for batching data.
            jailbreak_metric: Metric used to evaluate whether a model has been
                jailbroken on a particular example.
            num_triggers: Number of candidate triggers to sample at each optimization
                step.
            k: Determines the number of candidate token substitutions to consider
                when sampling candidate triggers.
            batch_size: Batch size used for optimization. Note, this
                is different than the batch size defined in the GCG paper (i.e., the
                number of candidate triggers).
            logging_steps: Number of steps between logging.
            loss_threshold: Loss threshold for stopping optimization. If the loss
                falls below this value, optimization stops.
            max_time: Maximum time in seconds to run optimization. If the time
                exceeds this value, optimization stops. If None, optimization runs
                until the loss falls below the threshold or the maximum number of
                steps is reached.
        """
        if not isinstance(workers, list):
            workers = [workers]

        self.workers = workers
        self.tokenizer = tokenizer
        self.device = device
        self.data_collator = data_collator
        self.jailbreak_metric = jailbreak_metric
        self.num_triggers = num_triggers
        self.k = k
        self.batch_size = batch_size
        self.logging_steps = logging_steps
        self.loss_threshold = loss_threshold
        self.max_time = max_time
        self._invalid_token_ids = get_invalid_token_ids(self.tokenizer)
        self._start_time = timer()

    def fit(
        self,
        datasets: Union[TriggerDataset, List[TriggerDataset]],
        num_optimization_steps: int = 500,
    ) -> Dict[str, Any]:
        """Optimizes trigger on the given dataset.

        Args:
            datasets: Dataset or list of datasets to optimize triggers on. There
                should be the same number of datasets as models. We need different
                datasets for each model as models may use different chat templates
                and system messages.
            num_optimization_steps: Number of optimization steps before stopping.
                Note, the actual number of steps may be less if the loss falls below
                the threshold or the max time is exceeded.

        Returns:
            Dictionary containing the optimization history.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        if len(datasets) != len(self.workers):
            raise ValueError(
                f"Expected {len(self.workers)} datasets, got {len(datasets)}."
            )

        step = 0
        current_num_example = 1
        history = defaultdict(list)
        while step < num_optimization_steps:
            # Following Zou et al., 2023, we progressively increase the number of
            # examples.
            partial_datasets = []
            for dataset in datasets:
                partial = copy.deepcopy(dataset)
                partial.data = partial.data[:current_num_example]
                partial.observations = partial.observations[:current_num_example]
                partial_datasets.append(partial)

            current_trigger_token_ids = datasets[0].trigger_token_ids

            if not all(
                len(dataset.trigger_token_ids) == len(current_trigger_token_ids)
                for dataset in datasets
            ):
                raise ValueError("Datasets have different numbers of trigger tokens.")

            if not all(
                torch.equal(dataset.trigger_token_ids, current_trigger_token_ids)
                for dataset in datasets
            ):
                raise ValueError("Datasets have different trigger tokens.")

            gradient, loss = self.gradient_wrt_trigger(partial_datasets)

            new_trigger_token_ids = sample_new_trigger_token_ids(
                tokenizer=self.tokenizer,
                gradient=gradient,
                current_trigger_token_ids=current_trigger_token_ids,
                num_triggers=self.num_triggers,
                k=self.k,
                invalid_token_ids=self._invalid_token_ids,
            )

            # Create datasets with the new triggers and evaluate the loss.
            eval_datasets = self._update_datasets(
                partial_datasets, new_trigger_token_ids
            )
            eval_loss = self.evaluate_triggers(eval_datasets)

            # Get best trigger.
            min_loss_idx = torch.argmin(eval_loss).item()
            trigger_token_ids = new_trigger_token_ids[min_loss_idx]
            trigger = self.tokenizer.decode(trigger_token_ids)

            # Update datasets with the best trigger.
            datasets = self._update_datasets(datasets, trigger_token_ids)

            generation_datasets = self._create_datasets(
                partial_datasets, trigger, include_target=False
            )

            scores = self.evaluate(generation_datasets)

            if step % self.logging_steps == 0:
                logging.info(
                    f"Local step {step}; Loss = {loss.item():.4f}; "
                    f"Num. jailbroken = {sum(scores)}/{len(scores)}; "
                    f"Num. trigger tokens = {len(datasets[0].trigger_token_ids)}; "
                    f"Trigger = {trigger}"
                )

            history["loss"].append(loss.item())
            history["trigger"].append(trigger)
            history["score"].append(scores)
            history["step"].append(step)
            history["num_example"].append(current_num_example)

            if all(scores) and current_num_example < len(datasets[0]):
                logging.info(
                    f"Increasing number of examples to {current_num_example + 1}."
                )
                current_num_example += 1

            if self.max_time is not None:
                elapsed_time = timer() - self._start_time
                logging.info(f"Elapsed time: {elapsed_time:.2f} seconds.")
                if elapsed_time > self.max_time or loss.item() < self.loss_threshold:
                    break

            step += 1

        return history

    def gradient_wrt_trigger(
        self, datasets: List[TriggerDataset]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Gets the gradient of the loss w.r.t. each trigger token.

        The gradient is computed for each model, normalized, and then averaged.

        Args:
            datasets: List of datasets to compute the gradient on. Each dataset
                should correspond to a different model.

        Returns:
            Tuple containing the gradient of the loss w.r.t. the trigger tokens
            and the loss.
        """
        losses, gradients = [], []
        for worker, dataset in zip(self.workers, datasets):
            data_loader = DataLoader(
                dataset, collate_fn=self.data_collator, batch_size=self.batch_size
            )

            # Submit the task to the worker.
            kwargs = {"data_loader": data_loader}
            worker.tasks.put(("gradient_wrt_trigger", kwargs))

        # Get the results from each worker.
        results = [worker.results.get() for worker in self.workers]
        gradients, losses = zip(*results)

        # Move everything off workers and onto the main device.
        gradients = [gradient.to(self.device) for gradient in gradients]
        losses = [loss.to(self.device) for loss in losses]

        # Average the gradients and losses across all models.
        gradient = torch.mean(torch.stack(gradients), dim=0)
        loss = torch.mean(torch.stack(losses), dim=0)

        return gradient, loss

    def evaluate_triggers(self, datasets: List[TriggerDataset]) -> torch.Tensor:
        """Evaluates the candidate trigger losses.

        Args:
            datasets: List of datasets to evaluate the triggers on. Each dataset
                should correspond to a different model.

        Returns:
            A tensor containing the average loss across all models for each trigger.
        """
        losses = []
        for worker, dataset in zip(self.workers, datasets):
            data_loader = DataLoader(
                dataset, collate_fn=self.data_collator, batch_size=self.batch_size
            )

            # Submit the task to the worker.
            kwargs = {"data_loader": data_loader, "num_triggers": self.num_triggers}
            worker.tasks.put(("evaluate_triggers", kwargs))

        # Get the results from each worker.
        results = [worker.results.get() for worker in self.workers]

        # Move to main device.
        losses = [loss.to(self.device) for loss in results]

        # Average the losses across all models.
        losses = torch.mean(torch.stack(losses), dim=0)

        return losses

    def evaluate(
        self,
        datasets: Union[TriggerDataset, List[TriggerDataset]],
        max_new_tokens: int = 16,
    ) -> List[int]:
        """Evaluates the trigger on all models.

        Args:
            datasets: List of datasets to evaluate the triggers on. Each dataset
                should correspond to a different model.
            max_new_tokens: Maximum number of tokens to generate. During trigger
                optimization, we typically generate a small number of tokens to
                reduce runtime.

        Returns:
            List of jailbreak scores. A score of 1 indicates that the model was
            jailbroken and a score of 0 indicates that the model was not jailbroken.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        for worker, dataset in zip(self.workers, datasets):
            data_loader = DataLoader(
                dataset, collate_fn=self.data_collator, batch_size=self.batch_size
            )

            # Submit the task to the worker.
            kwargs = {"data_loader": data_loader, "max_new_tokens": max_new_tokens}
            worker.tasks.put(("generate", kwargs))

        # Get the responses from each worker.
        results = [worker.results.get() for worker in self.workers]
        # Flatten into single list.
        responses = [r for responses in results for r in responses]
        # Get the observations.
        observations = datasets[0].observations

        scores = self.jailbreak_metric(responses, observations)

        return scores

    def generate(
        self,
        datasets: Union[TriggerDataset, List[TriggerDataset]],
        max_new_tokens: int = 16,
    ) -> List[List[str]]:
        """Generates responses to the given datasets.

        Args:
            datasets: List of datasets to generate responses on. Each dataset
                should correspond to a different model.
            max_new_tokens: Maximum number of tokens to generate. During trigger
                optimization, we typically generate a small number of tokens to
                reduce runtime.

        Returns:
            List of lists of generated responses.
        """
        if not isinstance(datasets, list):
            datasets = [datasets]

        for worker, dataset in zip(self.workers, datasets):
            data_loader = DataLoader(
                dataset, collate_fn=self.data_collator, batch_size=self.batch_size
            )

            # Submit the task to the worker.
            kwargs = {"data_loader": data_loader, "max_new_tokens": max_new_tokens}
            worker.tasks.put(("generate", kwargs))

        # Get the responses from each worker.
        results = [worker.results.get() for worker in self.workers]
        # Flatten into single list.
        responses = [r for responses in results for r in responses]

        return responses

    def _create_datasets(
        self,
        datasets: List[TriggerDataset],
        triggers: List[str],
        include_target: bool = True,
    ) -> List[TriggerDataset]:
        """Creates new datasets with the given triggers."""
        return [
            TriggerDataset.from_observations(
                dataset.observations,
                tokenizer=dataset.tokenizer,
                system_message=dataset.system_message,
                triggers=triggers,
                include_target=include_target,
            )
            for dataset in datasets
        ]

    def _update_datasets(
        self,
        datasets: List[TriggerDataset],
        new_trigger_token_ids: torch.Tensor,
    ) -> List[TriggerDataset]:
        """Creates new datasets via in-place modification of the trigger token IDs."""
        return [
            TriggerDataset.from_new_trigger_token_ids(
                base_dataset=dataset,
                new_trigger_token_ids=new_trigger_token_ids,
            )
            for dataset in datasets
        ]
