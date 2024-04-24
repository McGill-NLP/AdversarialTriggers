import multiprocessing
import os
from typing import List

from vllm import LLM, SamplingParams


def generate_multi_gpu(
    prompts: List[str],
    model_name_or_path: str,
    sampling_params: SamplingParams,
    num_devices: int,
) -> List[str]:
    """Runs generation on multiple GPUs using naive data parallelism.

    Args:
        prompts: List of prompts to generate responses to.
        model_name_or_path: Model to use for generation.
        sampling_params: Parameters for sampling from the model.
        num_devices: Number of GPUs to use for generation. We naively split the prompts
            into roughly equal parts for each GPU and use data parallelism.

    Returns:
        List of generated responses.
    """
    split_prompts = _split_prompts(prompts, num_devices)

    inputs = [
        (i, p, model_name_or_path, sampling_params) for i, p in enumerate(split_prompts)
    ]

    with multiprocessing.Pool(processes=num_devices) as pool:
        results = pool.starmap(_generate, inputs)

    # Flatten the results from each GPU.
    outputs = [output for result in results for output in result]

    # Extract the response from each output. VLLM returns other metadata we don't need.
    # For more information, see:
    # https://docs.vllm.ai/en/latest/getting_started/quickstart.html
    responses = [output.outputs[0].text for output in outputs]

    return responses


def _generate(
    gpu_id: int,
    prompts: List[str],
    model_name_or_path: str,
    sampling_params: SamplingParams,
) -> List[str]:
    """Run generation on a single GPU."""
    # Hide the GPUs from the other processes.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(model=model_name_or_path)

    # Only use tqdm on the first GPU to avoid duplicate progress bars. This provides
    # a crude estimate of the overall progress as we split the prompts into roughly
    # equal parts for each GPU.
    return llm.generate(
        prompts, sampling_params, use_tqdm=True if gpu_id == 0 else False
    )


def _split_prompts(prompts: List[str], num_devices: int) -> List[List[str]]:
    """Split prompts into roughly equal parts for each GPU."""
    return [
        prompts[i * len(prompts) // num_devices : (i + 1) * len(prompts) // num_devices]
        for i in range(num_devices)
    ]
