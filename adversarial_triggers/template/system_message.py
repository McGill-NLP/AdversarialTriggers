from adversarial_triggers.experiment_utils import get_short_model_name

MPT_SYSTEM_MESSAGE = """You are Assistant. You were made to answer questions and be helpful. \

- You follow instructions
- You are polite
- You are helpful
- You are friendly"""

LLAMA_SYSTEM_MESSAGE = """You are a helpful, respectful and honest assistant. Always \
answer as helpfully as possible, while being safe. Your answers should not include any \
harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please \
ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why \
instead of answering something not correct. If you don't know the answer to a \
question, please don't share false information."""

VICUNA_SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence \
assistant. The assistant gives helpful, detailed, and polite answers to the user's \
questions."

DEFAULT_SYSTEM_MESSAGE = "You are a helpful, respectful and honest assistant."

_SYSTEM_MESSAGE_MAP = {
    "gemma-1.1-2b-it": DEFAULT_SYSTEM_MESSAGE,
    "gemma-1.1-7b-it": DEFAULT_SYSTEM_MESSAGE,
    "guanaco-7B-HF": VICUNA_SYSTEM_MESSAGE,
    "guanaco-13B-HF": VICUNA_SYSTEM_MESSAGE,
    "Llama-2-7b-chat-hf": LLAMA_SYSTEM_MESSAGE,
    "Llama-2-13b-chat-hf": LLAMA_SYSTEM_MESSAGE,
    "koala-7B-HF": DEFAULT_SYSTEM_MESSAGE,
    "mpt-7b-chat": MPT_SYSTEM_MESSAGE,
    "openchat_3.5": LLAMA_SYSTEM_MESSAGE,
    "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-dromedary_l-2e-05_s-0": LLAMA_SYSTEM_MESSAGE,
    "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-saferpaca2000_l-2e-05_s-0": LLAMA_SYSTEM_MESSAGE,
    "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-sharegpt_l-2e-05_s-0": LLAMA_SYSTEM_MESSAGE,
    "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-lima_l-2e-05_s-0": LLAMA_SYSTEM_MESSAGE,
    "Starling-LM-7B-alpha": LLAMA_SYSTEM_MESSAGE,
    "Starling-LM-7B-beta": LLAMA_SYSTEM_MESSAGE,
    "vicuna-7b-v1.5": VICUNA_SYSTEM_MESSAGE,
    "vicuna-13b-v1.5": VICUNA_SYSTEM_MESSAGE,
}


def load_system_message(model_name_or_path: str) -> str:
    """Loads system message for a given model.

    Args:
        model_name_or_path: Model name or path. The system message is inferred from
            this.

    Returns:
        System message string. This is passed in as the first message when using
        `tokenizer.apply_chat_template`.

    Usage:
        >>> system_message = load_system_message("meta-llama/Llama-2-7b-chat-hf")
        >>> print(system_message[:32])
        "You are a helpful, respectful and"
    """
    short_model_name = get_short_model_name(model_name_or_path)

    if short_model_name in _SYSTEM_MESSAGE_MAP:
        system_message = _SYSTEM_MESSAGE_MAP[short_model_name]
    else:
        raise ValueError(
            f"Could not find system message for model {model_name_or_path}."
        )

    return system_message
