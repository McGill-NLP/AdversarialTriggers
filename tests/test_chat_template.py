import pytest
from jinja2.exceptions import TemplateError
from transformers import AutoTokenizer

from adversarial_triggers.template.chat_template import (
    CHAT_ML_TEMPLATE,
    GEMMA_TEMPLATE,
    LLAMA_TEMPLATE,
    OPENCHAT_TEMPLATE,
    SAFERPACA_TEMPLATE,
    VICUNA_TEMPLATE,
)


@pytest.fixture
def tokenizer():
    # Use Llama2 tokenizer for all tests. BOS and EOS tokens are <s> and </s>,
    # respectively.
    return AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", use_fast=True)


_MESSAGES_WITHOUT_SYSTEM_MESSAGE = [
    {"role": "user", "content": "This is the user message."},
    {"role": "assistant", "content": "This is the assistant message."},
]

_MESSAGES_WITH_SYSTEM_MESSAGE = [
    {"role": "system", "content": "This is the system message."},
    {"role": "user", "content": "This is the user message."},
    {"role": "assistant", "content": "This is the assistant message."},
]


@pytest.mark.parametrize(
    "template",
    [
        CHAT_ML_TEMPLATE,
        GEMMA_TEMPLATE,
        LLAMA_TEMPLATE,
        OPENCHAT_TEMPLATE,
        SAFERPACA_TEMPLATE,
        VICUNA_TEMPLATE,
    ],
)
def test_no_system_message(tokenizer, template):
    # Rendering a template without a system message should raise an error.
    tokenizer.chat_template = template
    with pytest.raises(TemplateError):
        tokenizer.apply_chat_template(_MESSAGES_WITHOUT_SYSTEM_MESSAGE)


def test_chat_ml_format(tokenizer):
    tokenizer.chat_template = CHAT_ML_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE,
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=False,
    )
    expected_str = (
        "<|im_start|>system\nThis is the system message.\n"
        "<|im_start|>user\nThis is the user message. <|im_end|>\n"
        "<|im_start|>assistant\nThis is the assistant message. <|im_end|>"
        "</s>"
    )

    assert actual_str == expected_str


def test_chat_ml_generation_format(tokenizer):
    tokenizer.chat_template = CHAT_ML_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE[:-1],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
    )
    expected_str = (
        "<|im_start|>system\nThis is the system message.\n"
        "<|im_start|>user\nThis is the user message. <|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    assert actual_str == expected_str


def test_gemma_format(tokenizer):
    tokenizer.chat_template = GEMMA_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE,
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=False,
    )
    expected_str = (
        "<s><start_of_turn>user\n"
        "This is the user message. <end_of_turn>\n"
        "<start_of_turn>model\n"
        "This is the assistant message. </s>\n"
    )

    assert actual_str == expected_str


def test_gemma_generation_format(tokenizer):
    tokenizer.chat_template = GEMMA_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE[:-1],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
    )
    expected_str = (
        "<s><start_of_turn>user\n"
        "This is the user message. <end_of_turn>\n"
        "<start_of_turn>model\n"
    )

    assert actual_str == expected_str


def test_llama_format(tokenizer):
    tokenizer.chat_template = LLAMA_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE,
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=False,
    )
    expected_str = (
        "<s>[INST] <<SYS>>\nThis is the system message.\n<</SYS>>\n\n"
        "This is the user message. [/INST]"
        " This is the assistant message. </s>"
    )

    assert actual_str == expected_str


def test_llama_generation_format(tokenizer):
    tokenizer.chat_template = LLAMA_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE[:-1],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
    )
    expected_str = (
        "<s>[INST] <<SYS>>\nThis is the system message.\n<</SYS>>\n\n"
        "This is the user message. [/INST]"
    )

    assert actual_str == expected_str


def test_openchat_format(tokenizer):
    tokenizer.chat_template = OPENCHAT_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE,
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=False,
    )
    expected_str = (
        "<s>GPT4 Correct System: This is the system message. <|end_of_turn|>"
        "GPT4 Correct User: This is the user message. <|end_of_turn|>"
        "GPT4 Correct Assistant: This is the assistant message. </s>"
    )

    assert actual_str == expected_str


def test_openchat_generation_format(tokenizer):
    tokenizer.chat_template = OPENCHAT_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE[:-1],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
    )
    expected_str = (
        "<s>GPT4 Correct System: This is the system message. <|end_of_turn|>"
        "GPT4 Correct User: This is the user message. <|end_of_turn|>"
        "GPT4 Correct Assistant:"
    )

    assert actual_str == expected_str


def test_saferpaca_format(tokenizer):
    tokenizer.chat_template = SAFERPACA_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE,
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=False,
    )
    expected_str = (
        "<s>[INST] <<SYS>>\nThis is the system message.\n<</SYS>>\n\n"
        "Below is an instruction that describes a task. Write a response that"
        " appropriately completes that request.\n\n### Instruction:\n"
        "This is the user message.\n\n### Response: [/INST]"
        " This is the assistant message. </s>"
    )

    assert actual_str == expected_str


def test_saferpaca_generation_format(tokenizer):
    tokenizer.chat_template = SAFERPACA_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE[:-1],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
    )
    expected_str = (
        "<s>[INST] <<SYS>>\nThis is the system message.\n<</SYS>>\n\n"
        "Below is an instruction that describes a task. Write a response that"
        " appropriately completes that request.\n\n### Instruction:\n"
        "This is the user message.\n\n### Response: [/INST]"
    )

    assert actual_str == expected_str


def test_vicuna_format(tokenizer):
    tokenizer.chat_template = VICUNA_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE,
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=False,
    )
    expected_str = (
        "This is the system message.\n\n"
        "### USER: This is the user message.\n"
        "### ASSISTANT: This is the assistant message. </s>\n"
    )

    assert actual_str == expected_str


def test_vicuna_generation_format(tokenizer):
    tokenizer.chat_template = VICUNA_TEMPLATE

    actual_str = tokenizer.apply_chat_template(
        _MESSAGES_WITH_SYSTEM_MESSAGE[:-1],
        tokenize=False,
        add_special_tokens=False,
        add_generation_prompt=True,
    )
    expected_str = (
        "This is the system message.\n\n"
        "### USER: This is the user message.\n"
        "### ASSISTANT:"
    )

    assert actual_str == expected_str
