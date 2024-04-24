# We define templates and system messages for various models below. These templates
# are written in Jinja2 and are used to convert a list of messages into a single
# string.
#
# For more information on Hugging Face chat templates, see:
# https://huggingface.co/docs/transformers/main/en/chat_templating

from adversarial_triggers.experiment_utils import (
    get_short_model_name,
    parse_experiment_id,
)

CHAT_ML_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ raise_exception('First message must be a system message.') }}"
    "{% endif %}"
    "{% set system_message = messages[0]['content'] %}"
    "{% set loop_messages = messages[1:] %}"
    "{% for message in loop_messages %}"
    "{% if loop.index0 == 0 %}"
    "{{ '<|im_start|>system\n' + system_message.strip() + '\n'}}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + ' ' + '<|im_end|>' }}"
    "{% else %}"
    "{{ '\n' + '<|im_start|>' + message['role'] + '\n' + message['content'] + ' ' + '<|im_end|>' }}"
    "{% endif %}"
    "{% if add_generation_prompt %}"
    "{{ '\n' + '<|im_start|>' + 'assistant' + '\n' }}"
    "{% elif (message['role'] == 'assistant') %}"
    "{{ eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)

GEMMA_TEMPLATE = (
    # Although Gemma doesn't support system messages, we expect the first message to
    # still be a dummy system message. This system message is not included in the
    # resulting string. This is to make the templates consistent across models.
    "{{ bos_token }}"
    "{% if messages[0]['role'] != 'system' %}"
    "{{ raise_exception('First message must be a system message.') }}"
    "{% endif %}"
    "{% set loop_messages = messages[1:] %}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if (message['role'] == 'assistant') %}"
    "{% set role = 'model' %}"
    "{% else %}"
    "{% set role = message['role'] %}"
    "{% endif %}"
    "{% if role == 'model' %}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + ' ' + eos_token + '\n' }}"
    "{% else %}"
    "{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + ' ' + '<end_of_turn>' + '\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{'<start_of_turn>model\n'}}"
    "{% endif %}"
)

LLAMA_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"  # Extract system message.
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{{ raise_exception('First message must be a system message.') }}"
    "{% endif %}"
    "{% for message in loop_messages %}"  # Loop over all non-system messages.
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 %}"  # Embed system message in first message.
    "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}"
    "{% else %}"
    "{% set content = message['content'] %}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way.
    "{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' '  + content.strip() + ' ' + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)

SAFERPACA_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"  # Extract system message.
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{{ raise_exception('First message must be a system message.') }}"
    "{% endif %}"
    "{% for message in loop_messages %}"  # Loop over all non-system messages.
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if loop.index0 == 0 %}"  # Embed system message in first message.
    "{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + 'Below is an instruction that describes a task. Write a response that appropriately completes that request.\\n\\n### Instruction:\\n' + message['content'] + ' \\n\\n### Response:' %}"
    "{% else %}"
    "{% set content = message['content'] %}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"  # After all of that, handle messages/roles in a fairly normal way.
    "{{ bos_token + '[INST] ' +  content.strip() + ' [/INST]' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ ' '  + content.strip() + ' ' + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
)

OPENCHAT_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ raise_exception('First message must be a system message.') }}"
    "{% endif %}"
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'assistant' %}"
    "{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + ' ' + eos_token }}"
    "{% else %}"
    "{{ 'GPT4 Correct ' + message['role'].title() + ': ' + message['content'] + ' ' + '<|end_of_turn|>' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ 'GPT4 Correct Assistant:' }}"
    "{% endif %}"
)

VICUNA_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ raise_exception('First message must be a system message.') }}"
    "{% endif %}"
    "{% set loop_messages = messages[1:] %}"  # Extract system message.
    "{% set system_message = messages[0]['content'] %}"
    "{{ system_message + '\\n\\n' }}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"  # Loop over all non-system messages.
    "{{ '### USER: ' + message['content'] + '\\n' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '### ASSISTANT: ' + message['content'] + ' ' + eos_token + '\\n' }}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '### ASSISTANT:' }}"
    "{% endif %}"
    "{% endfor %}"
)


KOALA_TEMPLATE = (
    "{% if messages[0]['role'] != 'system' %}"
    "{{ raise_exception('First message must be a system message.') }}"
    "{% endif %}"
    "{% set loop_messages = messages[1:] %}"  # Skip system message.
    "{{ 'BEGINNING OF CONVERSATION: ' }}"
    "{% for message in loop_messages %}"
    "{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}"
    "{% endif %}"
    "{% if message['role'] == 'user' %}"  # Loop over all non-system messages.
    "{{ 'USER: ' + message['content'] + ' ' }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ 'GPT: ' + message['content'] + ' ' + eos_token + ' ' }}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ 'GPT:' }}"
    "{% endif %}"
    "{% endfor %}"
)


ZEPHYR_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{% set loop_messages = messages[1:] %}"  # Extract system message.
    "{% set system_message = messages[0]['content'] %}"
    "{% else %}"
    "{{ raise_exception('First message must be a system message.') }}"
    "{% endif %}"
    "{{ '<|system|>\n' + system_message + eos_token }}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ '<|user|>\n' + message['content'] + eos_token }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ '<|assistant|>\n'  + message['content'] + eos_token }}"
    "{% endif %}"
    "{% if loop.last and add_generation_prompt %}"
    "{{ '<|assistant|>' }}"
    "{% endif %}"
    "{% endfor %}"
)


_CHAT_TEMPLATE_MAP = {
    "gemma-1.1-2b-it": GEMMA_TEMPLATE,
    "gemma-1.1-7b-it": GEMMA_TEMPLATE,
    "guanaco-7B-HF": VICUNA_TEMPLATE,
    "guanaco-13B-HF": VICUNA_TEMPLATE,
    "koala-7B-HF": KOALA_TEMPLATE,
    "Llama-2-7b-chat-hf": LLAMA_TEMPLATE,
    "Llama-2-13b-chat-hf": LLAMA_TEMPLATE,
    "mpt-7b-chat": CHAT_ML_TEMPLATE,
    "openchat_3.5": OPENCHAT_TEMPLATE,
    "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-dromedary_l-2e-05_s-0": LLAMA_TEMPLATE,
    "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-saferpaca2000_l-2e-05_s-0": SAFERPACA_TEMPLATE,
    "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-sharegpt_l-2e-05_s-0": LLAMA_TEMPLATE,
    "sft_m-Llama-2-7b-hf_i-Llama-2-7b-chat-hf_d-lima_l-2e-05_s-0": LLAMA_TEMPLATE,
    "Starling-LM-7B-alpha": OPENCHAT_TEMPLATE,
    "Starling-LM-7B-beta": OPENCHAT_TEMPLATE,
    "vicuna-7b-v1.5": VICUNA_TEMPLATE,
    "vicuna-13b-v1.5": VICUNA_TEMPLATE,
}


def load_chat_template(model_name_or_path: str) -> str:
    """Loads chat template for a given model.

    Args:
        model_name_or_path: Model name or path. The chat template is inferred from
            this.

    Returns:
        Chat template string. This can be set as the `chat_template` attribute
        of a tokenizer.

    Usage:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> chat_template = load_chat_template("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer.chat_template = chat_template
    """
    short_model_name = get_short_model_name(model_name_or_path)

    if short_model_name in _CHAT_TEMPLATE_MAP:
        chat_template = _CHAT_TEMPLATE_MAP[short_model_name]
    else:
        raise ValueError(
            f"Could not find chat template for model {model_name_or_path}."
        )

    return chat_template
