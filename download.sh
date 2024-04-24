#!/bin/bash

module load python/3.10 git-lfs/3.3.0 gcc/9.3.0 arrow/8 cuda/11.7

TMP_ENV=$(mktemp -d)
virtualenv --app-data "${SCRATCH}/virtualenv" --no-download "${TMP_ENV}/env"
source "${TMP_ENV}/env/bin/activate"

python3 -m pip install --no-index -U pip setuptools wheel
python3 -m pip install -U build

# Install package.
cd "${HOME}/workspace/AdversarialTriggers"
python3 -m pip install --no-index -e .

python3 -m pip install --no-index "flash-attn>=2.2.0,<=2.4.2" --no-build-isolation

# Download models and tokenizers.
cd "${SCRATCH}/AdversarialTriggers/checkpoint"
git clone https://huggingface.co/google/gemma-1.1-2b-it
git clone https://huggingface.co/google/gemma-1.1-7b-it
git clone https://huggingface.co/TheBloke/guanaco-7B-HF
git clone https://huggingface.co/TheBloke/guanaco-13B-HF
git clone https://huggingface.co/TheBloke/koala-7B-HF
git clone https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
git clone https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
git clone https://huggingface.co/meta-llama/LlamaGuard-7b
git clone https://huggingface.co/mosaicml/mpt-7b-chat
git clone https://huggingface.co/openchat/openchat_3.5
git clone https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha
git clone https://huggingface.co/Nexusflow/Starling-LM-7B-beta
git clone https://huggingface.co/lmsys/vicuna-7b-v1.5
git clone https://huggingface.co/lmsys/vicuna-13b-v1.5
