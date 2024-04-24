#!/bin/bash

module load python/3.10 cuda/11.8

virtualenv "${SLURM_TMPDIR}/env"
source "${SLURM_TMPDIR}/env/bin/activate"

cd "${HOME}/workspace/AdversarialTriggers"
python3 -m pip install -e .

python3 -m pip install "vllm<=0.3.2"
python3 -m pip install "flash-attn>=2.2.0,<=2.4.2" --no-build-isolation
