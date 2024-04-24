#!/bin/bash

cd "/workspace/AdversarialTriggers"
python3 -m pip install -e .

python3 -m pip install "deepspeed<=0.13.1"
python3 -m pip install "vllm<=0.3.2"
python3 -m pip install "flash-attn>=2.2.0,<=2.4.2" --no-build-isolation
python3 -m pip uninstall -y "transformer-engine"

export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

"$@" --persistent_dir "${SCRATCH}/AdversarialTriggers" \
    > "${SCRATCH}/AdversarialTriggers/logs/${experiment_id}.out" \
    2> "${SCRATCH}/AdversarialTriggers/logs/${experiment_id}.err"
