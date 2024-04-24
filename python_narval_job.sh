#!/bin/bash
#SBATCH --account=ctb-timod
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00

module load python/3.10 gcc/9.3.0 arrow/8 cuda/11.7

TMP_ENV=$(mktemp -d)
virtualenv --app-data "${SCRATCH}/virtualenv" --no-download "${TMP_ENV}/env"
source "${TMP_ENV}/env/bin/activate"

python3 -m pip install --no-index -U pip setuptools wheel

cd "${HOME}/workspace/AdversarialTriggers"
python3 -m pip install --no-index -e .

python3 -m pip install --no-index "vllm<=0.3.2"
python3 -m pip install --no-index "flash-attn>=2.2.0,<=2.4.2" --no-build-isolation

export TRANSFORMERS_OFFLINE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

"$@" --persistent_dir "${SCRATCH}/AdversarialTriggers"
