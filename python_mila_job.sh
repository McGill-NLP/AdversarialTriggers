#!/bin/bash
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100l:1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00

module load python/3.10 cuda/11.8

virtualenv "${SLURM_TMPDIR}/env"
source "${SLURM_TMPDIR}/env/bin/activate"

cd "${HOME}/workspace/AdversarialTriggers"
python3 -m pip install -e .

python3 -m pip install "vllm<=0.3.2"
python3 -m pip install "flash-attn>=2.2.0,<=2.4.2" --no-build-isolation

export TRANSFORMERS_NO_ADVISORY_WARNINGS=true

"$@" --persistent_dir "${SCRATCH}/AdversarialTriggers"
