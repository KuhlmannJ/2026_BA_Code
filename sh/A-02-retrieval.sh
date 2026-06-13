#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuh200mini  # Is enough as VRAM peak = 29GB for 1,5 min with 22GB RAM

#SBATCH --time=00:05:00

#SBATCH --output=/scratch/tmp/jkuhlma1/logs/%j_out.log    # stdout → Datei (%j = Job-ID)
#SBATCH --error=/scratch/tmp/jkuhlma1/logs/%j_out.log     # stderr → Datei

#SBATCH --job-name=A-02-retrieval-8B
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jannik.kuhlmann@uni-muenster.de 

# LOAD MODULES
ml palma/2024a GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.2

# SOURCE PYTHON VENV
source $HOME/venvs/colembed3Bv2-h200mini/bin/activate

# SET EXISTING CHACHE
export HF_HOME=$WORK/cache/huggingface
export CUDA_HOME=$EBROOTCUDA
export PIP_CACHE_DIR=$WORK/.cache/pip
export RUN_TS=$(date +%m%d_%H%M) # To map the only the new retirevals

# pip install --upgrade pip
# pip install wheel torch transformers pymupdf Pillow python-dotenv datasets polars pydantic psutil accelerate torchvision pypdf cryptography pandas

## https://github.com/mjun0812/flash-attention-prebuild-wheels => Because bulidng them from scratch takes very very long time
# pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl
## Laut Nvidia die benutzte Verison
# pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.6.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl

## ALTERNATIV selbst bauen
# pip install flash-attn==2.6.3 --no-build-isolation

# START THE APPLICATION
# Set a MODEL_NAME flag '-3B' or '-4B' or '-8B' when calling A-02-retrieval.sh
# Set a MODE flag '-t' or '-gt' or '-q' or '-a' when calling A-02-retrieval.sh
MODEL_FLAG=${1:--4B}
MODE="$2"
ARGS=("$MODEL_FLAG")
[[ -n "$MODE" ]] && ARGS+=("$MODE")

python -u "$HOME/2026_BA_Code/src/pipelines/pipelineA/A-02-retrieval.py" "${ARGS[@]}"

python -u "$HOME/2026_BA_Code/evaluations/A-02/A-02.py"

mkdir -p $WORK/requirements/A-02-retrieval
pip freeze > $WORK/requirements/A-02-retrieval/requirements$(date +%m%d_%H%M).txt