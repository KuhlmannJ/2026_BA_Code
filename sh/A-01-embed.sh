#!/bin/bash

#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node

####
# H200 for 4B or 8B, takes ~22min with 53 Reports
####

#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8           # the number cpus per task
#SBATCH --partition=gpuh200         # on which partition to submit the job
#SBATCH --time=01:00:00             # the max wallclock time (time limit your job will run)

#SBATCH --output=/scratch/tmp/jkuhlma1/logs/%j_out.log    # stdout → Datei (%j = Job-ID)
#SBATCH --error=/scratch/tmp/jkuhlma1/logs/%j_err.log     # stderr → Datei

#SBATCH --job-name=A-01-embed-ColEmbed3Bv2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jannik.kuhlmann@uni-muenster.de 

# LOAD MODULES HERE IF REQUIRED
ml palma/2024a GCCcore/13.3.0 Python/3.12.3 CUDA/13.0.2
# ml uv/0.9.5

# SOURCE PYTHON VENV

# source $HOME/venvs/colpali-a100/bin/activate
source $HOME/venvs/colembed3Bv2-h200mini/bin/activate

# SET EXISTING CHACHE
export HF_HOME=$WORK/cache/huggingface
export CUDA_HOME=$EBROOTCUDA
export PIP_CACHE_DIR=$WORK/.cache/pip
# export UV_CACHE_DIR=/scratch/tmp/jkuhlma1/cache/uv

# pip install --upgrade pip
# pip install wheel torch transformers pymupdf Pillow python-dotenv datasets polars pydantic psutil accelerate torchvision 

## https://github.com/mjun0812/flash-attention-prebuild-wheels => Because bulidng them from scratch takes very very long time
# pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl
## Laut Nvidia die benutzte Verison
# pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.6.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl

## ALTERNATIV selbst bauen
# pip install flash-attn==2.6.3 --no-build-isolation



# START THE APPLICATION
# Set a MODEL_NAME flag '-3B' or '-4B' or '-8B'
python $HOME/2026_BA_Code/src/pipelines/pipelineA/A-01-embed-ColEmbed3Bv2.py -3B

pip freeze > $WORK/requirements/A-01-embed/3Bv2_requirements$(date +%m%d_%H%M).txt