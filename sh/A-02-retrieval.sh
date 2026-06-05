#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mem=50G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuh200mini

#SBATCH --time=00:10:00

#SBATCH --output=/scratch/tmp/jkuhlma1/logs/%j_out.log    # stdout → Datei (%j = Job-ID)
#SBATCH --error=/scratch/tmp/jkuhlma1/logs/%j_err.log     # stderr → Datei

#SBATCH --job-name=A-02-retrieval-test
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

# pip install --upgrade pip
# pip install wheel torch transformers pymupdf Pillow python-dotenv datasets polars pydantic psutil accelerate torchvision 

## https://github.com/mjun0812/flash-attention-prebuild-wheels => Because bulidng them from scratch takes very very long time
# pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.8.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl
## Laut Nvidia die benutzte Verison
# pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.9.17/flash_attn-2.6.3+cu130torch2.12-cp312-cp312-linux_x86_64.whl

## ALTERNATIV selbst bauen
# pip install flash-attn==2.6.3 --no-build-isolation

# START THE APPLICATION
python $HOME/2026_BA_Code/src/pipelines/pipelineA/A-02-retrieval.py -t | tee $HOME/latest.out