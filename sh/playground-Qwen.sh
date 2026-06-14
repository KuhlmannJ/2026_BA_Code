#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

#SBATCH --mem=150G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpuh200

#SBATCH --time=00:05:00

#SBATCH --output=/scratch/tmp/jkuhlma1/logs/%j_out.log    # stdout → Datei (%j = Job-ID)
#SBATCH --error=/scratch/tmp/jkuhlma1/logs/%j_out.log     # stderr → Datei

#SBATCH --job-name=Qwen_Playground
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

python -u "$HOME/2026_BA_Code/src/playground/HPC-01-Playground/Qwen.py"