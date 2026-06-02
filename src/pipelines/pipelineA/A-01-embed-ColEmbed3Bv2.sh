#!/bin/bash

#SBATCH --nodes=1                   # the number of nodes you want to reserve
#SBATCH --ntasks-per-node=1         # the number of tasks/processes per node

#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=4           # the number cpus per task
#SBATCH --partition=gpua100     # on which partition to submit the job
#SBATCH --time=00:15:00             # the max wallclock time (time limit your job will run)

#SBATCH --output=/scratch/tmp/jkuhlma1/logs/%j_out.log    # stdout → Datei (%j = Job-ID)
#SBATCH --error=/scratch/tmp/jkuhlma1/logs/%j_err.log     # stderr → Datei

#SBATCH --job-name=A-01-embed
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jannik.kuhlmann@uni-muenster.de 

# LOAD MODULES HERE IF REQUIRED
ml palma/2024a
ml GCCcore/13.3.0
ml Python/3.12.3
ml CUDA/13.0.2
ml uv/0.9.5

# SOURCE PYTHON VENV
# source $HOME/venvs/colpali-a100/bin/activate

# SET EXISTING CHACHE
export HF_HOME=$WORK/cache/huggingface
export UV_CACHE_DIR=/scratch/tmp/jkuhlma1/cache/uv

# START THE APPLICATION
uv run --no-build-isolation $HOME/2026_BA_Code/src/pipelines/pipelineA/A-01-embed-ColEmbed3Bv2.py