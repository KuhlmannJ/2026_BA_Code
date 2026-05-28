#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpuh200mini
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/tmp/jkuhlma1/logs/%j_out.log
#SBATCH --error=/scratch/tmp/jkuhlma1/logs/%j_err.log
#SBATCH --job-name=freeze-venv-h200

ml palma/2024a
ml GCCcore/13.3.0
ml Python/3.12.3

source $HOME/venvs/colpali-20260517/bin/activate
pip freeze > $HOME/requirements-colpali.txt

echo "Done. requirements-colpali.txt gespeichert."