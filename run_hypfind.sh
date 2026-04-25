#!/bin/bash

#SBATCH --time=11:00:00

#SBATCH --mem=48G

#SBATCH --cpus-per-task=6

#SBATCH --partition=gpu_h200

#SBATCH --gpus=1

#SBATCH --mail-type=ALL



module load miniconda

conda init

conda activate torch311

python run_hp_search.py --seed 203 --screening-fold 0 --n-configs 500 --phase2 --top-k 100

