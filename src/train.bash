#!/bin/bash
#SBATCH --job-name=train_one
#SBATCH --output=slurm_logs/train_one.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=30-00:00:00
#SBATCH --nodelist=inhccne1403

. ../../.jepa/bin/activate
python train.py

