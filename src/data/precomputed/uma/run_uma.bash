#!/bin/bash
#SBATCH --job-name=uma
#SBATCH --output=uma.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=30-00:00:00

. ../../../../../.jepa/bin/activate

python compute_uma_embeddings.py --csv data_with_benchmarks.csv --batch_size 16