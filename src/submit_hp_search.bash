#!/bin/bash
#SBATCH --job-name=search
#SBATCH --output=slurm_logs/search.txt
#SBATCH --time=30-00:00:00

. ../../.jepa/bin/activate
python train.py --config-name moljepa_hp