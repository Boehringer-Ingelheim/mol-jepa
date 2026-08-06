#!/bin/bash
#SBATCH --job-name=ablation
#SBATCH --output=slurm_logs/ablation%j.txt
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=30-00:00:00
#SBATCH --constraint=icelake 


. ../../.jepa/bin/activate

export version=param_01

echo "Run version: $version"
echo "GPU name: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
python train.py --config-path=config/ablations --config-name=$version trainer.logger.version=$version
