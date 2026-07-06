#!/bin/bash
#SBATCH --job-name=boltz_w3
#SBATCH --output=slurm_logs/boltz_w3.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-gpu=48GB
#SBATCH --gres=gpu:1
#SBATCH --nodelist=inhccne0803
#SBATCH --time=30-00:00:00

cd /home/rottach/phd/p3_jepa/
. .boltz/bin/activate
cd boltz

for idx in $(seq 0 2000); do
    if [ ! -d "./boltz_results/yamls_$idx" ]; then
        echo "Running boltz predict for ./yamls/yamls_$idx"
        boltz predict "./yamls/yamls_$idx"
        echo "Cleanup: Deleting existing files in ./boltz_results/yamls_$idx"
        find "./boltz_results/yamls_$idx" -type f \( -name "*.npz" -o -name "*.pkl" -o -name "*.cif" \) -delete
    else
        echo "Directory ./boltz_results/yamls_$idx already exists."
    fi
done