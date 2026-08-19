#!/bin/bash
#SBATCH --job-name=moljepa_large
#SBATCH --output=slurm_logs/moljepa_large%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32GB
#SBATCH --gres=gpu:3
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=14-00:00:00

. ../../.jepa/bin/activate

# Get master node address for distributed communication
master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=bond0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$master_addr:8899 \
    train.py --config-name=moljepa_large trainer.num_nodes=$SLURM_JOB_NUM_NODES trainer.logger.version=moljepa_large
