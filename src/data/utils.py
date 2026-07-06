import os
import torch.distributed as dist

def barrier(processed_file_names):
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    else:
        import time

        while not processed_file_names[0].is_file():
            time.sleep(1)


def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    # torchrun sets RANK = global rank; fall back to NODE_RANK*LOCAL_WORLD_SIZE+LOCAL_RANK
    rank = os.environ.get("RANK")
    if rank is not None:
        return int(rank) == 0
    node_rank = int(os.environ.get("NODE_RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    return node_rank == 0 and local_rank == 0