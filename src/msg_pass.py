# torchrun --nnodes=1 --nproc-per-node=2 src/msg_pass.py

import os
import torch
import torch.distributed as dist


def init_process(rank: int, world_size: int) -> None:
    """
    Initializes the distributed process group.

    This function sets up the distributed environment.
    `init_process_group`, specifying the backend, process rank, and world size.

    Args:
        rank (int): The rank (ID) of the current process.
        world_size (int): The total number of processes in the distributed group.

    Returns:
        None
    """
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)


def run(rank: int, world_size: int) -> None:
    """
    Implements a simple point-to-point communication.

    - Rank 0 sends a tensor containing the value 42 to all other ranks.
    - Other ranks receive the tensor from rank 0 and print the received value.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes in the distributed group.

    Returns:
        None
    """
    tensor = torch.zeros(1)
    if rank == 0:
        msg = torch.tensor([42], dtype=torch.float32)
        # Send the tensor to other processes(except for 0)
        for i in range(1, world_size):
            dist.send(tensor=msg, dst=1)
            print(f"Rank 0 send to Rank {i}")
    else:
        # Receive tensor from process 0
        dist.recv(tensor=tensor, src=0)
        print(f"Rank {rank} received message: {tensor.item()}")


if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    init_process(rank, world_size)
    run(rank, world_size)

    dist.destroy_process_group()
