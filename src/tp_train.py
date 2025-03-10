# original code: https://github.com/pytorch/examples/blob/main/distributed/tensor_parallelism/tensor_parallel_example.py

# torchrun --nnodes=1 --nproc-per-node=2 src/tp_train.py

import os
import torch
import torch.nn as nn

import torch.distributed as dist
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
)
from torch.distributed._tensor.device_mesh import init_device_mesh
import utils


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(ToyModel, self).__init__()
        self.in_proj = nn.Linear(in_features, 2 * in_features)
        self.relu = nn.ReLU()
        self.out_proj = nn.Linear(2 * in_features, out_features)

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.out_proj(self.relu(self.in_proj(x)))


def train_setp(
    model: nn.Module,
    optimizer: torch.optim,
    inputs: torch.tensor,
    targets: torch.tensor,
    loss_fn: torch.nn,
    num_iters: int,
) -> None:
    print("Tensor Parallel training starting...")
    for i in range(num_iters):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"iter: {i + 1}/{num_iters} | Loss: {loss.item()}")


if __name__ == "__main__":
    # seed
    # For TP, input needs to be same across all TP ranks.
    # Setting the random seed is to mimic the behavior of dataloader.
    random_seed_number = 42
    utils.seed_everything(random_seed_number)

    # DDP setting
    device = "cuda"
    world_size = int(os.environ["WORLD_SIZE"])
    device_mesh = init_device_mesh(device_type=device, mesh_shape=(world_size,))
    rank = device_mesh.get_rank()
    print(f"Starting PyTorch TP example on rank {rank}.")
    assert world_size % 2 == 0, (
        f"TP examples require even number of GPUs, but got {world_size} gpus"
    )

    print(f"Device Mesh created: {device_mesh=}")

    # training setting
    batch_size = 3
    in_features = 5
    out_features = 3
    lr = 0.25
    num_iters = 10
    loss_fn = nn.MSELoss()

    inputs = torch.randn(batch_size, in_features).to(device)
    targets = torch.randn(batch_size, out_features).to(device)
    model = ToyModel(in_features, out_features).to(device)
    model.train()

    # Custom parallelization plan for the model
    tp_model = parallelize_module(
        module=model,
        device_mesh=device_mesh,
        parallelize_plan={
            "in_proj": ColwiseParallel(),
            "out_proj": RowwiseParallel(),
        },
    )
    optimizer = torch.optim.AdamW(tp_model.parameters(), lr=lr)

    train_setp(tp_model, optimizer, inputs, targets, loss_fn, num_iters)

    dist.destroy_process_group()
