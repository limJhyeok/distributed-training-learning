# torchrun --nnodes=1 --nproc-per-node=2 src/ddp_train.py

import os
import json
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
from model import ToyModel
import utils


def demo_ddp(rank: int) -> None:
    simple_logger = {}

    # create model and move it to GPU with id rank
    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()

    # set random seed to make different torch.randn by each rank
    utils.seed_everything(rank)
    inputs = torch.randn(3, 5).to(rank)

    simple_logger["inputs"] = inputs.cpu().numpy().tolist()

    outputs = ddp_model(inputs).to(rank)
    labels = torch.randn(3, 5).to(rank)

    simple_logger["labels"] = labels.cpu().numpy().tolist()
    loss = loss_fn(outputs, labels)

    loss.backward()

    simple_logger["loss"] = loss.cpu().detach().numpy().tolist()

    optimizer.step()

    # Gradient
    for name, param in model.named_parameters():
        simple_logger[f"{name}'s gradient"] = param.grad.cpu().numpy().tolist()

    # After Update
    for name, param in model.named_parameters():
        simple_logger[f"{name}'s updated weights"] = (
            param.cpu().detach().numpy().tolist()
        )

    # Write results
    dir_name = "logs"
    filename = f"ddp_rank-{rank}.json"
    with open(os.path.join(dir_name, filename), "w") as f:
        json.dump(simple_logger, f, indent=4)

    print(f"Finished running basic DDP example on rank {rank}.")


if __name__ == "__main__":
    # setup
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Running basic DDP example on rank {rank}.")

    # ddp forward & backward
    demo_ddp(rank)

    # free
    dist.destroy_process_group()
