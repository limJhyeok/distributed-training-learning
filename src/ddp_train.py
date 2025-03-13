# torchrun --nnodes=1 --nproc-per-node=2 src/ddp_train.py

import os
import torch.distributed as dist
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim
import models
import utils


if __name__ == "__main__":
    # distirbuted training setup
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    print(f"Running basic DDP example on rank {rank}.")

    # training setting
    batch_size = 3
    in_features = 5
    out_features = 5
    lr = 0.25
    num_iters = 1
    loss_fn = nn.MSELoss()

    # set random seed to make different torch.randn by each rank
    utils.seed_everything(rank)
    inputs = torch.randn(3, 5).to(rank)
    targets = torch.randn(3, 5).to(rank)

    # log
    data_log = {
        "inputs": inputs.cpu().numpy().tolist(),
        "targets": targets.cpu().numpy().tolist(),
    }
    utils.SimpleDictLogger.log(data_log)

    model = models.ToyModel(in_features, out_features).to(rank)
    model.train()
    ddp_model = DDP(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=lr)

    hooks = [utils.simple_dict_logging_training_step_hook]

    utils.train_step(
        ddp_model, optimizer, inputs, targets, loss_fn, num_iters, hooks=hooks
    )

    # log save
    dir_name = "logs"
    filename = f"ddp_rank-{rank}.json"
    utils.SimpleDictLogger.save(os.path.join(dir_name, filename))

    # free
    dist.destroy_process_group()
