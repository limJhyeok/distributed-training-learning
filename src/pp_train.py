import torch
import os
import torch.distributed as dist
from torch.distributed.pipelining import ScheduleGPipe
import config
import utils
import models


if __name__ == "__main__":
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = (
        torch.device(f"cuda:{rank}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # This group can be a sub-group in the N-D parallel case
    pp_group = dist.new_group()
    stage_index = rank
    num_stages = world_size
    model_args = config.ModelArgs()

    n_microbatches = 4
    batch_size = 32

    assert batch_size % n_microbatches == 0, (
        f"Batch size must be divisible by the number of microbatches.\n"
        f"Given: batch_size={batch_size}, n_microbatches={n_microbatches}"
    )
    micro_batch_size = batch_size // n_microbatches

    example_input_microbatch = torch.ones(
        micro_batch_size, model_args.dim, dtype=torch.long
    )
    stage = utils.create_pipeline_stage(
        models.Transformer,
        model_args,
        stage_index,
        num_stages,
        device,
        example_input_microbatch,
    )
    dist.barrier()

    # Create a pipeline schedule(All Forward All Backward)
    schedule = ScheduleGPipe(stage, n_microbatches)

    # Input data (whole batch)
    x = torch.ones(batch_size, model_args.dim, dtype=torch.long)
    x = x.to(device)

    dist.barrier()

    # Run the pipeline with input `x`
    # `x` will be divided into microbatches automatically
    if rank == 0:
        schedule.step(x)
    else:
        output = schedule.step()
        print(f"output_size: {output.size()}")

    dist.destroy_process_group()
