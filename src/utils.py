import random
import torch
import json
import torch.nn as nn
from typing import Callable
from torch.distributed.pipelining import PipelineStage
import config


class SimpleDictLogger:
    logger = {}

    @staticmethod
    def log(info: dict) -> None:
        for k, v in info.items():
            SimpleDictLogger.logger[k] = v

    @staticmethod
    def log_training_step(step_info: dict) -> None:
        iter_num = step_info["iteration"]
        SimpleDictLogger.logger[f"iter: {iter_num}"] = {
            "loss": step_info["loss"],
            "lr": step_info["optimizer"].param_groups[0]["lr"],
        }
        model = step_info["model"]
        # gradient log
        for name, param in model.named_parameters():
            SimpleDictLogger.logger[f"iter: {iter_num}"][f"{name}'s gradient"] = (
                param.grad.cpu().numpy().tolist()
            )

        # updated weights log
        for name, param in model.named_parameters():
            SimpleDictLogger.logger[f"iter: {iter_num}"][
                f"{name}'s updated weights"
            ] = param.cpu().detach().numpy().tolist()

    @staticmethod
    def save(path: str) -> None:
        with open(path, "w") as f:
            json.dump(SimpleDictLogger.logger, f, indent=4)
        print(f"[Logger] Training log saved to {path}")


def train_step(
    model: nn.Module,
    optimizer: torch.optim,
    inputs: torch.tensor,
    targets: torch.tensor,
    loss_fn: torch.nn,
    num_iters: int,
    hooks: list[Callable[[dict], None]] = [],
) -> None:
    print("training starting...")
    for i in range(num_iters):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()
        print(f"iter: {i + 1}/{num_iters} | Loss: {loss.item()}")

        step_info = {
            "iteration": i + 1,
            "num_iters": num_iters,
            "loss": loss.item(),
            "model": model,
            "optimizer": optimizer,
        }

        for hook in hooks:
            hook(step_info)


def create_pipeline_stage(
    model_architecture: nn.Module,
    model_args: config.ModelArgs,
    stage_index: int,
    num_stages: int,
    device: torch.device,
    example_input: torch.Tensor,
):
    """
    Creates a pipeline stage by constructing the full model in torch meta device and keeping only the relevant layers for the given stage.

    * c.f) "meta" device
        | This significantly reduces memory usage and enables flexible model partitioning.
        | for more information, see: https://pytorch.org/docs/stable/meta.html
        | Using the "meta" device allows us to define the model structure without allocating memory for parameters.

    Args:
        model_architecture (nn.Module): Model architecture to build following ModelArgs.
        model_args (ModelArgs): Model configuration parameters.
        stage_index (int): Index of the current stage in the pipeline.
        num_stages (int): Total number of pipeline stages.
        device (torch.device): Device on which to place the model.
        example_input (torch.Tensor): Example input tensor for defining the stage.

    Returns:
        PipelineStage: The configured pipeline stage.
    """
    assert num_stages == 2, "This function is designed for a simple 2-stage pipeline."

    with torch.device("meta"):
        model = model_architecture(model_args)

        if stage_index == 0:
            # Retain only the first half of the layers for stage 0.
            for i in range(4, 8):
                del model.layers[str(i)]
            model.norm = None  # Stage 0 does not require LayerNorm
            model.output = None  # Output layer is only needed in the final stage
        elif stage_index == 1:
            # Retain only the second half of the layers for stage 1.
            model.tok_embeddings = None  # Token embedding is only needed in stage 0
            for i in range(4):
                del model.layers[str(i)]

        # Use .to_empty() to materialize the model on the actual device (from meta) without initializing parameters.
        return PipelineStage(
            model.to_empty(device=device),
            stage_index,
            num_stages,
            device,
            input_args=example_input,  # Fake input to initialize the pipeline stage
        )


def simple_dict_logging_training_step_hook(step_info: dict):
    SimpleDictLogger.log_training_step(step_info)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
