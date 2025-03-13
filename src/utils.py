import random
import torch
import json
import torch.nn as nn
from typing import Callable


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


def simple_dict_logging_training_step_hook(step_info: dict):
    SimpleDictLogger.log_training_step(step_info)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
