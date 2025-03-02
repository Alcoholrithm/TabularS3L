from typing import Tuple
import torch
from torch import nn


def first_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step of TabularBinning during the first phase

    Args:
        model (nn.Module): An instance of TabularBinning.
        batch (Tuple[torch.Tensor, torch.Tensor]): The input batch.

    Returns:
        torch.Tensor: The predicted bins of the given batch.
    """
    x, _ = batch
    bin_preds = model(x)
    return bin_preds


def first_phase_loss(
    bins: torch.Tensor,
    bin_preds: torch.Tensor,
    bin_loss_fn: nn.Module
) -> torch.Tensor:
    """Calculate the first phase loss of TabularBinning

    Args:
        bins (torch.Tensor): The original bin tensor.
        bins_preds (torch.Tensor): The predicted bin tensor.
        bin_loss_fn (nn.Module): The loss function for the first phase learning of TabularBinning.

    Returns:
        torch.Tensor: The loss between the predicted bins and the original bins.
    """
    loss = bin_loss_fn(bin_preds, bins)
    return loss


def second_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step of TabularBinning during the second phase.

    Args:
        model (nn.Module): An instance of TabularBinning.
        batch (Tuple[torch.Tensor, torch.Tensor]): The input batch.

    Returns:
        torch.Tensor: The predicted label (logit).
    """
    x, _ = batch
    y_hat = model(x).squeeze()
    return y_hat


def second_phase_loss(
    y: torch.Tensor, y_hat: torch.Tensor, task_loss_fn: nn.Module
) -> torch.Tensor:
    """Calculate the second phase loss of TabularBinning.

    Args:
        y (torch.Tensor): The ground truth label.
        y_hat (torch.Tensor): The predicted label.
        task_loss_fn (nn.Module): The loss function for the given task.

    Returns:
        torch.Tensor: The loss for the given task.
    """
    return task_loss_fn(y_hat, y)
