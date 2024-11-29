from typing import Tuple, Any
import torch
from torch import nn


def first_phase_step(
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    u_label: Any = -1,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Forward step of SwitchTab during the first phase.

    Args:
        model (nn.Module): An instance of SwitchTab.
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch.
        u_label (Any): The specifier for unlabeled sample. Defaults to -1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The reconstructed input vector, the ground truth label vector, and the predicted label vector for the labeled samples.
    """

    _, xcs, ys = batch
    
    cls_idx = ys != u_label

    x_hat, y_hat = model(xcs)

    labeled_y_hat, labeled_y = y_hat[cls_idx.squeeze(), :].squeeze(), ys[cls_idx]

    return x_hat, labeled_y, labeled_y_hat


def first_phase_loss(
    xs: torch.Tensor,
    x_hat: torch.Tensor,
    labeled_y: torch.Tensor,
    labeled_y_hat: torch.Tensor,
    reconstruction_loss_fn: nn.Module,
    task_loss_fn: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the first phase loss of SwitchTab

    Args:
        xs (torch.Tensor): The input vector.
        x_hat (torch.Tensor): The reconstructed input vector.
        labeled_y (torch.Tensor): The ground truth labels.
        labeled_y_hat (torch.Tensor): The predicted labels for the labeled samples.
        reconstruction_loss_fn (nn.Module): The loss function for the feature reconstruction.
        task_loss_fn (nn.Module): The loss function for the given task.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The reconstruction loss and the task loss.
    """

    recon_loss = reconstruction_loss_fn(x_hat, xs)

    task_loss = torch.tensor(0, device=xs.device)
    if len(labeled_y) > 0:
        task_loss = task_loss_fn(labeled_y_hat, labeled_y)

    return recon_loss, task_loss


def second_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step of SwitchTab during the second phase.

    Args:
        model (nn.Module): An instance of SwitchTab.
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
    """Calculate the second phase loss of SwitchTab.

    Args:
        y (torch.Tensor): The ground truth label.
        y_hat (torch.Tensor): The predicted label.
        task_loss_fn (nn.Module): The loss function for the given task.

    Returns:
        torch.Tensor: The loss for the given task.
    """
    return task_loss_fn(y_hat, y)
