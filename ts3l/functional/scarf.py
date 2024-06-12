from typing import Tuple
import torch
from torch import nn

def first_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward step of SCARF during the first phase

    Args:
        model (nn.Module): An instance of SCARF.
        batch (Tuple[torch.Tensor, torch.Tensor]): The input batch.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The anchor vector and the corrupted vector.
    """
    x, x_corrupted = batch
    emb_anchor, emb_corrupted = model(x, x_corrupted)
    return emb_anchor, emb_corrupted    


def first_phase_loss(
    emb_anchor: torch.Tensor, 
    emb_corrupted: torch.Tensor,
    contrastive_loss_fn: nn.Module
) -> torch.Tensor:
    """Calculate the first phase loss of SCARF

    Args:
        emb_anchor (torch.Tensor): The anchor vector.
        emb_corrupted (torch.Tensor): The corrupted vector.
        contrastive_loss_fn (nn.Module): The loss function for the contrastive learning.

    Returns:
        torch.Tensor: The contrastive loss between the anchor vector and the corrupted vector.
    """
    loss = contrastive_loss_fn(emb_anchor, emb_corrupted)
    return loss


def second_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step of SCARF during the second phase.

    Args:
        model (nn.Module): An instance of SCARF.
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
    """Calculate the second phase loss of SCARF.

    Args:
        y (torch.Tensor): The ground truth label.
        y_hat (torch.Tensor): The predicted label.
        task_loss_fn (nn.Module): The loss function for the given task.

    Returns:
        torch.Tensor: The loss for the given task.
    """
    return task_loss_fn(y_hat, y)
