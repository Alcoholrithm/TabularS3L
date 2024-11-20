from typing import Tuple, Dict, Any
import torch
from torch import nn

def get_duplicated_input(x: torch.Tensor, n_subsets: int) -> torch.Tensor:
    """Duplicates the input tensor across the batch dimension to match the number of subsets for reconstruction loss.

    Args:
        x (torch.Tensor): The input tensor to duplicate.
        n_subsets (int): The number of subsets for each sample.

    Returns:
        torch.Tensor: A tensor with the input 'n_subsets' times along the batch dimension.
    """
    x_duplicated = x
    for _ in range(1, n_subsets):
        x_duplicated = torch.concat((x_duplicated, x), dim = 0)
    return x_duplicated

def arrange_tensors(x: torch.Tensor, n_subsets: int) -> torch.Tensor:
    """
    Rearranges the input tensor into a sequence of subsets concatenated along the first dimension.

    This function reshapes the input tensor 'x' of shape '(no, dim)' into a tensor of shape 
    '(n_subsets, samples, dim)', where 'no' is the total number of tensors, 'dim' is the feature 
    dimension, and 'samples' is the batch size (calculated as 'no // n_subsets'). 
    The reshaped tensor is then split into subsets, which are concatenated along the first dimension to produce 
    the output tensor.

    Args:
        projections (torch.Tensor): A tensor of shape '(no, dim)' where 'no' is the total number of tensors and 'dim' is the feature dimension. 
        n_subsets (int): The number of subsets for each sample.

    Returns:
        torch.Tensor: A reshaped tensor where subsets of tensors are concatenated along the first dimension. 
                        The resulting shape is '(no, dim)', maintaining the original dimensionality 
                        but rearranging the order of observations to align with subset divisions.
    """
    no, dim = x.shape
    samples = int(no / n_subsets)
    
    x = x.reshape((n_subsets, samples, dim))
    
    return torch.concat([x[:, i] for i in range(samples)])
    
def first_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward step of SubTab during the first phase.

    Args:
        model (nn.Module): An instance of SubTab.
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The projections of each subset and the reconstructed input feature vectors.
    """

    x, _ = batch
    
    projections, x_recons = model(x)
    
    return projections, x_recons


def first_phase_loss(
    projections: torch.Tensor, 
    x_recons: torch.Tensor, 
    x_originals: torch.Tensor, 
    n_subsets: int, 
    joint_loss_fn: nn.Module
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the first phase loss of SubTab.

    Args:
        projections (torch.Tensor): The projections of each subset.
        x_recons (torch.Tensor): The reconstructed input tensor from the each subset.
        x_originals (torch.Tensor): The original input tensors for the reconstruction loss.
        n_subsets (int): The number of subsets for each sample.
        joint_loss_fn (nn.Module): The joint loss function for SubTab during the first phase learning.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The total loss, contrastive loss, reconstruction loss, 
                                                                        and distance loss during the first phase of learning.
    """
    
    x_originals = get_duplicated_input(x_originals, n_subsets)
    
    arranged_projections = arrange_tensors(projections, n_subsets)
    
    total_loss, contrastive_loss, recon_loss, dist_loss = joint_loss_fn(arranged_projections, x_recons, x_originals)
    
    return total_loss, contrastive_loss, recon_loss, dist_loss


def second_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step of SubTab during the second phase.

    Args:
        model (nn.Module): An instance of SubTab.
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch.

    Returns:
        torch.Tensor: The predicted label (logit).
    """
    x, _ = batch
    return model(x).squeeze()

def second_phase_loss(
    y: torch.Tensor,
    y_hat: torch.Tensor,
    task_loss_fn: nn.Module,
) -> torch.Tensor:
    """Calculate the second phase loss of SubTab

    Args:
        y (torch.Tensor): The ground truth label.
        y_hat (torch.Tensor): The predicted label.
        task_loss_fn (nn.Module): The loss function for the given task.
    Returns:
        torch.Tensor: The losse for the given task.
    """

    task_loss = task_loss_fn(y_hat, y)
    
    return task_loss
