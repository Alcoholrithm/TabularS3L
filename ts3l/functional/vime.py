from typing import Tuple, Dict, Any
import torch
from torch import nn

def first_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Forward step of VIME during the first phase

    Args:
        model (nn.Module): An instance of VIME
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The predicted mask vector and the predicted feature vector
    """
    mask_preds, feature_preds = model(batch[0])
    return mask_preds, feature_preds

def first_phase_loss(
    x_cat: torch.Tensor,
    x_cont: torch.Tensor,
    mask: torch.Tensor,
    cat_feature_preds: torch.Tensor,
    cont_feature_preds: torch.Tensor,
    mask_preds: torch.Tensor,
    mask_loss_fn: nn.Module,
    categorical_loss_fn: nn.Module,
    continuous_loss_fn: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the first phase loss of VIME

    Args:
        x_cat (torch.Tensor): The categorical input feature vector
        x_cont (torch.Tensor): The continuous input feature vector
        mask (torch.Tensor): The ground truth mask vector
        cat_feature_preds (torch.Tensor): The predicted categorical feature vector
        cont_feature_preds (torch.Tensor): The predicted continuous feature vector
        mask_preds (torch.Tensor): The predicted mask vector
        mask_loss_fn (nn.Module): The loss function for the mask estimation
        categorical_loss_fn (nn.Module): The loss function for the categorical feature reconstruction
        continuous_loss_fn (nn.Module): The loss function for the continuous feature reconstruction

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The losses for mask estimation and feature reconstruction
    """
    mask_loss = mask_loss_fn(mask_preds, mask)
    categorical_feature_loss = torch.tensor(0.0)
    continuous_feature_loss = torch.tensor(0.0)
    
    if x_cat.shape[1] > 0:
        categorical_feature_loss += categorical_loss_fn(cat_feature_preds, x_cat)
    if x_cont.shape[1] > 0:
        continuous_feature_loss += continuous_loss_fn(cont_feature_preds, x_cont)

    return mask_loss, categorical_feature_loss, continuous_feature_loss

def second_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step of VIME during the second phase

    Args:
        model (nn.Module): An instance of VIME
        batch (Tuple[torch.Tensor, torch.Tensor]): The input batch

    Returns:
        torch.Tensor: The predicted label (logit)
    """
    x, _ = batch
    return model(x).squeeze()

def second_phase_loss(
    y: torch.Tensor, y_hat: torch.Tensor, consistency_loss_fn: nn.Module, loss_fn: nn.Module, u_label: Any, consistency_len: int, K: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calculate the second phase loss of VIME

    Args:
        y (torch.Tensor): The ground truth label
        y_hat (torch.Tensor): The predicted label
        loss_fn (nn.Module): The loss function for the given task

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: The losses for the given task and consistency regularization and the ground truth labels
    """
    consistency_loss = torch.tensor(0.0)
    labeled_idx = (y != u_label).flatten()
    unlabeled_idx = (y == u_label).flatten()

    unlabeled = y_hat[unlabeled_idx]
    if len(unlabeled) > 0:
        target = unlabeled[::consistency_len]
        target = target.repeat(1, K).reshape((-1, unlabeled.shape[-1]))
        preds = torch.stack([unlabeled[i, :] for i in range(len(unlabeled)) if i % consistency_len != 0], dim = 0)
        consistency_loss += consistency_loss_fn(preds, target)
    
    labeled_y = y[labeled_idx].squeeze()
    task_loss = loss_fn(y_hat[labeled_idx], labeled_y)
    
    
    return task_loss, consistency_loss, labeled_y