from typing import Tuple, List
import torch
from torch import nn

def first_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
    """Forward step of Denoising AutoEncoder during the first phase

    Args:
        model (nn.Module): An instance of Denoising AutoEncoder
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

    Returns:
        Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]: The predicted mask vector and the predicted feature vector
    """
    _, x_bar, _ = batch
    mask_preds, cat_preds, cont_preds = model(x_bar)
    return mask_preds, cat_preds, cont_preds


def first_phase_loss(
    x_cat: torch.Tensor,
    x_cont: torch.Tensor,
    mask: torch.Tensor,
    cat_feature_preds: List[torch.Tensor],
    cont_feature_preds: torch.Tensor,
    mask_preds: torch.Tensor,
    mask_loss_fn: nn.Module,
    categorical_loss_fn: nn.Module,
    continuous_loss_fn: nn.Module,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the first phase loss of DAE

    Args:
        x_cat (torch.Tensor): The categorical input feature vector
        x_cont (torch.Tensor): The continuous input feature vector
        mask (torch.Tensor): The ground truth mask vector
        cat_feature_preds (List[torch.Tensor]): The predicted categorical feature vector
        cont_feature_preds (torch.Tensor): The predicted continuous feature vector
        mask_preds (torch.Tensor): The predicted mask vector
        mask_loss_fn (nn.Module): The loss function for the mask estimation
        categorical_loss_fn (nn.Module): The loss function for the categorical feature reconstruction
        continuous_loss_fn (nn.Module): The loss function for the continuous feature reconstruction

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The losses for mask estimation and feature reconstruction
    """
    mask_loss = mask_loss_fn(mask_preds, mask)
    feature_loss = torch.tensor(0.0, device=mask_preds.device)
    
    if x_cat.shape[1] > 0:
        for idx in range(x_cat.shape[1]):
            feature_loss += categorical_loss_fn(cat_feature_preds[idx], x_cat[:, idx].long())
    if x_cont.shape[1] > 0:
        feature_loss += continuous_loss_fn(cont_feature_preds, x_cont)

    return mask_loss, feature_loss


def second_phase_step(
    model: nn.Module, batch: Tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    """Forward step of Denoising AutoEncoder during the second phase

    Args:
        model (nn.Module): An instance of Denoising AutoEncoder
        batch (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): The input batch

    Returns:
        torch.Tensor: The predicted label (logit)
    """
    x, _ = batch
    return model(x).squeeze()


def second_phase_loss(
    y: torch.Tensor, y_hat: torch.Tensor, loss_fn: nn.Module
) -> torch.Tensor:
    """Calculate the second phase loss of DAE

    Args:
        y (torch.Tensor): The ground truth label
        y_hat (torch.Tensor): The predicted label
        loss_fn (nn.Module): The loss function for the given task

    Returns:
        torch.Tensor: The loss for the given task
    """
    return loss_fn(y_hat, y)
