import torch
import torch.nn as nn

class VIMESelfSupervised(nn.Module):
    def __init__(self, input_dim:int):
        """Initialize self-supervised module of VIME

        Args:
            input_dim (int): The dimension of the encoder
        """
        super().__init__()
        self.h = nn.Linear(input_dim, input_dim, bias=True)
        self.mask_output = nn.Linear(input_dim, input_dim, bias=True)
        self.feature_output = nn.Linear(input_dim, input_dim, bias=True)

    def forward(self, x):
        """The forward pass of self-supervised module of VIME

        Args:
            x (torch.FloatTensor): The input batch.

        Returns:
            torch.FloatTensor: The predicted mask vector of VIME
            torch.FloatTensor: The predicted features of VIME
        """
        h = torch.relu(self.h(x))
        mask = torch.sigmoid(self.mask_output(h))
        feature = torch.sigmoid(self.feature_output(h))
        return mask, feature
