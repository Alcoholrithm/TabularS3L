import torch
import torch.nn as nn

class VIMESemiSupervised(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        """_summary_

        Args:
            input_dim (int): The input dimension of the predictor. Must be same to the dimension of the encoder.
            hidden_dim (int): The hidden dimension of the predictor
            output_dim (int): The output dimension of the predictor
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """The forward pass of semi-supervised module of VIME

        Args:
            x (torch.FloatTensor): The input batch.

        Returns:
            torch.FloatTensor: The predicted logits of VIME
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
