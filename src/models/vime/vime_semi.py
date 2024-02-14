import torch
import torch.nn as nn

class VIMESemiSupervised(nn.Module):
    def __init__(self, predictor_input_dim, predictor_hidden_dim, predictor_output_dim):
        """_summary_

        Args:
            predictor_input_dim (_type_): The input dimension of the predictor. Must be same to the dimension of the encoder.
            predictor_hidden_dim (_type_): The hidden dimension of the predictor
            predictor_output_dim (_type_): The output dimension of the predictor
        """
        super().__init__()
        self.fc1 = nn.Linear(predictor_input_dim, predictor_hidden_dim)
        self.fc2 = nn.Linear(predictor_hidden_dim, predictor_hidden_dim)
        self.fc3 = nn.Linear(predictor_hidden_dim, predictor_output_dim)

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
