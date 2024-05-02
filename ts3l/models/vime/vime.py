from typing import Tuple

import torch
import torch.nn as nn

from .vime_self import VIMESelfSupervised
from .vime_semi import VIMESemiSupervised

class VIME(nn.Module):
    def __init__(self, 
                input_dim: int, hidden_dim: int, output_dim: int):
        """Initialize VIME

        Args:
            input_dim (int): The dimension of the encoder
            hidden_dim (int): The hidden dimension of the predictor
            output_dim (int): The output dimension of the predictor
        """
        super().__init__()
        
        self.self_net = VIMESelfSupervised(input_dim)
        self.semi_net = VIMESemiSupervised(input_dim, hidden_dim, output_dim)
        
        self.set_first_phase()
    
    def set_first_phase(self):
        """Set first phase step as the forward pass
        """
        self.forward = self.__first_phase_step
    
    def set_second_phase(self):
        """Set second phase step as the forward pass
        """
        self.forward = self.__second_phase_step
        
    def __first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The first phase step of VIME

        Args:
            x (torch.Tensor): The input batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The predicted mask vector and predicted features
        """
        mask_output, feature_output = self.self_net(x)
        return mask_output, feature_output
    
    
    def __second_phase_step(self, x):
        """The second phase step of VIME

        Args:
            x (torch.FloatTensor): The input batch.

        Returns:
            torch.FloatTensor: The predicted logits of VIME
        """
        x = self.self_net.h(x)
        logits = self.semi_net(x)
        return logits