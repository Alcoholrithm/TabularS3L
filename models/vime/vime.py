import torch.nn as nn

from .vime_self import VIMESelfSupervised
from .vime_semi import VIMESemiSupervised

class VIME(nn.Module):
    def __init__(self, 
                encoder_dim: int, predictor_hidden_dim: int, predictor_output_dim: int):
        """Initialize VIME

        Args:
            encoder_dim (int): The dimension of the encoder
            predictor_hidden_dim (int): The hidden dimension of the predictor
            predictor_output_dim (int): The output dimension of the predictor
        """
        super().__init__()
        
        self.self_net = VIMESelfSupervised(encoder_dim)
        self.semi_net = VIMESemiSupervised(encoder_dim, predictor_hidden_dim, predictor_output_dim)
        
        self.do_pretraining()
    
    def do_pretraining(self):
        """Set pretraining step as the forward pass
        """
        self.forward = self.pretraining_step
    
    def do_finetunning(self):
        """Set finetunning step as the forward pass
        """
        self.forward = self.finetunning_step
        
    def pretraining_step(self, x):
        """The pretraining step of VIME

        Args:
            x (torch.FloatTensor): The input batch

        Returns:
            torch.FloatTensor: The predicted mask vector of VIME
            torch.FloatTensor: The predicted features of VIME
        """
        mask_output, feature_output = self.self_net(x)
        return mask_output, feature_output
    
    
    def finetunning_step(self, x):
        """The finetunning step of VIME

        Args:
            x (torch.FloatTensor): The input batch.

        Returns:
            torch.FloatTensor: The predicted logits of VIME
        """
        x = self.self_net.h(x)
        logits = self.semi_net(x)
        return logits