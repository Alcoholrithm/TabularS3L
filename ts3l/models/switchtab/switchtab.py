import torch
from torch import nn

from typing import Tuple, List, Union
from ts3l.models.common import TS3LModule
from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig

class Projector(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        """Initializes the projector module used in the SwitchTab

        Args:
            hidden_dim (int): The dimensionality of both the input and output of the projector.
        """
        super().__init__()
        
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(x))

class Decoder(nn.Module):
    def __init__(self, hidden_dim: int, output_dim:int) -> None:
        """Initializes the decoder module used in the SwitchTab

        Args:
            hidden_dim (int): The dimensionality of the output of the projector.
            output_dim (int): The dimensionality of input features of SwitchTab.
        """
        super().__init__()
        
        self.linear = nn.Linear(hidden_dim * 2, output_dim)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(x))

class SwitchTab(TS3LModule):
    def __init__(self,
                    embedding_config: BaseEmbeddingConfig,
                    backbone_config: BaseBackboneConfig,
                    output_dim: int,
                    **kwargs) -> None:
        """Initialize SwitchTab

        Args:
            embedding_config (BaseEmbeddingConfig): Configuration for the embedding layer.
            backbone_config (BaseBackboneConfig): Configuration for the backbone network.
            output_dim (int): The dimensionality of the output.
        """
        super(SwitchTab, self).__init__(embedding_config, backbone_config)
        self.output_dim = output_dim
        self.__return_salient_feature = False
        
        self.projector_m = Projector(self.backbone_module.output_dim)
        self.projector_s = Projector(self.backbone_module.output_dim)
        
        self.decoder = Decoder(self.backbone_module.output_dim, self.embedding_module.input_dim)
        self.head = nn.Linear(self.backbone_module.output_dim, output_dim)
        self.activation = nn.SiLU()

    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module
        
    @property
    def return_salient_feature(self) -> bool:
        """Gets the value of the private attribute '__return_salient_feature' which indicates whether 
        salient features should be returned by the model.

        Returns:
            bool: The current state of the '__return_salient_feature' flag.
        """
        return self.__return_salient_feature
    
    @return_salient_feature.setter
    def return_salient_feature(self, flag: bool) -> None:
        """Sets the value of the private attribute '__return_salient_feature' to control whether 
        salient features should be returned by the model.

        Args:
            flag (bool): A boolean value to set the '__return_salient_feature' attribute.
        """
        self.__return_salient_feature = flag

    def _first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The first phase step of SwitchTab
        Processes the given samples to decuple salient and mutual embeddings across data samples.
        
        Args:
            x (torch.Tensor): The input batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Reconstructed tensors and predicted labels.
        """
        
        size = len(x) // 2
        x = self.embedding_module(x)
        zs = self.encoder(x)
        
        ms = self.projector_m(zs)
        ss = self.projector_s(zs)
        
        m1, s1 = ms[:size], ss[:size]
        m2, s2 = ms[size:], ss[size:]
        
        x1_tilde_hat = torch.concat([torch.concat([m1, s1], dim=1), torch.concat([m2, s1], dim=1)])
        x2_hat_tilde = torch.concat([torch.concat([m1, s2], dim=1), torch.concat([m2, s2], dim=1)])

        x1_recover_switch = self.decoder(x1_tilde_hat)
        x2_switch_recover = self.decoder(x2_hat_tilde)
        
        x_hat = torch.concat([x1_recover_switch, x2_switch_recover])
        y_hat = self.head(self.activation(zs))

        return x_hat, y_hat

    
    def _second_phase_step(self, 
                x : torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """The second phase step of SwitchTab

        Args:
            x (torch.Tensor): The input batch

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: If 'return_salient_feature' is False,
                returns a tensor 'y_hat' representing the predicted label. If True, returns a tuple
                where the first element is 'y_hat' and the second element is a tensor of salient features extracted
                from the input.
        """
        x = self.embedding_module(x)
        emb = self.encoder(x)
        y_hat = self.head(self.activation(emb))
        if not self.return_salient_feature:
            return y_hat
        else:
            salient_features = self.projector_s(emb)
            return y_hat, salient_features