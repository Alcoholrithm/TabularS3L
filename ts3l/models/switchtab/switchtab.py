import torch
from torch import nn

from typing import Tuple, List, Union
from ts3l.models.switchtab.ft_transformer import FTTransformer

class Encoder(nn.Module):
    def __init__(self, 
                    cont_nums: int,
                    category_dims: List[int],
                    ffn_factor: float,
                    hidden_dim: int,
                    dropout_rate: float,
                    encoder_depth: int = 3,
                    n_head: int = 2) -> None:
        """Initializes the encoder module used in the SwitchTab, employing an FT-Transformer architecture

        Args:
            cont_nums (int): The number of continuous features.
            category_dims (List[int]): A list of dimensions of the categorical features.
            ffn_factor (float): The scaling factor for the size of the feedforward network within the transformer blocks.
            hidden_dim (int): The dimensionality of the hidden layers within the network.
            dropout_rate (float): The dropout rate used within the encoder.
            encoder_depth (int, optional): The number of layers in the encoder. Defaults to 3.
            n_head (int, optional): The number of attention heads in the encoder. Defaults to 2.
        """
        super().__init__()
        
        self.transformer = FTTransformer(cont_nums=cont_nums, cat_dims=category_dims, emb_dim=hidden_dim, n_heads=n_head, attn_dropout=dropout_rate, ffn_dropout=dropout_rate, ffn_factor_dim=ffn_factor, depth=encoder_depth)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)
    
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
    def __init__(self, input_dim:int, hidden_dim: int) -> None:
        """Initializes the decoder module used in the SwitchTab

        Args:
            input_dim (int): The dimensionality of input features of SwitchTab.
            hidden_dim (int): The dimensionality of the output of the projector.
        """
        super().__init__()
        
        self.linear = nn.Linear(hidden_dim * 2, input_dim)
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(x))

class SwitchTab(nn.Module):
    def __init__(self,
                    input_dim: int,
                    output_dim: int,
                    category_dims: List[int],
                    hidden_dim: int,
                    ffn_factor: int,
                    dropout_rate: float,
                    encoder_depth: int = 3,
                    n_head: int = 2) -> None:
        """Initialize SwitchTab

        Args:
            input_dim (int): The dimensionality of the input features.
            output_dim (int): The dimensionality of the output.
            category_dims (List[int]): A list of dimensions of the categorical features.
            hidden_dim (int): The dimensionality of the hidden layers within the network.
            ffn_factor (int): The scaling factor for the size of the feedforward network inside the encoder.
            dropout_rate (float): The dropout rate used within the encoder.
            encoder_depth (int, optional): The number of layers in the encoder. Defaults to 3.
            n_head (int, optional): The number of attention heads in the encoder. Defaults to 2.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.__return_salient_feature = False
        
        
        self.encoder = Encoder(input_dim - len(category_dims), category_dims, ffn_factor, hidden_dim, dropout_rate, encoder_depth, n_head)
        self.projector_m = Projector(hidden_dim)
        self.projector_s = Projector(hidden_dim)
        self.decoder = Decoder(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.SiLU()
        
    def set_first_phase(self) -> None:
        """Set first phase step as the forward pass
        """
        self.forward = self.__first_phase_step
    
    def set_second_phase(self) -> None:
        """Set second phase step as the forward pass
        """
        self.forward = self.__second_phase_step

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
        
    def __first_phase_step(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """The first phase step of SwitchTab
        Processes the given samples to decuple salient and mutual embeddings across data samples.
        
        Args:
            x (torch.Tensor): The input batch

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Reconstructed tensors and predicted labels.
        """

        size = len(x) // 2
        
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

    
    def __second_phase_step(self, 
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
        emb = self.encoder(x)
        y_hat = self.head(self.activation(emb))
        if not self.return_salient_feature:
            return y_hat
        else:
            salient_features = self.projector_s(emb)
            return y_hat, salient_features