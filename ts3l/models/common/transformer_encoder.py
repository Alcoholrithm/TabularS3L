import torch
from torch import nn

class TransformerEncoder(nn.Module):
    def __init__(self, 
                    d_model: int,
                    ffn_factor: float,
                    hidden_dim: int,
                    dropout_rate: float,
                    encoder_depth: int = 3,
                    n_head: int = 2,
                    **kwargs) -> None:
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
        
        if d_model % n_head != 0:
            divisors = [n for n in range(1, d_model + 1) if d_model % n == 0]
    
            # Find the closest divisor to the original num_heads
            closest_num_heads = min(divisors, key=lambda x: abs(x - n_head))
            
            if closest_num_heads != n_head:
                print(f"Adjusting num_heads from {n_head} to {closest_num_heads} (closest valid divisor of {d_model})")
            
            n_head = closest_num_heads
        
        self.encoder_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=int(hidden_dim*ffn_factor), dropout=dropout_rate, batch_first=True)
                for _ in range(encoder_depth)
            ])

        self.output_dim = d_model
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        for layer in self.encoder_layers:
            x = layer(x)
        cls_token = x[:, 0]
        
        return cls_token