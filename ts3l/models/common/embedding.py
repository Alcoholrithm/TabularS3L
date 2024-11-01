import torch
from torch import nn
from typing import Union, Type, List

from ts3l.utils import BaseEmbeddingConfig

class FeatureTokenizer(nn.Module):

    def __init__(self, 
                emb_dim : int,
                cont_nums : int,
                cat_dims : List[int],
                required_token_dim: int = 1,
                **kwargs,
        ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.cont_nums = cont_nums
        self.cat_nums = len(cat_dims)
        
        self.cat_dims = cat_dims

        bias_dim = 0

        if cont_nums is not None:
            bias_dim += cont_nums

        if cat_dims is not None:
            bias_dim += len(cat_dims)

            category_offsets = torch.tensor([0] + cat_dims[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)

            self.cat_weights = nn.Embedding(sum(cat_dims), emb_dim)
        
        self.weight = nn.Parameter(torch.Tensor(cont_nums + 1, emb_dim))
        self.bias = nn.Parameter(torch.Tensor(bias_dim, emb_dim))
        
        self.n_features = self.cont_nums + self.cat_nums
        
        if required_token_dim == 2:
            self.forward = self.__generate_tokens
        else:
            self.forward = self.__generate_flattened_toke

    def __generate_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x_cats, x_conts = x[:, :self.cat_nums].long(), x[:, self.cat_nums:]
        
        x_conts = torch.cat(
            [torch.ones(
                len(x_conts) if x_conts is not None else len(x_cats), 1, 
                device = x_conts.device if x_conts is not None else x_cats.device)
            ]
            + ([] if x_conts is None else [x_conts]),
            dim = 1
        )

        x = self.weight[None] * x_conts[:, :, None]

        if x_cats is not None:

            x = torch.cat(
                [x, self.cat_weights(x_cats + self.category_offsets[None])],
                dim=1,
            )
        if self.bias is not None:
            bias = torch.cat(
                [
                    torch.zeros(1, self.bias.shape[1], device=x.device),
                    self.bias,
                ]
            )
            x = x + bias[None]

        return x

    def __generate_flattened_toke(self, 
                x: torch.Tensor
        ) -> torch.Tensor:

        x = self.__generate_tokens(x)
        
        return x.reshape(-1, self.emb_dim * self.n_features)
        
class TS3LEmbeddingModule(nn.Module):
    def __init__(self, config: BaseEmbeddingConfig):
        super().__init__()
        self.config = config
        
        self.__set_embedding_layer()
        self.__set_embedding_dim()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embeding_layer(x)
        
    def __set_embedding_dim(self):
        if self.config.module == "identity":
            self.output_dim = self.config.input_dim
        else:
            self.output_dim = self.config.emb_dim
    
    def __set_embedding_layer(self):
        if self.config.module == "identity":
            self.embeding_layer = nn.Identity()
        elif self.config.module == "feature_tokenizer":
            self.embeding_layer = FeatureTokenizer(**self.config.__dict__)