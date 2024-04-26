import torch
from torch import nn

from typing import Tuple, List, Union

from ts3l.models.common import initialize_weights
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
        super().__init__()
        
        self.transformer = FTTransformer(cont_nums=cont_nums, cat_dims=category_dims, emb_dim=hidden_dim, n_heads=n_head, attn_dropout=dropout_rate, ffn_dropout=dropout_rate, ffn_factor_dim=ffn_factor, depth=encoder_depth)
        # self.tokenizer = FeatureTokenizer(hidden_dim, cont_nums, category_dims)
        # self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = hidden_dim, nhead = n_head, dim_feedforward=int(hidden_dim * ffn_factor), dropout=dropout_rate, batch_first=True), num_layers=encoder_depth)
        # self.transformer = nn.Sequential(MLP(input_dim, hidden_dim, encoder_depth - 1, dropout_rate), nn.LeakyReLU(),                               nn.Linear(hidden_dim, input_dim))
        # self.transformers = nn.Sequential(
        #     *[nn.TransformerEncoderLayer(d_model = input_dim, nhead = n_head, dim_feedforward=hidden_dim, dropout=dropout_rate) for _ in range(encoder_depth)]
        # )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x  = self.tokenizer(x)
        return self.transformer(x)
    
class Projector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        
        self.linear = nn.Linear(input_dim, hidden_dim)
        # self.activation = nn.Sigmoid()
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(x))

class Decoder(nn.Module):
    def __init__(self, input_dim:int, hidden_dim: int) -> None:
        super().__init__()
        
        self.linear = nn.Linear(hidden_dim * 2, input_dim)
        # self.activation = nn.Sigmoid()
        self.activation = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.activation(x))
        # return self.activation(self.linear(x))

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
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.__return_salient_feature = False
        
        # self.tokenizer = MLP(input_dim, hidden_dim, 1, dropout_rate)
        self.encoder = Encoder(input_dim - len(category_dims), category_dims, ffn_factor, hidden_dim, dropout_rate, encoder_depth, n_head)
        self.projector_m = Projector(hidden_dim, hidden_dim)
        self.projector_s = Projector(hidden_dim, hidden_dim)
        self.decoder = Decoder(input_dim, hidden_dim)
        self.head = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.SiLU()
        
        initialize_weights(self.encoder)
        initialize_weights(self.projector_m)
        initialize_weights(self.projector_s)
        initialize_weights(self.decoder)
        initialize_weights(self.head)
        
    def set_first_phase(self) -> None:
        self.forward = self.__first_phase_step
    
    def set_second_phase(self) -> None:
        self.forward = self.__second_phase_step

    @property
    def return_salient_feature(self):
        return self.__return_salient_feature
    
    @return_salient_feature.setter
    def return_salient_feature(self, flag):
        self.__return_salient_feature = flag
        
    def __first_phase_step(self, xcs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # xs, xcs, ys = batch
        size = len(xcs) // 2
        # print(xcs)
        # emb = self.tokenizer(xcs)
        # print(emb.shape)
        # print(emb.shape, emb)
        zs = self.encoder(xcs)
        ms = self.projector_m(zs)
        ss = self.projector_s(zs)
        
        m1, s1 = ms[:size], ss[:size]
        m2, s2 = ms[size:], ss[size:]
        
        # m1s1 = torch.concat([m1, s1])
        # m1s2 = torch.concat([m1, s2])
        # m2s1 = torch.concat([m2, s1])
        # m2s2 = torch.concat([m2, s2])
        # print(m1.shape, s1.shape)
        # print(torch.concat([m1, s1], dim=1).shape)
        x1_tilde_hat = torch.concat([torch.concat([m1, s1], dim=1), torch.concat([m2, s1], dim=1)])
        x2_hat_tilde = torch.concat([torch.concat([m1, s2], dim=1), torch.concat([m2, s2], dim=1)])
        # print(x1_tilde_hat.shape, x2_hat_tilde.shape, self.decoder)
        # print(x1_tilde_hat, x2_hat_tilde)
        x1_recover_switch = self.decoder(x1_tilde_hat)
        x2_switch_recover = self.decoder(x2_hat_tilde)
        
        x_hat = torch.concat([x1_recover_switch, x2_switch_recover])
        # x1_recover = self.decoder(m1s1)
        # x2_switch = self.decoder(m1s2)
        # x2_recover = self.decoder(m2s1)
        # x1_switch = self.decoder(m2s2)
        
        # cls_idx = (ys != None)
        y_hat = self.head(self.activation(zs))
        # print("xhat,yhat", zs, x_hat, y_hat)
        return x_hat, y_hat

    
    def __second_phase_step(self, 
                x : torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        emb = self.encoder(x)
        y_hat = self.head(self.activation(emb))
        if not self.return_salient_feature:
            return y_hat
        else:
            salient_features = self.projector_s(emb)
            return y_hat, salient_features