import torch
from torch import nn

from ts3l.models.common import TS3LModule
from ts3l.models.common import MLP
from ts3l.utils import BaseEmbeddingConfig, BaseBackboneConfig


class TabularBinning(TS3LModule):
    def __init__(self,
                 embedding_config: BaseEmbeddingConfig,
                 backbone_config: BaseBackboneConfig,
                 output_dim: int,
                 n_bin: int,
                 n_decoder: int,
                 decoder_dim: int,
                 decoder_depth: int,
                 first_phase_output_dim: int,
                 dropout_rate: float = 0.04,
                 **kwargs) -> None:
        """Initialize TabularBinning

        Args:
            embedding_config (BaseEmbeddingConfig): Configuration for the embedding layer.
            backbone_config (BaseBackboneConfig): Configuration for the backbone network.
        """
        super(TabularBinning, self).__init__(embedding_config, backbone_config)

        self.first_phase_output_dim = first_phase_output_dim

        decoder_output_dim = self.embedding_module.input_dim if n_decoder == 1 else n_bin
        self.decoders = nn.ModuleList()
        for _ in range(n_decoder):
            decoder = MLP(
                input_dim=self.backbone_module.output_dim,
                hidden_dims=decoder_dim,
                n_hiddens=decoder_depth,
                output_dim=decoder_output_dim,
                dropout_rate=dropout_rate
            )
            self.decoders.append(decoder)

        self.head = nn.Linear(self.backbone_module.output_dim, output_dim)

    @property
    def encoder(self) -> nn.Module:
        return self.backbone_module

    def _first_phase_step(self, x: torch.Tensor) -> torch.Tensor:

        x = self.embedding_module(x)
        z_e = self.encoder(x)

        batch_size = z_e.size(0)
        z_d = torch.empty(batch_size, len(
            self.decoders), self.decoders[0].output_dim, device=z_e.device)  # type: ignore

        for i, decoder in enumerate(self.decoders):
            z_d[:, i] = decoder(z_e)

        return z_d.reshape(-1, self.first_phase_output_dim)

    def _second_phase_step(self,
                           x: torch.Tensor) -> torch.Tensor:
        x = self.embedding_module(x)
        z_e = self.encoder(x)
        y_hat = self.head(z_e)
        return y_hat
