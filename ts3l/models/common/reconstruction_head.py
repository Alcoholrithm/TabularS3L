import torch
from torch import nn

from typing import List, Tuple

class ReconstructionHead(nn.Module):
    def __init__(self, input_dim: int, num_continuous: int, cat_cardinality: List[int]):
        super().__init__()
        self.categorical_head = nn.ModuleList([nn.Linear(input_dim, card) for card in cat_cardinality])
        self.continuous_head = nn.Linear(input_dim, num_continuous)
        
    def forward(self, x: torch.Tensor)-> Tuple[List[torch.Tensor], torch.Tensor]:
        
        cat_preds = [linear(x) for linear in self.categorical_head]
        cont_preds = self.continuous_head(x)
        
        return cat_preds, cont_preds