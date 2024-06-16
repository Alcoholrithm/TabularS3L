import torch
from torch import nn
from torch.nn import functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature=1.0):
        """NT-Xent loss for contrastive learning using cosine distance as similarity metric as used in [SimCLR](https://arxiv.org/abs/2002.05709).
        Implementation adapted from https://theaisummer.com/simclr/#simclr-loss-implementation
        Args:
            temperature (float, optional): scaling factor of the similarity metric. Defaults to 1.0.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """Compute NT-Xent loss using only anchor and positive batches of samples. Negative samples are the 2*(N-1) samples in the batch
        Args:
            z_i (torch.tensor): anchor batch of samples
            z_j (torch.tensor): positive batch of samples
        Returns:
            float: loss
        """
        batch_size = z_i.size(0)
        
        # Compute similarity between the sample's embedding and its corrupted view
        z = torch.cat([z_i, z_j], dim=0)
        similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)
    
        # Extract positive samples
        positives = similarity[range(batch_size), range(batch_size, 2 * batch_size)]
        positives = torch.cat([positives, similarity[range(batch_size, 2 * batch_size), range(batch_size)]], dim=0)

        # Create mask to exclude self-comparisons
        mask = torch.ones((2 * batch_size, 2 * batch_size), dtype=torch.bool, device=z.device)
        mask.fill_diagonal_(0)

        # Compute numerator and denominator
        exp_sim = torch.exp(similarity / self.temperature)
        numerator = torch.exp(positives / self.temperature)
        denominator = exp_sim * mask

        # Compute loss
        loss = -torch.log(numerator / denominator.sum(dim=1)).mean()
        
        return loss