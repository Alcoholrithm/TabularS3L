import torch
from torch import nn
import numpy as np
import itertools

class JointLoss(nn.Module):
    """ JointLoss for SubTab during the first phase learning.
    
    When computing the constrastive loss, we use a similarity matrix of size (N x k) x N. The matrix includes k positive samples and all other samples are considered negatives. 
    The matrix is shown below as an 8x4 array, assuming a batch size of 4 and 2 subsets.
                                                        P . . .
                                                        P . . .
                                                        . P . .
                                                        . P . .
                                                        . . P .
                                                        . . P .
                                                        . . . P
                                                        . . . P
    """

    def __init__(self,
                 # batch_size: int,
                 tau: float,
                 n_subsets: int,
                 use_contrastive: bool = True,
                 use_distance: bool = True,
                 use_cosine_similarity: bool = False
        ) -> None:
        super(JointLoss, self).__init__()

        # n_subsets
        self.n_subsets = n_subsets
        # Temperature to use scale logits
        self.temperature = tau

        # Function to generate similarity matrix: Cosine, or Dot product
        self.similarity_fn = self._cosine_simililarity if use_cosine_similarity else self._dot_simililarity
        # Loss function
        self.mse_loss = nn.MSELoss()
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
        self.use_contrastive = use_contrastive
        self.use_distance = use_distance
        

    @staticmethod
    def _dot_simililarity(x, y):
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.T.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        similarity = torch.tensordot(x, y, dims=2)
        return similarity

    def _cosine_simililarity(self, x, y):
        similarity = nn.CosineSimilarity(dim=-1)
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        return similarity(x, y)
    
    def get_anchor_loss(self, turn, similarity):
        group_start = (turn // self.n_subsets) * self.n_subsets
        group_end = min(group_start + self.n_subsets, len(similarity))

        # Select the positive and negative pairs for the anchor
        positives = similarity[group_start:group_end]
        negatives = torch.cat((similarity[:group_start], similarity[group_end:]), dim=0)

        # Exclude self-similarity
        positives = positives[positives != 1]

        # Loss calculation for the anchor
        pos_sum = torch.sum(torch.exp(positives))
        neg_sum = torch.sum(torch.exp(negatives))
        # return -torch.log(pos_sum / (pos_sum + neg_sum))
        anchor_loss = -torch.log(pos_sum / (pos_sum + neg_sum))
        return anchor_loss
        
    def XNegloss(self, projections: torch.FloatTensor) -> torch.Tensor:
        
        
        # Compute cosine similarity using the provided function
        similarity = self.similarity_fn(projections, projections) / self.temperature
        
        # return torch.stack([self.get_anchor_loss(turn, similarity[turn]) for turn in range(len(similarity))]).mean()
        anchor_losses = [self.get_anchor_loss(turn, similarity[turn]) for turn in range(len(similarity))]
        
        return torch.stack(anchor_losses).mean()

    def forward(self, projections, xrecon, xorig):
        """
        Args:
            projections (torch.FloatTensor): Projections for each subset.
            xrecon (torch.FloatTensor): Reconstructed sample x from subsets.
            xorig (torch.FloatTensor): Original features of x
        """
        
        # recontruction loss
        recon_loss = self.mse_loss(xrecon, xorig)

        # Initialize contrastive and distance losses with recon_loss as placeholder
        closs, dist_loss = None, None

        # Start with default loss i.e. reconstruction loss
        loss = recon_loss

        if self.use_contrastive:
            closs = self.XNegloss(projections)
            loss = loss + closs

        if self.use_distance:
            # recontruction loss for z
            combi = np.array(list(itertools.combinations(range(self.n_subsets), 2)))
            left = combi[:, 0]
            right = combi[:, 1]
            dist_loss = torch.stack([self.mse_loss(projections[i:i + self.n_subsets][left], projections[i:i + self.n_subsets][right]) for i in range(0, len(projections), self.n_subsets)]).mean()
            loss += dist_loss

        return loss, closs, recon_loss, dist_loss
