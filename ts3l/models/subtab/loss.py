import torch
from torch import nn
import numpy as np
import itertools
from torch.nn import functional as F
class JointLoss(nn.Module):
    """ JointLoss for SubTab during the first phase learning.
    
    When computing the constrastive loss, we use a similarity matrix of size (N x k) x N. 
    he matrix includes k positive samples and all other samples are considered negatives. 
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
                 tau: float,
                 n_subsets: int,
                 use_contrastive: bool = True,
                 use_distance: bool = True,
                 use_cosine_similarity: bool = False
        ) -> None:
        super(JointLoss, self).__init__()


        self.n_subsets = n_subsets
        
        # Temperature to use scale logits
        self.temperature = tau

        self.use_cosine_similarity = use_cosine_similarity
        
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
        # Reshape x: (2N, C) -> (2N, 1, C)
        x = x.unsqueeze(1)
        # Reshape y: (2N, C) -> (1, C, 2N)
        y = y.unsqueeze(0)
        # Similarity shape: (2N, 2N)
        return F.cosine_similarity(x, y, dim=-1)
    
    def get_anchor_loss(self, similarity: torch.Tensor) -> torch.Tensor:
        batch_size = similarity.size(0)
        group_size = self.n_subsets
        
        # Create a mask to exclude self-similarity
        identity_mask = torch.eye(
            batch_size, dtype=torch.bool, device=similarity.device
        )

        # Create masks for positive and negative pairs
        group_indices = torch.arange(batch_size, device=similarity.device) // group_size
        group_mask = group_indices.unsqueeze(0) == group_indices.unsqueeze(1)

        positives_mask = group_mask & ~identity_mask
        negatives_mask = ~group_mask

        # Compute positive and negative sums
        pos_sum = torch.sum(torch.exp(similarity) * positives_mask.float(), dim=1)
        neg_sum = torch.sum(torch.exp(similarity) * negatives_mask.float(), dim=1)

        # Exclude zero positive sums to avoid log(0)
        pos_sum = torch.clamp(pos_sum, min=1e-10)

        # Compute anchor losses
        anchor_loss = -torch.log(pos_sum / (pos_sum + neg_sum))

        return anchor_loss
        
    def XNegloss(self, projections: torch.FloatTensor) -> torch.Tensor:
        
        # Compute cosine similarity using the provided function
        similarity = self.similarity_fn(projections, projections)
        
        # return torch.stack([self.get_anchor_loss(turn, similarity[turn]) for turn in range(len(similarity))]).mean()
        anchor_losses = self.get_anchor_loss(similarity)
        
        return anchor_losses.mean()

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
            
            # Create an index tensor for the pairs
            indices = torch.arange(len(projections)).view(-1, self.n_subsets)
            left_indices = indices[:, left].reshape(-1)
            right_indices = indices[:, right].reshape(-1)

            # Compute the MSE loss in a vectorized manner
            dist_loss = self.mse_loss(projections[left_indices], projections[right_indices])
        
            loss += dist_loss

        return loss, closs, recon_loss, dist_loss
