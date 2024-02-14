import torch
from torch import nn
import numpy as np
import itertools

class JointLoss(nn.Module):
    """
    When computing loss, we use a similarity matrix of size (N x k) x N. The matrix includes k positive samples and all other samples are considered negatives. 
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
                 batch_size: int,
                 tau: float,
                 n_subsets: int,
                 use_contrastive: bool = True,
                 use_distance: bool = True,
                 use_cosine_similarity: bool = False
        ) -> None:
        super(JointLoss, self).__init__()


        # Batch size
        self.batch_size = batch_size
        # n_subsets
        self.n_subsets = n_subsets
        # Temperature to use scale logits
        self.temperature = tau
        # initialize softmax
        self.softmax = nn.Softmax(dim=-1)
        # Mask to use to get negative samples from similarity matrix
        self.mask_for_neg_samples = self._get_mask_for_neg_samples().type(torch.bool)
        # Function to generate similarity matrix: Cosine, or Dot product
        self.similarity_fn = self._cosine_simililarity if use_cosine_similarity else self._dot_simililarity
        # Loss function
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
        self.use_contrastive = use_contrastive
        self.use_distance = use_distance
        

    def _get_mask_for_neg_samples(self):
        # Diagonal 2Nx2N identity matrix, which consists of four (NxN) quadrants
        diagonal = np.eye(2 * self.batch_size)
        # Diagonal 2Nx2N matrix with 1st quadrant being identity matrix
        q1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        # Diagonal 2Nx2N matrix with 3rd quadrant being identity matrix
        q3 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        # Generate mask with diagonals of all four quadrants being 1.
        mask = torch.from_numpy((diagonal + q1 + q3))
        # Reverse the mask: 1s become 0, 0s become 1. This mask will be used to select negative samples
        mask = (1 - mask).type(torch.bool)
        # Transfer the mask to the device and return
        return mask

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
        
    def XNegloss(self, projections: torch.FloatTensor) -> torch.FloatTensor:
        
        # N = projections.shape[0]
        
        # Compute cosine similarity using the provided function
        similarity = self.similarity_fn(projections, projections) / self.temperature
        
        
        # return torch.stack([self.get_anchor_loss(turn, similarity[turn]) for turn in range(len(similarity))]).mean()
        anchor_losses = [self.get_anchor_loss(turn, similarity[turn]) for turn in range(len(similarity))]
        
        return torch.stack(anchor_losses).mean()
    
    def getMSEloss(self, recon, target):
        """
        Args:
            recon (torch.FloatTensor):
            target (torch.FloatTensor):
        """
        loss = torch.mean(torch.square(recon - target))
        return loss

    def forward(self, projections, xrecon, recon_label):
        """
        Args:
            projections (torch.FloatTensor):
            xrecon (torch.FloatTensor):
            xorig (torch.FloatTensor):
        """
        
        # recontruction loss
        recon_loss = self.getMSEloss(xrecon, recon_label)
        # recon_loss = self.getMSEloss(xrecon, xorig)

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
            dist_loss = torch.stack([self.getMSEloss(projections[i:i + self.n_subsets][left], projections[i:i + self.n_subsets][right]) for i in range(0, len(projections), self.n_subsets)]).mean()
            loss += dist_loss

        return loss, closs, recon_loss, dist_loss
