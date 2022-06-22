import torch
import torch.nn as nn

from typing import Tuple

class SimplexSampler(nn.Module):
    """One way to train seq2slate model is autoregressive.
    I.e. at each time step we choose j candidate, s.t. the 
    generative probability of resulting permutation is optimal 
    :math:`perm_* = argmax(P(perm_j|perm_{<j}, candidate))`. 
    Authors suggest two ways to sample the candidate:
    * Greedy: At each time step we choose the item with highest proability
    * Sampling: Sample candidate, theoretically converges to the expected 
        value (one would be optimized)
    """

    def forward(self, probas: torch.Tensor, greedy: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode one-step

        Args:
            probas (torch.Tensor): Probability distributions of decoder.
                Shape: batch_size, target_seq_len, num_of_candidates
            greedy (bool): Whether to greedily pick or sample the next symbol.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Chosen candidate & Generative probabilities of last step
                shape: batch_size, 1
                shape: batch_size, num_of_candidates
        """
        batch_size = probas.shape[0]
        # Retrieve the last observed probabilities
        probas_dist = probas[:, -1, :]
        if greedy:
            _, candidate = torch.max(probas_dist, dim=1)
        else:
            # idx = min({i in {1, ..., len(probas_dist)}: sum(probas_dist, i) - X >= 0}})
            # Where X ~ U(0,1) and probas_dist sorted descendically.
            candidate = torch.multinomial(probas_dist, num_samples=1, replacement=False)
        candidate = candidate.reshape(batch_size, 1)
        return candidate, probas_dist