import torch
import torch.nn as nn
import torch.nn.functional as F

class EntityNodeFilter(nn.Module):
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout)
        )

    def forward(self, token_embeddings, labels=None, mask=None):
        """
        Args:
            token_embeddings: [B, L, D] from EntityTokenRep
            labels: [B, L] binary labels (1 = part of entity, 0 = not)
            mask: [B, L] binary padding mask (1 = real token, 0 = padding)

        Returns:
            scores: [B, L] sigmoid scores for each token
            loss: scalar loss if labels are provided
        """
        B, L, D = token_embeddings.size()
        logits = self.scorer(token_embeddings).squeeze(-1)  # [B, L]
        scores = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            if mask is not None:
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), reduction="none"
                )
                loss = (loss * mask.float()).sum() / mask.float().sum()
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return scores, loss
