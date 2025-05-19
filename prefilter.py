import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenFilter(nn.Module):
    def __init__(self, hidden_size, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout)
        )

    def forward(self, token_embeddings, mask=None, labels=None):
        """
        Args:
            token_embeddings: [*, L, D] - 임의의 배치 차원을 가진 토큰 임베딩
                            (* can be B or [B1, B2] or any other batch dimensions)
            labels: [*, L] - 임베딩과 동일한 배치 차원을 가진 binary labels (padded labels)
            mask: [*, L] - 임베딩과 동일한 배치 차원을 가진 padding mask

        Returns:
            scores: [*, L] - 각 토큰에 대한 sigmoid scores
            loss: scalar loss if labels are provided
        """
        # 원래 shape 저장
        orig_shape = token_embeddings.shape[:-1]  # 마지막 차원(D)을 제외한 모든 차원
        
        # 임베딩을 2D로 변환 [*, L, D] -> [B', L, D] where B' = prod(*)
        flat_embs = token_embeddings.view(-1, *token_embeddings.shape[-2:])
        B = flat_embs.size(0)
        L = flat_embs.size(1)
        
        # Scoring
        logits = self.scorer(flat_embs).squeeze(-1)  # [B', L]
        scores = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            # labels와 mask도 동일하게 flatten
            flat_labels = labels.view(-1, labels.size(-1))  # [B', L]
            
            if mask is not None:
                flat_mask = mask.view(-1, mask.size(-1))  # [B', L]
                loss = F.binary_cross_entropy_with_logits(
                    logits, flat_labels.float(), reduction="none"
                )
                loss = (loss * flat_mask.float()).sum() / flat_mask.float().sum()
            else:
                loss = F.binary_cross_entropy_with_logits(logits, flat_labels.float())

        # 원래 shape으로 복원
        scores = scores.view(*orig_shape)  # [*, L]

        return scores, loss


class SpanFilter(nn.Module):
    def __init__(self, hidden_size, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Dropout(dropout)
        )

    def forward(self, span_embeddings, labels=None, mask=None):
        """
        Args:
            span_embeddings: [B, N, D] (N = num spans)
            labels: [B, N] binary labels (1 = positive span, 0 = not)
            mask: [B, N] binary mask (1 = valid, 0 = ignore/padding span)

        Returns:
            scores: [B, N] sigmoid scores for each span
            loss: scalar loss if labels are provided
        """

        B, N, D = span_embeddings.size()
        logits = self.scorer(span_embeddings)  # [B, N, 1]
      
        scores = torch.sigmoid(logits)

        loss = None
        if labels is not None:
            if mask is not None:
                # BCE with logits before sigmoid, and apply mask
                loss = F.binary_cross_entropy_with_logits(
                    logits, labels.float(), reduction='none'
                )
                loss = (loss * mask.float()).sum() / (mask.float().sum() + 1e-6)
            else:
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())

        return scores, loss