import torch
import torch.nn as nn
from typing import List, Tuple, Dict

class RelationLoss(nn.Module):
    def __init__(self, relation_types: List[str], device: str = "cuda"):
        super().__init__()
        self.relation_types = relation_types
        self.device = device
        self.relation_threshold = 0.5

    def forward(self, 
                predictions: List[Tuple[str, str, str, float]],
                gold_relations: List[Tuple[str, str, str]],
                reduction: str = 'mean') -> torch.Tensor:
        """
        predictions: 예측된 트리플 리스트 [(head_text, rel_type, tail_text, score), ...]
        gold_relations: 정답 트리플 리스트 [(head_text, rel_type, tail_text), ...]
        reduction: loss reduction 방식 ('mean' 또는 'sum')
        Returns:
            loss: 계산된 loss 값 (1 - accuracy)
        """
        return self.compute_relation_loss(
            predictions=predictions,
            gold_relations=gold_relations,
            reduction=reduction
        )

    def compute_relation_loss(self, 
                            predictions: List[Tuple[str, str, str, float]],
                            gold_relations: List[Tuple[str, str, str]],
                            reduction: str = 'mean') -> torch.Tensor:
        """
        predictions: 예측된 트리플 리스트 [(head_text, rel_type, tail_text, score), ...]
        gold_relations: 정답 트리플 리스트 [(head_text, rel_type, tail_text), ...]
        reduction: loss reduction 방식 ('mean' 또는 'sum')
        Returns:
            loss: 계산된 loss 값 (1 - accuracy)
        """
        if not gold_relations or len(gold_relations) == 0:
            return torch.tensor(0.0, requires_grad=True)
        if predictions is None or len(predictions) == 0:
            return torch.tensor(1.0, requires_grad=True)  # 예측이 없으면 loss=1
        # gold set 만들기
        gold_set = set(tuple(g) for g in gold_relations)
        pred_set = set((h, r, t) for h, r, t, _ in predictions)
        # 정답과 예측의 교집합
        correct = len(gold_set & pred_set)
        total = len(gold_set)
        accuracy = correct / total if total > 0 else 0.0
        loss = 1.0 - accuracy
        print(f"\n=== Relation Loss 통계 (Text 기반) ===")
        print(f"정답 관계 수: {total}")
        print(f"예측된 관계 수: {len(pred_set)}")
        print(f"정답과 일치하는 예측 수: {correct}")
        print(f"Loss (1-accuracy): {loss:.4f}")
        return torch.tensor(loss, requires_grad=True)


def flatten_gold_relations(gold_relations, sentences):
    """
    gold_relations: 2중 리스트, 각 gold relation은 [h_start, h_end, t_start, t_end, rel_type]
    sentences: 문장별 토큰 리스트 (예: [['A', 'B'], ['C', 'D', 'E']])
    반환: [(head_text, rel_type, tail_text), ...]
    """
    tokens = [token for sent in sentences for token in sent]

    flat_gold = []
    for rel_list in gold_relations:
        if not isinstance(rel_list, list):
            continue
        for rel in rel_list:
            if not (isinstance(rel, (list, tuple)) and len(rel) == 5):
                continue
            h_start, h_end, t_start, t_end, rel_type = rel
            # head
            if h_start == h_end:
                head_text = tokens[h_start]
            else:
                head_text = " ".join(tokens[h_start:h_end+1])
            # tail
            if t_start == t_end:
                tail_text = tokens[t_start]
            else:
                tail_text = " ".join(tokens[t_start:t_end+1])
            flat_gold.append((head_text, rel_type, tail_text))
    return flat_gold