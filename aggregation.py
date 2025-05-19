import torch
from torch import nn
from typing import List, Tuple



class EntityEmbeddingUpdater(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self,
                entity_emb: torch.Tensor,                  # [N, D]
                cls_embeddings: torch.Tensor,              # [K, D] - top triple의 CLS 벡터
                triple_indices: List[int],                 # top triple의 글로벌 인덱스
                triples: List[Tuple[int, int, int, str]],  # 전체 triple 리스트
                target_head_id: int                        # 업데이트할 entity 노드 ID
                ) -> torch.Tensor:
        """
        하나의 head entity 노드를, 관련된 triple의 CLS 임베딩을 weighted sum + Linear 통과하여 업데이트.

        Returns:
            updated_embedding: [D]
        """
        device = entity_emb.device
        selected_cls = []

        for local_idx, global_idx in enumerate(triple_indices):
            h, _, _, _ = triples[global_idx]
            if h == target_head_id:
                selected_cls.append(cls_embeddings[local_idx])  # local_idx는 top triple 기준

        if len(selected_cls) == 0:
            return entity_emb[target_head_id]  # 업데이트 없음

        cls_tensor = torch.stack(selected_cls, dim=0)  # [K, D]
        weights = torch.softmax(torch.ones(len(selected_cls), device=device), dim=0)  # uniform weight
        aggregated = torch.sum(cls_tensor * weights.unsqueeze(1), dim=0)  # [D]

        updated = self.linear(aggregated)  # Linear 통과
        print(f"updated: {target_head_id}")

        return updated



# 단순 Linear layer transformation
class RelationEmbeddingUpdater(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_emb: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            node_emb: [N, D] - 전체 노드 임베딩 (entity + relation)
            node_type: [N]- 0: entity, 1: relation

        Returns:
            updated_emb: [N, D]- relation node만 업데이트됨
        """
        updated = node_emb.clone()
        relation_mask = (node_type == 1)

        # relation 노드만 선형 변환
        updated[relation_mask] = self.linear(node_emb[relation_mask])

        return updated

# class RelationEmbeddingUpdater(nn.Module):
#     def __init__(self, hidden_dim: int):
#         super().__init__()
#         self.linear = nn.Linear(hidden_dim, hidden_dim)

#     def forward(self,
#                 relation_emb: torch.Tensor,                  # [N, D]
#                 cls_embeddings: torch.Tensor,                # [K, D] - top triple의 CLS 벡터
#                 triple_indices: list,                        # top triple의 글로벌 인덱스
#                 triples: list,                               # 전체 triple 리스트 (head, rel, tail, type)
#                 target_rel_id: int                           # 업데이트할 relation 노드 ID
#                 ) -> torch.Tensor:
#         """
#         하나의 relation 노드를, 관련된 triple의 CLS 임베딩을 weighted sum + Linear 통과하여 업데이트.
#         Returns:
#             updated_embedding: [D]
#         """
#         device = relation_emb.device
#         selected_cls = []
#         print("%%%%%%%%%%%in aggregation triple_indices: ", triple_indices)
#         for local_idx, global_idx in enumerate(triple_indices):
#             _, r, _, _ = triples[global_idx]
#             if r == target_rel_id:
#                 selected_cls.append(cls_embeddings[local_idx])
#         if len(selected_cls) == 0:
#             return relation_emb[target_rel_id]  # 업데이트 없음
#         cls_tensor = torch.stack(selected_cls, dim=0)  # [K, D]
#         weights = torch.softmax(torch.ones(len(selected_cls), device=device), dim=0)  # uniform weight
#         aggregated = torch.sum(cls_tensor * weights.unsqueeze(1), dim=0)  # [D]
#         updated = self.linear(aggregated)  # Linear 통과
#         print(f"updated relation: {target_rel_id}")
#         return updated
