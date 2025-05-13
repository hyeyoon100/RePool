import torch
from torch import nn
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from typing import List
import random

def get_topk_candidates(scores, candidates, topk):
    """
    scores: [B, L] — 필터링 스코어
    candidates: [B, L, D] or [B, L]
    topk: int — top-k 개수

    Returns:
        topk_candidates: 선택된 후보들
        sorted_idx: 정렬된 인덱스
    """
    sorted_idx = torch.sort(scores, dim=-1, descending=True)[1]  # [B, L] → 내림차순 정렬 인덱스
    topk_idx = sorted_idx[:, :topk]                              # [B, topk]

    if candidates.dim() == 3:
        # [B, topk, D] 임베딩 
        B, L, D = candidates.shape
        selected = candidates.gather(1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        # topk_idx.unsqueeze(-1) → [B, topk, 1]
        #   (차원을 하나 추가해서 broadcast 준비)
        # expand(-1, -1, D) → [B, topk, D]
        #   각 인덱스를 D차원에 걸쳐 확장함 → 이제 gather 할 수 있음
        # gather(1, ...) → dim=1 
        #   (두 번째 차원 = 토큰 시퀀스 차원)에서 top-k 인덱스에 해당하는 D차원 벡터들을 추출

    else:
        # [B, topk] 라벨, 마스크, 인덱스
        selected = candidates.gather(1, topk_idx)

    return selected, topk_idx



def build_bipartite_graph(entity_emb, keep_ids, relation_emb, bidirectional=True, use_node_type=True):
    """
    Build a PyG homogeneous Data object with all entity nodes included, but only
    connect pre-filtered entity nodes (keep_ids) to relation nodes.

    Args:
        entity_emb (Tensor): [num_entity, dim]
        keep_ids (List[int]): indices of entity nodes to connect to relations
        relation_emb (Tensor): [num_relation, dim]
        bidirectional (bool): Whether to add reverse edges
        use_node_type (bool): Whether to attach node type annotation

    Returns:
        data (torch_geometric.data.Data): PyG graph object
    """
    num_entity = entity_emb.size(0)
    num_relation = relation_emb.size(0)
    dim = entity_emb.size(1)

    # Combine entity and relation embeddings
    x = torch.cat([entity_emb, relation_emb], dim=0)  # [num_nodes, dim]

    edge_index = [] 
    edge_type = []  # 각 엣지에 해당하는 relation type ID 사용 #! (왜 필요하지?)

    # 엣지 생성 (entity <-> relation bipartite)
    # entity 노드는 0부터 시작
    # relation 노드는 num_entity 이후부터 인덱스 할당됨
    # → 그래서 r + num_entity 로 relation 노드의 실제 인덱스를 지정함
    # edge_from = e  # entity node index
    # edge_to = r + num_entity  # relation node index 이런 형태.
    for r in range(num_relation):
        for e in keep_ids:
            edge_index.append([e, r + num_entity])
            edge_type.append(r)
            if bidirectional: # 양방향 처리
                edge_index.append([r + num_entity, e])
                edge_type.append(r)

    # 텐서 변환
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long)

    # entity node는 0, relation node는 1 할당 (optional)
    node_type = None
    if use_node_type:
        node_type = torch.cat([
            torch.zeros(num_entity, dtype=torch.long),
            torch.ones(num_relation, dtype=torch.long)
        ])

    data = Data(x=x, edge_index=edge_index, edge_type=edge_type)
    # x: 노드 임베딩 전체 [num_entity + num_relation, dim]
    if node_type is not None:
        data.node_type = node_type

    return data

#* for input_text_embedding
class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-cased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, sentences: List[str]) -> torch.Tensor:
        """
        Given a list of sentences, return the [CLS] embedding of their concatenation.
        """
        joined_text = " ".join(sentences)
        encoded = self.tokenizer(
            joined_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        input_ids = encoded['input_ids'].to(self.encoder.device)
        attention_mask = encoded['attention_mask'].to(self.encoder.device)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [B, H]
        return cls_embedding.squeeze(0)  # [H]


#* triple set에서 일정 수의 트리플 랜덤하게 샘플링
def sample_triples(triples, scores, sample_size):   # random sampling
    """
    Args:
        triples: List[Tuple[str, str, str]]
        scores: List[float] or Tensor
        sample_size: int

    Returns:
        sampled_triples, sampled_scores
    """
    if len(triples) <= sample_size:
        return triples, scores  # 샘플 수보다 작으면 그대로 반환

    indices = random.sample(range(len(triples)), sample_size)
    sampled_triples = [triples[i] for i in indices]
    if isinstance(scores, torch.Tensor):    # 왜 필요?
        sampled_scores = scores[indices]
    else:
        sampled_scores = [scores[i] for i in indices]
    return sampled_triples, sampled_scores
