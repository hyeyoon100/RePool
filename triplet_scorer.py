import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple


class BERTTripleScorer(nn.Module):
    def __init__(self, model_name: str = "bert-base-cased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size
        self.plausibility_head = nn.Linear(self.hidden_size, 1)

    def encode_triples(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
        texts = [f"{h} {r} {t}" for h, r, t in triples] #! <- node id to token 으로 변환 필요
        # 텍스트로 변환
        # texts = [f"{node_id_to_token[h]} {node_id_to_token[r]} {node_id_to_token[t]}" for (h, r, t) in triples]
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.bert(**inputs)
        cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, H]
        return cls_embeddings

    def forward(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
        cls_embeddings = self.encode_triples(triples)
        scores = self.plausibility_head(cls_embeddings).squeeze(-1)  # [B]
        return scores

    # def compute_loss(self, pos_triples, neg_triples, margin=1.0):
    #     pos_scores = self.forward(pos_triples)
    #     neg_scores = self.forward(neg_triples)
    #     target = torch.ones_like(pos_scores)
    #     loss = nn.MarginRankingLoss(margin=margin)(pos_scores, neg_scores, target)
    #     return loss, pos_scores, neg_scores


def extract_candidate_triples(data):
    edge_index = data.edge_index
    node_type = data.node_type

    rel_ids = (node_type == 1).nonzero(as_tuple=True)[0]
    triples = []

    for r in rel_ids.tolist():
        head_ents = edge_index[0][edge_index[1] == r]  # h → r
        tail_ents = edge_index[1][edge_index[0] == r]  # r → t
        for h in head_ents.tolist():
            for t in tail_ents.tolist():
                if h != t:
                    triples.append((h, r, t))
    return triples

def node_id_to_token_map(entity_tokens: List[str], relation_types: List[str]) -> dict:
    """
    Build a mapping from node index (after concatenation) to string token.

    Args:
        entity_tokens: List of entity token strings (length = num_entity)
        relation_types: List of relation type strings (length = num_relation)

    Returns:
        node_id_to_token: Dict[int, str] mapping from node index to token
    """
    entity_map = {i: tok for i, tok in enumerate(entity_tokens)}
    relation_map = {i + len(entity_tokens): rel for i, rel in enumerate(relation_types)}
    return {**entity_map, **relation_map}


def convert_id_triples_to_text(triples: List[Tuple[int, int, int]], node_id_to_token: dict) -> List[Tuple[str, str, str]]:
    return [(node_id_to_token[h], node_id_to_token[r], node_id_to_token[t]) for (h, r, t) in triples]


#* top / bottom 트리플셋 추출
def split_triples_by_score(triples, scores, top_k=10):
    """
    triples: List[Tuple[str, str, str]]
    scores: Tensor of shape [N]
    top_k: int - number of top triples to keep

    Returns:
        top_triples, top_scores, bottom_triples, bottom_scores
    """
    assert len(triples) == scores.size(0), "Mismatch between triples and scores"

    sorted_indices = torch.argsort(scores, descending=True)
    top_idx = sorted_indices[:top_k]
    bottom_idx = sorted_indices[top_k:]

    top_triples = [triples[i] for i in top_idx]
    top_scores = scores[top_idx]

    bottom_triples = [triples[i] for i in bottom_idx]
    bottom_scores = scores[bottom_idx]

    return top_triples, top_scores, bottom_triples, bottom_scores
