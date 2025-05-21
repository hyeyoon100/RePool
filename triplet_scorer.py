import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict, Set
from torch_geometric.data import Data
import random

from onelayer_bert import OneLayerBertModel

#! one-layer BERT, segmentation embedding 추가, 한번에 하나씩 처리하도록 변경
# class BERTTripleScorer(nn.Module):
    # def __init__(self, model_name: str = "bert-base-cased"):
    #     super().__init__()
    #     self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    #     self.bert = AutoModel.from_pretrained(model_name)
    #     self.hidden_size = self.bert.config.hidden_size
    #     self.plausibility_head = nn.Linear(self.hidden_size, 1)

    # def encode_triples(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
    #     texts = [f"{h} {r} {t}" for h, r, t in triples] #! <- node id to token 으로 변환 필요
    #     # 텍스트로 변환
    #     # texts = [f"{node_id_to_token[h]} {node_id_to_token[r]} {node_id_to_token[t]}" for (h, r, t) in triples]
    #     inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    #     # 모든 입력 텐서를 self.bert의 디바이스로 이동
    #     inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        
    #     outputs = self.bert(**inputs)
    #     cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [B, H]
    #     return cls_embeddings

    # def forward(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
    #     cls_embeddings = self.encode_triples(triples)
    #     scores = self.plausibility_head(cls_embeddings).squeeze(-1)  # [B]
    #     return scores

    # def compute_loss(self, pos_triples, neg_triples, margin=1.0):
    #     pos_scores = self.forward(pos_triples)
    #     neg_scores = self.forward(neg_triples)
    #     target = torch.ones_like(pos_scores)
    #     loss = nn.MarginRankingLoss(margin=margin)(pos_scores, neg_scores, target)
    #     return loss, pos_scores, neg_scores

class BERTTripleScorer(nn.Module):
    def __init__(self, hidden_size, device="cuda"):
        super().__init__()
        self.device = device
        self.bert = OneLayerBertModel(pretrained_model_name="bert-base-cased").to(device)
        self.hidden_size = hidden_size

        # 학습 가능한 CLS 벡터 정의 (shared across all inputs)
        self.cls_embedding = nn.Parameter(torch.randn(1, 1, hidden_size))  # [1, 1, D]
        self.compound_embedding = nn.Parameter(torch.randn(hidden_size))   # learnable compound vector

        self.plausibility_head = nn.Linear(self.hidden_size, 1)
        self.relation_embeddings = None  # relation embeddings를 저장할 속성 추가
        self.to(self.device)

    def encode_token_triples(self,
                       triples: List[Tuple[int, int, int, str]],
                       token_embeddings: torch.Tensor,  # [L+R, D]
                       relation_embeddings: torch.Tensor,  # [R, D]
                       actual_entity_length: int,  # 실제 엔티티 토큰 개수
                       edge_type_emb: torch.Tensor  # [num_edge_types, D]
                       ) -> torch.Tensor:
        """
        Args:
            triples: list of (head_id, rel_id, tail_id, r_type)
            token_embeddings: [L+R, D] - 전체 노드 임베딩 (entity + relation)
            relation_embeddings: [R, D] - relation type 임베딩 (R = len(relation_types))
            actual_entity_length: int - 실제 엔티티 토큰의 개수
            edge_type_emb: [num_edge_types, D] - 엣지 타입 임베딩
        Returns:
            cls_embeddings: [B, D]
        """
        triple_toks = []
        seg_ids = []
        cls_embeddings_list = []  # 각 트리플의 CLS 벡터를 저장할 리스트

        # entity 임베딩만 분리 (실제 엔티티 토큰 개수만큼)
        token_embeddings = token_embeddings.to(self.device)
        relation_embeddings = relation_embeddings.to(self.device)
        edge_type_emb = edge_type_emb.to(self.device)

        for h_id, edge_type_id, t_id in triples:
            # 1. Head 노드 임베딩 선택
            if h_id < actual_entity_length:  # entity 노드
                h_tok = token_embeddings[h_id]
            else:  # relation 노드
                h_tok = relation_embeddings[h_id - actual_entity_length]
                
            # 2. Edge type 임베딩 선택
            edge_tok = edge_type_emb[edge_type_id]
            
            # 3. Tail 노드 임베딩 선택
            if t_id < actual_entity_length:  # entity 노드
                t_tok = token_embeddings[t_id]
            else:  # relation 노드
                t_tok = relation_embeddings[t_id - actual_entity_length]
            
            # 트리플 구성
            triple_tok = torch.stack([h_tok, edge_tok, t_tok], dim=0)  # [3, D]
            
            # 5. CLS 토큰 추가
            cls_token = self.cls_embedding.squeeze(0)  # [1, D]
            input_triple_embs = torch.cat([cls_token, triple_tok], dim=0)  # [4, D]
            
            # 6. Segment IDs 생성
            seg_ids = torch.tensor([0, 1, 2, 3], device=self.device)  # [CLS, HEAD, EDGE, TAIL]
            
            # 7. OneLayerBERT로 인코딩 (단일 트리플)
            input_triple_embs = input_triple_embs.unsqueeze(0)  # [1, 4, D]
            seg_ids = seg_ids.unsqueeze(0)  # [1, 4]
            
            cls_embedding = self.bert(input_embs=input_triple_embs, seg_ids=seg_ids)  # [1, D]
            cls_embeddings_list.append(cls_embedding)

        if not cls_embeddings_list:  # 유효한 트리플이 없는 경우
            return torch.empty(0, self.hidden_size, device=self.device)

        # 모든 트리플의 CLS 벡터를 하나의 텐서로 결합
        cls_embeddings = torch.cat(cls_embeddings_list, dim=0)  # [num_triples, D]
        
        print(f"Encoded {len(triples)} triples individually")
        print(f"CLS embeddings shape: {cls_embeddings.shape}")
        
        return cls_embeddings  # [num_triples, D]

    def encode_span_triples(self,
                          triples: List[Tuple[int, int, int]],
                          span_embeddings: torch.Tensor,   # [num_spans, D]
                          token_embeddings: torch.Tensor,  # [num_tokens, D]
                          relation_embeddings: torch.Tensor = None  # [R, D]
                          ) -> torch.Tensor:
        """
        Return CLS embeddings for triples that may contain token and span nodes.
        
        Args:
            triples: List[Tuple[int, int, int]] - (head_id, rel_id, tail_id, r_type)
            span_embeddings: [num_spans, D] - 스팬 노드 임베딩
            token_embeddings: [num_tokens, D] - 토큰 임베딩
            relation_embeddings: [R, D] - relation 임베딩
            
        Returns:
            cls_embeddings: [num_triples, D] - 각 트리플의 CLS 임베딩
        """
        cls_embeddings_list = []
        num_tokens = token_embeddings.size(0)  # 토큰 노드 수
        num_relations = relation_embeddings.size(0) if relation_embeddings is not None else 0  # relation 노드 수
        
        for h_id, rel_id, t_id in triples:
            # 1. Head 노드 임베딩 선택
            if h_id < num_tokens:  # token 노드
                h_tok = token_embeddings[h_id]
            else:  # span 노드
                span_idx = h_id - (num_tokens + num_relations)  # relation 노드 수도 고려
                if span_idx >= span_embeddings.size(0):  # 유효성 검사
                    continue
                h_tok = span_embeddings[span_idx]
            
            # 2. Relation 임베딩 선택
            if relation_embeddings is not None:
                r_tok = relation_embeddings[rel_id]
            else:
                continue  # relation 임베딩이 없으면 스킵
            
            # 3. Tail 노드 임베딩 선택
            if t_id < num_tokens:  # token 노드
                t_tok = token_embeddings[t_id]
            else:  # span 노드
                span_idx = t_id - (num_tokens + num_relations)  # relation 노드 수도 고려
                if span_idx >= span_embeddings.size(0):  # 유효성 검사
                    continue
                t_tok = span_embeddings[span_idx]
            
            # 4. 트리플 구성
            triple_tok = torch.stack([h_tok, r_tok, t_tok], dim=0)  # [3, D]
            
            # 5. CLS 토큰 추가
            cls_token = self.cls_embedding.squeeze(0)  # [1, D]
            input_triple_embs = torch.cat([cls_token, triple_tok], dim=0)  # [4, D]
            
            # 6. Segment IDs 생성
            seg_ids = torch.tensor([0, 1, 2, 3], device=self.device)  # [CLS, HEAD, REL, TAIL]
            
            # 7. OneLayerBERT로 인코딩 (단일 트리플)
            input_triple_embs = input_triple_embs.unsqueeze(0)  # [1, 4, D]
            seg_ids = seg_ids.unsqueeze(0)  # [1, 4]
            
            cls_embedding = self.bert(input_embs=input_triple_embs, seg_ids=seg_ids)  # [1, D]
            cls_embeddings_list.append(cls_embedding)

        if not cls_embeddings_list:  # 유효한 트리플이 없는 경우
            return torch.empty(0, self.hidden_size, device=self.device)

        # 모든 트리플의 CLS 벡터를 하나의 텐서로 결합
        cls_embeddings = torch.cat(cls_embeddings_list, dim=0)  # [num_triples, D]
        
        return cls_embeddings

    # token-level triple scoring
    def forward(self,
            cls_vectors: torch.Tensor = None,  # 새로운 옵션
            **kwargs  # 기존 파라미터들은 옵션으로 유지
            ) -> torch.Tensor:
        if cls_vectors is not None:
            # 이미 CLS 벡터가 있다면 바로 스코어 계산
            scores = self.plausibility_head(cls_vectors).squeeze(-1)
        else:
            # 기존 방식대로 인코딩부터 수행
            cls_vectors = self.encode_token_triples(**kwargs)
            scores = self.plausibility_head(cls_vectors).squeeze(-1)
        return scores

    # span-level triple scoring
    def score_span_triples(self,
                          triples: List[Tuple[int, int, int, str]],
                          token_embeddings: torch.Tensor,  # [num_tokens, D]
                          span_embeddings: torch.Tensor,   # [num_spans, D]
                          relation_embeddings: torch.Tensor,  # [R, D]
                          num_tokens: int  # 실제 토큰 개수 (스팬 노드 제외)
                          ) -> torch.Tensor:
        """
        Return plausibility scores for triples that may contain token and span nodes.
        
        Args:
            triples: List[Tuple[int, int, int, str]] - (head_id, rel_id, tail_id, r_type)
            token_embeddings: [num_tokens, D] - 토큰 임베딩
            span_embeddings: [num_spans, D] - 스팬 노드 임베딩
            relation_embeddings: [R, D] - relation 임베딩
            num_tokens: int - 실제 토큰 개수
            
        Returns:
            scores: [num_triples] - 각 트리플의 plausibility 점수
        """
        cls_embeddings = self.encode_span_triples(
            triples,
            span_embeddings,
            token_embeddings
        ).to(self.device)  # [B, D]
        
        # 유효한 트리플이 없는 경우 모든 트리플에 대해 매우 낮은 점수 할당
        if cls_embeddings.size(0) == 0:
            return torch.full((len(triples),), -1e9, device=self.device)
        
        # 필터링된 트리플에 대해서는 매우 낮은 점수 할당
        scores = torch.full((len(triples),), -1e9, device=self.device)
        valid_scores = self.plausibility_head(cls_embeddings).squeeze(-1)  # [valid_B]
        
        # 유효한 트리플 인덱스 추적
        valid_indices = []
        num_spans = span_embeddings.size(0)
        
        for i, (h_id, r_id, t_id, r_type) in enumerate(triples):
            # head와 tail이 모두 유효한 범위 내에 있는지 확인
            is_valid = True
            
            if h_id >= num_tokens:  # span node
                if h_id - num_tokens >= num_spans:
                    is_valid = False
            if t_id >= num_tokens:  # span node
                if t_id - num_tokens >= num_spans:
                    is_valid = False
                    
            if is_valid:
                valid_indices.append(i)
        
        # 유효한 트리플에 대해서만 실제 점수 할당
        scores[valid_indices] = valid_scores
        
        return scores

#* token-level triple candidates
#! inverse 엣지는 X
def extract_candidate_triples(graph):
    """
    Extract candidate triples from the graph with a maximum limit.
    
    Args:
        data: PyG Data object
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    edge_index = graph.edge_index
    edge_types = graph.edge_type   
    node_types = graph.node_type
    
    CAN_FORM_SUBJECT_OF = 0
    CAN_FORM_OBJECT_OF = 1
    CAN_FORM_COMPOUND_WITH = 2

    triples = []
    # 각 엣지 타입에 대해 처리
    for e in range(edge_index.size(1)):
        src_idx = edge_index[0, e].item()  # 시작 노드
        dst_idx = edge_index[1, e].item()  # 끝 노드
        edge_type = edge_types[e].item()
        
        src_type = node_types[src_idx].item()  # 시작 노드 타입
        dst_type = node_types[dst_idx].item()  # 끝 노드 타입
        
        # 1. token -> relation ("can form subject of")
        if edge_type == CAN_FORM_SUBJECT_OF and src_type == 0 and dst_type == 1:
            triples.append((src_idx, CAN_FORM_SUBJECT_OF, dst_idx))
            
        # 2. relation -> token ("can form object of")
        elif edge_type == CAN_FORM_OBJECT_OF and src_type == 1 and dst_type == 0:
            triples.append((src_idx, CAN_FORM_OBJECT_OF, dst_idx))
            
        # 3. token -> token ("can form compound with")
        elif edge_type == CAN_FORM_COMPOUND_WITH and src_type == 0 and dst_type == 0:
            triples.append((src_idx, CAN_FORM_COMPOUND_WITH, dst_idx))
        
    return triples


# def extract_candidate_triples(data, compound_edge_type: int = 0, max_triples: int = 10000):
#     """
#     Extract candidate triples from the graph with a maximum limit.
    
#     Args:
#         data: PyG Data object
#         compound_edge_type: int - compound 엣지의 타입 ID
#         max_triples: int - 최대 트리플 수 (기본값 10000)
#     """

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     edge_index = data.edge_index
#     edge_type = data.edge_type    # compound_with_inv 제외시키기 위함
#     node_type = data.node_type

#     # === (1) Semantic relation 기반 triple 생성 ===

#     # entity 노드 (type 0)와 relation 노드 (type 1) 찾기
#     entity_ids = (node_type == 0).nonzero(as_tuple=True)[0]
#     rel_ids = (node_type == 1).nonzero(as_tuple=True)[0]

#     # edge_index와 같은 device로 이동
#     edge_index = edge_index.to(device)
#     entity_ids = entity_ids.to(device)
#     rel_ids = rel_ids.to(device)
#     edge_type = edge_type.to(device)  # edge_type도 같은 device로 이동
#     node_type = node_type.to(device)  # node_type도 같은 device로 이동


#     if len(rel_ids) == 0:
#         print("Warning: No relation nodes (type 1) found!")
#         return []
        
#     triples = []
    
#     # 모든 semantic 트리플 생성
#     for r in rel_ids.tolist(): 
#         # 엔티티 노드만 필터링하여 선택
#         head_ents = edge_index[0][(edge_index[1] == r) & torch.isin(edge_index[0], entity_ids)]  # h → r
#         tail_ents = edge_index[1][(edge_index[0] == r) & torch.isin(edge_index[1], entity_ids)]  # r → t
    
        
#         # relation 노드 ID를 relation type 인덱스로 변환
#         rel_type_id = r - rel_ids[0].item()  # 첫 번째 relation 노드의 ID를 빼서 0부터 시작하는 인덱스로 변환
        
#         for h in head_ents.tolist():
#             for t in tail_ents.tolist():
#                 if h != t:  # 자기 자신과의 연결 방지
#                     triples.append((h, rel_type_id, t, "semantic"))  # relation type 인덱스 사용

#     # === (2) Compound edge 기반 triple 생성 ===
#     for edge_id in range(edge_index.size(1)):
#         if edge_type[edge_id] == compound_edge_type:
#             h = edge_index[0, edge_id].item()
#             t = edge_index[1, edge_id].item()
#             if h != t:
#                 triples.append((h, compound_edge_type, t, "compound"))

#     total_triples = len(triples)
#     print(f"\n=== 트리플 추출 통계 ===")
#     print(f"전체 생성된 트리플 수: {total_triples}")
#     print(f"- Semantic 트리플 수: {sum(1 for t in triples if t[3] == 'semantic')}")
#     print(f"- Compound 트리플 수: {sum(1 for t in triples if t[3] == 'compound')}")

#     # 최대 개수 제한
#     if total_triples > max_triples:
#         print(f"\n트리플 수가 상한선({max_triples})을 초과하여 랜덤 샘플링을 수행합니다.")
#         triples = random.sample(triples, max_triples)
#         print(f"샘플링 후 트리플 수: {len(triples)}")
#         print(f"- Semantic 트리플 수: {sum(1 for t in triples if t[3] == 'semantic')}")
#         print(f"- Compound 트리플 수: {sum(1 for t in triples if t[3] == 'compound')}")
    
#     return triples

def node_id_to_token_map(entity_tokens: List[str], relation_types: List[str]) -> dict:
    """
    Build a mapping from node index to string token.

    Args:
        entity_tokens: List of entity token strings (length = num_entity)
        relation_types: List of relation type strings (length = num_relation)

    Returns:
        node_id_to_token: Dict[int, str] mapping from node index to token
    """
    # Entity tokens는 0부터 시작하는 인덱스 사용
    entity_map = {i: tok for i, tok in enumerate(entity_tokens)}
    
    # Relation은 별도의 매핑 생성 (relation_types 인덱스 기준)
    relation_map = {i: rel for i, rel in enumerate(relation_types)}
    
    return {
        "entity": entity_map,
        "relation": relation_map
    }



def convert_id_triples_to_text(triples: List[Tuple[int, int, int]], node_id_to_token: dict) -> List[Tuple[str, str, str]]:
    """
    Convert token-level triples with node IDs to text triples.
    
    Args:
        triples: List[Tuple[int, int, int]] - (head_id, edge_type_id, tail_id)
        node_id_to_token: dict - entity와 relation에 대한 별도의 매핑을 포함하는 딕셔너리
    
    Returns:
        List[Tuple[str, str, str]] - (head_text, edge_type_text, tail_text)
    """

    entity_map = node_id_to_token["entity"]
    edge_type_map = {
        0: "can_form_subject_of",
        1: "can_form_object_of",
        2: "can_form_compound_with"
    }
    
    converted_triples = []
    for (h, edge_type_id, t) in triples:
        # head와 tail 노드 텍스트 변환
        if h < len(entity_map):  # entity 노드
            h_text = entity_map[h]
        else:  # relation 노드
            h_text = node_id_to_token["relation"][h - len(entity_map)]
            
        if t < len(entity_map):  # entity 노드
            t_text = entity_map[t]
        else:  # relation 노드
            t_text = node_id_to_token["relation"][t - len(entity_map)]
        
        # edge type 변환
        edge_type_text = edge_type_map[edge_type_id]
            
        converted_triples.append((h_text, edge_type_text, t_text))
    
    return converted_triples

def convert_span_triples_to_text(triples: List[Tuple[int, int, int, str]], 
                               node_id_to_token: dict,
                               span_id_to_token_indices: Dict[int, List[int]],
                               num_tokens: int
                            ) -> List[Tuple[str, str, str, str]]:
    """
    Convert span-level triples to text.
    
    Args:
        triples: List[Tuple[int, int, int, str]] - (head_id, rel_id, tail_id, rel_type)
        node_id_to_token: dict - entity와 relation에 대한 매핑
        span_id_to_token_indices: Dict[int, List[int]] - span_id에 포함된 token indices
        num_tokens: int - 개별 토큰의 수 (span 노드 제외)
    """
    converted_triples = []
    for (h, r, t) in triples:
        # head text 변환
        if h < num_tokens:  # token node
            h_text = node_id_to_token.get(h, f"UNK_{h}")
        else:  # span node
            token_indices = span_id_to_token_indices.get(h, [])
            h_text = " ".join([node_id_to_token.get(idx, f"UNK_{idx}") for idx in token_indices])

        # tail text 변환
        if t < num_tokens:  # token node
            t_text = node_id_to_token.get(t, f"UNK_{t}")
        else:  # span node
            token_indices = span_id_to_token_indices.get(t, [])
            t_text = " ".join([node_id_to_token.get(idx, f"UNK_{idx}") for idx in token_indices])

        # relation text 변환
        # relation 노드 id는 num_tokens ~ num_tokens+num_relations-1
        r_text = node_id_to_token.get(r, f"UNK_{r}")

        converted_triples.append((h_text, r_text, t_text))
    return converted_triples


#* top / bottom 트리플셋 추출
def split_triples_by_score(triples, scores, threshold: float, return_indices=False):
    """
    triples: List[Tuple[str, str, str]]
    scores: Tensor of shape [N]
    threshold: float — threshold to split top/bottom
    return_indices: whether to return original indices of top/bottom

    Returns:
        top_triples, top_scores, bottom_triples, bottom_scores
        (+ optional top_indices, bottom_indices)
    """
    assert len(triples) == scores.size(0), "Mismatch between triples and scores"

    top_mask = scores >= threshold # score가 threshold 이상인 위치에 True, 그 외는 False
    bottom_mask = scores < threshold

    top_indices = torch.nonzero(top_mask, as_tuple=True)[0] # True인 위치를 반환
    bottom_indices = torch.nonzero(bottom_mask, as_tuple=True)[0]

    top_triples = [triples[i] for i in top_indices]
    bottom_triples = [triples[i] for i in bottom_indices]

    print("top_indices: ", top_indices)
    print("bottom_indices: ", bottom_indices)

    top_scores = scores[top_indices]
    bottom_scores = scores[bottom_indices]

    if return_indices:
        return top_triples, top_scores, bottom_triples, bottom_scores, top_indices.tolist(), bottom_indices.tolist()
    else:
        return top_triples, top_scores, bottom_triples, bottom_scores

#* token-level 하위 triple 엣지 삭제
def remove_edges_of_bottom_token_triples(
    graph: Data,
    bottom_triples: List[Tuple[int, int, int]],
) -> Data:
    """
    Remove specific edges in the graph based on bottom triples and their type.

    Args:
        data: PyG Data object with edge_index and edge_type
        bottom_triples: list of (head_id, rel_id, tail_id, r_type)

    Returns:
        data: PyG Data with filtered edge_index and edge_type
    """

    edge_index = graph.edge_index  # [2, E]
    edge_type = graph.edge_type    # [E]

    # 어떤 엣지를 유지할지(True) 삭제할지(False
    # 모든 엣지 유지로 초기화해두고 뒤에서 업데이트
    keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)
    
    for h, edge_type_id, t in bottom_triples:
        # 해당 엣지 찾기: head -> tail 방향이면서 edge_type이 일치하는 엣지
        match = (
            (edge_index[0] == h) & 
            (edge_index[1] == t) & 
            (edge_type == edge_type_id)
        )
        keep_mask = keep_mask & (~match)
    
    # 필터링 적용
    graph.edge_index = edge_index[:, keep_mask]
    graph.edge_type = edge_type[keep_mask]

    print("after removing edge")
    print("edge_index: ", graph.edge_index)
    
    return graph

    

#* 하위 트리플 엣지 삭제 _ Span-level (e,r,e) 트리플
def remove_edges_of_bottom_triples(
    data: Data,
    bottom_triples: List[Tuple[int, int, int]]
) -> Data:
    """
    Remove specific edges in the graph based on bottom triples.

    Args:
        data: PyG Data object with edge_index and edge_type
        bottom_triples: list of (head_id, rel_id, tail_id)

    Returns:
        data: PyG Data with filtered edge_index and edge_type
    """
    edge_index = data.edge_index  # [2, E]
    edge_type = data.edge_type    # [E]

    keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=edge_index.device)

    for h, r, t in bottom_triples:
        # 해당 triple의 edge type과 일치하는 엣지 찾아서 삭제
        match = (
            (edge_index[0] == h) & (edge_index[1] == t) & (edge_type == r)
        )
        keep_mask = keep_mask & (~match)

    # Apply filtering
    data.edge_index = edge_index[:, keep_mask]
    data.edge_type = edge_type[keep_mask]

    return data




#* span-level triple candidates
def extract_span_token_candidate_triples(
    token_ids: List[int],
    span_ids: List[int],
    node_relation_candidates: Dict[int, Dict[int, Set[int]]],  # {node_id: {edge_type: set(rel_id)}}
    span_id_to_token_indices: Dict[int, List[int]],
    relation_types: List[str],
    max_triples: int = 10000
) -> List[Tuple[int, int, int]]:
    """
    Generate candidate triples including:
    - token-token, token-span, span-token, span-span
    모두 head는 can_form_subject_of(0), tail은 can_form_object_of(1)로 같은 relation에 연결된 경우만 triple로 생성
    
    With constraints:
    - No self-loops (head != tail)
    - No triples between span and its internal tokens (for token-span, span-token)
    
    Args:
        token_ids: List of token node IDs
        span_ids: List of span node IDs
        node_relation_candidates: {node_id: {edge_type: set(rel_id)}}
        span_id_to_token_indices: Mapping from span_id to list of contained token IDs
        relation_types: List of relation type strings
    Returns:
        triples: List of (head_id, rel_id, tail_id) tuples
    """
    triples = []

    # 1. token-token
    for head in token_ids:
        for tail in token_ids:
            if head == tail:
                continue
            rels_head = set(node_relation_candidates.get(head, {}).get(0, set()))
            rels_tail = set(node_relation_candidates.get(tail, {}).get(1, set()))
            shared_rels = rels_head & rels_tail
            for rel_id in shared_rels:
                triples.append((head, rel_id, tail))

    # 2. token-span
    for head in token_ids:
        for tail in span_ids:
            # token이 span 내부 토큰이면 제외
            if head in span_id_to_token_indices.get(tail, []):
                continue
            rels_head = set(node_relation_candidates.get(head, {}).get(0, set()))
            rels_tail = set(node_relation_candidates.get(tail, {}).get(1, set()))
            shared_rels = rels_head & rels_tail
            for rel_id in shared_rels:
                triples.append((head, rel_id, tail))

    # 3. span-token
    for head in span_ids:
        for tail in token_ids:
            # token이 span 내부 토큰이면 제외
            if tail in span_id_to_token_indices.get(head, []):
                continue
            rels_head = set(node_relation_candidates.get(head, {}).get(0, set()))
            rels_tail = set(node_relation_candidates.get(tail, {}).get(1, set()))
            shared_rels = rels_head & rels_tail
            for rel_id in shared_rels:
                triples.append((head, rel_id, tail))

    # 4. span-span
    for head in span_ids:
        for tail in span_ids:
            if head == tail:
                continue
            rels_head = set(node_relation_candidates.get(head, {}).get(0, set()))
            rels_tail = set(node_relation_candidates.get(tail, {}).get(1, set()))
            shared_rels = rels_head & rels_tail
            for rel_id in shared_rels:
                triples.append((head, rel_id, tail))

    # 5. max_triples 제한
    if len(triples) > max_triples:
        print(f"\n트리플 수가 상한선({max_triples})을 초과하여 랜덤 샘플링을 수행합니다.")
        triples = random.sample(triples, max_triples)

    print(f"\n=== 생성된 트리플 통계 ===")
    print(f"전체 트리플 수: {len(triples)}")
    print(f"- Token-Token 트리플 수: {sum(1 for t in triples if t[0] in token_ids and t[2] in token_ids)}")
    print(f"- Token-Span 트리플 수: {sum(1 for t in triples if t[0] in token_ids and t[2] in span_ids)}")
    print(f"- Span-Token 트리플 수: {sum(1 for t in triples if t[0] in span_ids and t[2] in token_ids)}")
    print(f"- Span-Span 트리플 수: {sum(1 for t in triples if t[0] in span_ids and t[2] in span_ids)}")

    return triples
