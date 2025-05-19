import torch
from torch import nn
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple, Dict, Set
import random


def build_bipartite_graph(all_node_emb, keep_ids, num_relation, bidirectional=True, keep_mask=None):
    """
    Build a PyG homogeneous Data object with all entity nodes included, but only
    connect pre-filtered entity nodes (keep_ids) to relation nodes.

    Args:
        entity_emb (Tensor): [num_entity, dim] (pre-filtered 된 토큰도 포함)
        keep_ids (List[int]): indices of entity nodes to connect to relations (pre-filter 된 토큰 제외시키기 위함)
        relation_emb (Tensor): [num_relation, dim] (semantic type relations)
        bidirectional (bool): Whether to add reverse edges
        use_node_type (bool): Whether to attach node type annotation
        
    +   Additional edges between sequential entity tokens (connected/connected_inverse)

    Returns:
        data (torch_geometric.data.Data): PyG graph object
    """

    # 엣지 타입 정의 
    # COMPOUND_WITH = 0
    # COMPOUND_WITH_INV = 1
    # SEMANTIC = 2 # relation_types[]
    CAN_FORM_SUBJECT_OF = 0
    CAN_FORM_OBJECT_OF = 1
    CAN_FORM_COMPOUND_WITH = 2
    

    num_nodes = all_node_emb.size(0)
    num_entity = num_nodes - num_relation
    print('num_nodes: ', num_nodes)
    print('num_entity: ', num_entity)
    print('num_relation: ', num_relation)

    edge_index = [] 
    edge_type = []  # 각 엣지에 해당하는 relation type ID 사용 

    # === entity ↔ relation
    # token -can form subject of-> relation
    # relation -can form object of-> token
    for r in range(num_relation):
        rel_index = num_entity + r
        for e in keep_ids:
            # entity -> relation (can form subject of)
            edge_index.append([e, rel_index])
            edge_type.append(CAN_FORM_SUBJECT_OF)
            
            # relation -> entity (can form object of)
            edge_index.append([rel_index, e])
            edge_type.append(CAN_FORM_OBJECT_OF)

    
    # === entity ↔ entity (compound_with edges) _ inverse X 
    for i in range(num_entity - 1):
        src = i
        tgt = i + 1
        # src, tgt 모두 prefilter 단계에서 살아남은 것만
        if keep_mask[src] == 1 and keep_mask[tgt] == 1:
            # i → i+1: connected
            edge_index.append([src, tgt])
            edge_type.append(CAN_FORM_COMPOUND_WITH)

    # 텐서 변환
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_type = torch.tensor(edge_type, dtype=torch.long) 

    # 입력 텐서의 device 확인
    device = all_node_emb.device
    
    # Node type 설정
    # entity nodes: type 0
    entity_types = torch.zeros(num_entity, dtype=torch.long, device=device)
    # relation nodes: type 1
    relation_types = torch.ones(num_relation, dtype=torch.long, device=device)
    # 전체 node type 텐서 생성
    node_type = torch.cat([entity_types, relation_types])
    
    # 그래프 생성
    return Data(
        x=all_node_emb,
        edge_index=edge_index,
        edge_type=edge_type,
        node_type=node_type
    )

    return data

#* for input_text_embedding
class TextEncoder(nn.Module):
    def __init__(self, model_name="bert-base-cased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, text: str) -> torch.Tensor:
        """
        Given a list of sentences, return the [CLS] embedding of their concatenation.
        """
        # joined_text = " ".join(sentences)
        # print("joined_text: ", joined_text)
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        # input_ids = encoded['input_ids'].to(self.encoder.device)
        # attention_mask = encoded['attention_mask'].to(self.encoder.device)

        device = self.encoder.device
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.encoder(**encoded)

        # outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [B, H]
        return cls_embedding.squeeze(0)  # [H]


#* triple set에서 일정 수의 트리플 랜덤하게 샘플링
def sample_triples(triples, scores, sample_size, return_indices=False):   # random sampling
    """
    Args:
        triples: List[Tuple[str, str, str]]
        scores: List[float] or Tensor
        sample_size: int
        return_indices: bool - 샘플링된 인덱스 반환 여부

    Returns:
        sampled_triples, sampled_scores
    """
    if len(triples) <= sample_size:
        # 샘플 수보다 작으면 그대로 반환 _ 인덱스 반환하도록 수정 필요
        if return_indices:
            return triples, scores, list(range(len(triples)))
        return triples, scores

    indices = torch.randperm(len(triples))[:sample_size]
    sampled_triples = [triples[i] for i in indices]
    sampled_scores = scores[indices]

    if return_indices:
        return sampled_triples, sampled_scores, indices.tolist()
    return sampled_triples, sampled_scores
    
#* span-level _ span candidate 추출
# 문장별로 span 후보를 추출하도록 함
def get_candidate_spans(sentences, window_size=5, keep_mask=None):
    """
    Generate span candidates within each sentence (start_idx, end_idx),
    excluding single-token spans (start == end).
    
    Args:
        sentences: List[List[str]] — 문장별 토큰 리스트
        window_size: int — 최대 스팬 길이
        keep_mask: token-level prefilter 결과 

    Returns:
        all_spans: List[Tuple[int, int]] — global index 기준 span 후보들
    """
    all_spans = []
    offset = 0  # global index로 변환할 때 사용할 문장 시작 위치
    for sent in sentences:
        L = len(sent)
        for start in range(L):
            for end in range(start + 1, min(start + window_size, L)):
                start_global = start + offset
                end_global = end + offset
                if keep_mask is not None:
                    span_range = range(start + offset, end + offset + 1)
                    if any(keep_mask[i] == 0 for i in span_range):
                        continue
                all_spans.append((start_global, end_global))
        offset += L

    return all_spans


def get_span_embeddings(token_embeddings, candidate_spans):
    """
    Args:
        token_embeddings: [L, D] — 전체 토큰 임베딩
        candidate_spans: List[Tuple[int, int]] — (start_idx, end_idx)
    Returns:
        span_embeddings: [N, D] or [N, 2D] — 각 스팬에 대한 임베딩
        또는 에러 발생 시 None
    """
    span_embs = []
    for start, end in candidate_spans:
        if start > end or end >= token_embeddings.size(0):
            print(f"start: {start}, end: {end}, token_embeddings.size(0): {token_embeddings.size(0)}")
            print("skip")
            continue

        span_tokens = token_embeddings[start:end+1]  # [K, D]

        if span_tokens.size(0) == 0:
            continue  # 빈 span 방지
        
        # 임베딩 구성 방식: mean pooling
        emb = span_tokens.mean(dim=0)
        span_embs.append(emb)

    if len(span_embs) == 0:
        print("Warning: No valid span embeddings generated!")
        return None  # 에러 대신 None 반환
        
    output = torch.stack(span_embs)  # [N, D] or [N, 2D]
    print(">>> Final span_embs shape:", output.shape)
    return output


#* span prefilter (하위 후보는 삭제)
def filter_candidate_spans(span_scores, candidate_spans, threshold=0.5):
    """
    스팬 점수에 따라 후보 스팬을 필터링합니다.
    
    Args:
        span_scores: [1, N_spans] - 각 스팬의 점수
        candidate_spans: List[Tuple[int, int]] - 스팬 후보 리스트 [(start, end), ...]
        threshold: float - 필터링 임계값 (기본값: 0.5)
        
    Returns:
        filtered_spans: List[Tuple[int, int]] - 필터링된 스팬 리스트
        selected_indices: torch.Tensor - 선택된 스팬의 인덱스
    """
    # 점수가 threshold 이상인 스팬만 선택
    span_scores = span_scores.squeeze(0)  # [N_spans]
    selected_mask = span_scores >= threshold
    selected_indices = torch.where(selected_mask)[0]  # 1차원 텐서로 변경
    
    # 선택된 스팬만 필터링
    filtered_spans = [candidate_spans[i.item()] for i in selected_indices]
    
    return filtered_spans, selected_indices


#  토큰 노드와 relation 노드 사이의 엣지 정보
# (token_id, rel_id, rel_node_id) 형태
def extract_token_relation_edges(data):
    """
    Extract edges from token (entity) nodes to relation nodes.
    Considers all edge types:
    - CAN_FORM_SUBJECT_OF (0): token -> relation
    - CAN_FORM_OBJECT_OF (1): relation -> token
    - CAN_FORM_COMPOUND_WITH (2): token -> token
    
    Args:
        data: PyG Data object with:
            - edge_index: [2, num_edges]
            - edge_type: [num_edges] - edge type
            - node_type: [num_nodes] - node type (0: token, 1: relation, 2: span)

    Returns:
        List of (token_node_id, rel_id, rel_node_id)
        - token_node_id: token 노드의 ID
        - rel_id: relation type의 ID
        - rel_node_id: relation 노드의 ID
    """
    edge_index = data.edge_index      # [2, num_edges]
    edge_type = data.edge_type        # [num_edges]
    node_type = data.node_type        # [num_nodes]
    
    print(f"edge_type: {edge_type}")
    token_rel_edges = []
    print("\n=== Token-Relation Edge 추출 ===")
    print(f"전체 엣지 수: {edge_index.size(1)}")
    print(f"전체 노드 수: {node_type.size(0)}")
    
    # 노드 타입별 마스크 생성
    token_mask = (node_type == 0)
    relation_mask = (node_type == 1)
    
    # relation 노드 ID 목록 추출
    relation_node_ids = torch.where(relation_mask)[0]
    if relation_node_ids.size(0) == 0:
        print("Warning: No relation nodes found!")
        return []
    
    # relation node ID -> relation type ID 매핑 생성
    rel_id_mapping = {node_id.item(): idx for idx, node_id in enumerate(relation_node_ids)}
    
    # 엣지 타입 상수
    CAN_FORM_SUBJECT_OF = 0
    CAN_FORM_OBJECT_OF = 1
    CAN_FORM_COMPOUND_WITH = 2
    
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()  # 시작 노드
        tgt = edge_index[1, i].item()  # 끝 노드
        edge_t = edge_type[i].item()   # 엣지 타입
        
        # 각 엣지 타입별 처리
        if edge_t == CAN_FORM_SUBJECT_OF:
            # token -> relation
            if node_type[src] == 0 and node_type[tgt] == 1:
                rel_id = rel_id_mapping[tgt]
                token_rel_edges.append((src, rel_id, tgt))
                
        elif edge_t == CAN_FORM_OBJECT_OF:
            # relation -> token
            if node_type[src] == 1 and node_type[tgt] == 0:
                rel_id = rel_id_mapping[src]
                token_rel_edges.append((tgt, rel_id, src))  # 순서 주의: token이 앞으로
                
        elif edge_t == CAN_FORM_COMPOUND_WITH:
            # token -> token
            if node_type[src] == 0 and node_type[tgt] == 0:
                token_rel_edges.append((src, edge_t, tgt))
    
    # 엣지 타입별 통계
    subject_edges = sum(1 for _, rel_id, _ in token_rel_edges if rel_id == CAN_FORM_SUBJECT_OF)
    object_edges = sum(1 for _, rel_id, _ in token_rel_edges if rel_id == CAN_FORM_OBJECT_OF)
    compound_edges = sum(1 for _, rel_id, _ in token_rel_edges if rel_id == CAN_FORM_COMPOUND_WITH)
    
    print(f"\n추출된 엣지 통계:")
    print(f"- 전체 엣지 수: {len(token_rel_edges)}")
    print(f"- CAN_FORM_SUBJECT_OF: {subject_edges}")
    print(f"- CAN_FORM_OBJECT_OF: {object_edges}")
    print(f"- CAN_FORM_COMPOUND_WITH: {compound_edges}")
    print(f"Relation 노드 수: {len(relation_node_ids)}")
    
    return token_rel_edges

def create_span_relation_edges(token_rel_edges: List[Tuple[int, int, int]], 
                             candidate_spans: List[Tuple[int, int]],
                             span_ids: List[int]) -> List[Tuple[int, int, int]]:
    """
    Create edges between span nodes and relation nodes based on token-relation edges.
    
    Args:
        token_rel_edges: List[Tuple[int, int, int]] - (token_id, rel_id, rel_node_id)
        candidate_spans: List[Tuple[int, int]] - (start_idx, end_idx)
        span_ids: List[int] - span node indices
        
    Returns:
        span_rel_edges: List[Tuple[int, int, int]] - (span_id, rel_id, rel_node_id)
    """
    span_rel_edges = []
    
    # span_id -> 포함된 token indices 매핑 생성
    span_id_to_token_indices = {
        span_ids[i]: list(range(start, end + 1))
        for i, (start, end) in enumerate(candidate_spans)
    }
    
    # 각 span에 대해 포함된 토큰들의 relation 정보 수집
    for span_id, token_list in span_id_to_token_indices.items():
        for token_id in token_list:
            for t_id, rel_id, rel_node_id in token_rel_edges:
                if token_id == t_id:
                    span_rel_edges.append((span_id, rel_id, rel_node_id))
    
    # 중복 제거 (span 내부에 동일한 relation이 여러 번 등장할 경우 대비)
    span_rel_edges = list(set(span_rel_edges))
    
    return span_rel_edges

def extract_span_token_candidate_triples(
    token_ids: List[int],
    span_ids: List[int],
    span_relation_candidates: Dict[int, Set[int]],
    span_id_to_token_indices: Dict[int, List[int]],
    relation_types: List[str],
    max_triples: int = 10000
) -> List[Tuple[int, int, int]]:
    """
    Generate candidate triples including:
    - token-token triples
    - token-span triples (in both directions)
    - span-span triples
    
    With constraints:
    - No self-loops (head != tail)
    - No triples between span and its internal tokens
    
    Args:
        token_ids: List of token node IDs
        span_ids: List of span node IDs
        span_relation_candidates: Mapping from span_id to set of possible relation IDs
        span_id_to_token_indices: Mapping from span_id to list of contained token IDs
        relation_types: List of relation type strings
        max_triples: Maximum number of triples to generate
        
    Returns:
        triples: List of (head_id, rel_id, tail_id, rel_type) tuples
    """
    token_token_triples = []
    token_span_triples = []
    span_span_triples = []
    
    # 1. Token-Token triples
    for subj_id in token_ids:
        for obj_id in token_ids:
            if subj_id != obj_id:
                # 두 토큰이 연결 가능한지 확인
                for rel_id in range(len(relation_types)):
                    token_token_triples.append((subj_id, rel_id, obj_id, "semantic"))
    
    # 2. Token-Span triples
    for token_id in token_ids:
        for span_id in span_ids:
            # Skip if token is inside the span
            if token_id in span_id_to_token_indices.get(span_id, []):
                continue
                
            # Get possible relations for this span
            for rel_id in span_relation_candidates.get(span_id, set()):
                # token → span
                token_span_triples.append((token_id, rel_id, span_id, "semantic"))
                # span → token
                token_span_triples.append((span_id, rel_id, token_id, "semantic"))
    
    # 3. Span-Span triples
    for i, span_id1 in enumerate(span_ids):
        for span_id2 in span_ids[i+1:]:  # avoid self-loops and duplicates
            # Skip if spans overlap
            tokens1 = set(span_id_to_token_indices[span_id1])
            tokens2 = set(span_id_to_token_indices[span_id2])
            if tokens1.intersection(tokens2):
                continue
                
            # Get shared relations between the two spans
            shared_rels = span_relation_candidates.get(span_id1, set()).intersection(
                span_relation_candidates.get(span_id2, set())
            )
            
            for rel_id in shared_rels:
                span_span_triples.append((span_id1, rel_id, span_id2, "semantic"))
                span_span_triples.append((span_id2, rel_id, span_id1, "semantic"))
    
    # 모든 트리플 합치기
    triples = token_token_triples + token_span_triples + span_span_triples
    
    # 4. Apply max_triples limit if needed
    if len(triples) > max_triples:
        print(f"\n트리플 수가 상한선({max_triples})을 초과하여 랜덤 샘플링을 수행합니다.")
        triples = random.sample(triples, max_triples)
        
    # Print statistics
    print(f"\n=== 생성된 트리플 통계 ===")
    print(f"전체 트리플 수: {len(triples)}")
    print(f"- Token-Token 트리플 수: {len(token_token_triples)}")
    print(f"- Token-Span 트리플 수: {len(token_span_triples)}")
    print(f"- Span-Span 트리플 수: {len(span_span_triples)}")
    
    # 샘플링 후 비율 확인
    if len(triples) != len(token_token_triples) + len(token_span_triples) + len(span_span_triples):
        print(f"\n=== 샘플링 후 트리플 비율 ===")
        print(f"- Token-Token 트리플 수: {sum(1 for t in triples if t[0] in token_ids and t[2] in token_ids)}")
        print(f"- Token-Span 트리플 수: {sum(1 for t in triples if (t[0] in token_ids and t[2] in span_ids) or (t[0] in span_ids and t[2] in token_ids))}")
        print(f"- Span-Span 트리플 수: {sum(1 for t in triples if t[0] in span_ids and t[2] in span_ids)}")
    
    return triples
