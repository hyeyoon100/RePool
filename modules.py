import torch
import torch.nn as nn
import json 
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple, Dict, Set
from collections import defaultdict
from abc import ABC, abstractmethod

from data_processor import build_batch
from token_rep import EntityTokenRep, RelationTokenRep, EdgeTypeRep   # 토큰/엔티티 임베딩
from generate_labels import generate_token_labels, generate_span_labels # keep label 생성
from prefilter import TokenFilter, SpanFilter  # 엔티티/스팬 필터
from utils import build_bipartite_graph, get_candidate_spans, sample_triples, TextEncoder, create_span_relation_edges, get_span_embeddings, extract_token_relation_edges, filter_candidate_spans
from triplet_scorer import extract_candidate_triples, BERTTripleScorer, split_triples_by_score, remove_edges_of_bottom_triples, node_id_to_token_map, convert_id_triples_to_text, extract_span_token_candidate_triples, convert_span_triples_to_text, remove_edges_of_bottom_token_triples
from llm_guidance import TripleSetEncoder, verbalize_triples, preference_learning_loss, ask_llm_preference   # LLM-guided alignment/loss
# from aggregation import EntityEmbeddingUpdater, RelationEmbeddingUpdater   # aggregation
from prediction import predict_relations
from loss_functions import RelationLoss, flatten_gold_relations
from prediction import REPredictor



class Dual_View_BipartiteKG_Construction(nn.Module):
    def __init__(self, bert_model_name, hidden_size, device=None):
        super(Dual_View_BipartiteKG_Construction, self).__init__()
        self.hidden_size = hidden_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        self.relation_types = ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "COMPARE", "CONJUNCTION", "EVALUATE-FOR"]
        
        self.entity_token_rep = EntityTokenRep(model_name=bert_model_name, hidden_size=hidden_size).to(self.device)
        self.relation_token_rep = RelationTokenRep(self.relation_types).to(self.device)
        
        # Prefilter
        self.entity_prefilter = TokenFilter(hidden_size, dropout=0.1).to(self.device)
        

    def get_entity_embeddings(self, tokens: List[List[str]], lengths: torch.Tensor):
        """
        토큰들의 BERT 임베딩을 계산합니다.
        
        Args:
            tokens: List[List[str]] - 문장별 토큰 리스트
            lengths: torch.Tensor - 각 문장의 길이
            
        Returns:
            Dict[str, torch.Tensor]:
                - embeddings: [B, L, D] - 토큰 임베딩 (패딩 포함)
                - mask: [B, L] - 패딩 마스크 (1=실제토큰, 0=패딩)
        """
        token_out = self.entity_token_rep(tokens, lengths)
        return token_out

    def get_relation_embeddings(self, relation_ids: torch.Tensor):
        """
        Relation 타입들의 임베딩을 계산합니다.
        
        Args:
            relation_ids: torch.Tensor - relation type IDs
            
        Returns:
            torch.Tensor: [R, D] - relation 임베딩
        """
        rel_emb = self.relation_token_rep(relation_ids)
        return rel_emb

    def get_entity_prefilter(self, entity_embs, entity_mask, sentences, ner_spans):
        """
        엔티티 임베딩에 대해 prefilter를 적용합니다.
        
        Args:
            entity_embs: [num_sents, max_len, D] - 문장별 엔티티 임베딩
            entity_mask: [num_sents, max_len] - 문장별 패딩 마스크
            sentences: List[List[str]] - 문장별 토큰 리스트
            ner_spans: List[List[Tuple[int, int, str]]] - 문장별 NER span 정보
        """
        # 각 문장의 실제 길이 계산
        sent_lengths = [len(sent) for sent in sentences]

        # NER spans 기반으로 token 라벨 생성
        e_labels, total_length = generate_token_labels(sentences, ner_spans)  # List[int], int
        
        # 라벨을 텐서로 변환하고 패딩
        max_len = entity_embs.size(1)  # 최대 문장 길이
        num_sents = entity_embs.size(0)  # 문장 수
        
        # 라벨을 문장별로 다시 나누고 패딩
        padded_labels = torch.zeros((num_sents, max_len), dtype=torch.long, device=entity_embs.device)
        current_pos = 0
        
        for i, length in enumerate(sent_lengths):
            # 현재 문장의 라벨만 선택
            sent_labels = e_labels[current_pos:current_pos + length]
            # 패딩된 위치에 라벨 할당
            padded_labels[i, :length] = torch.tensor(sent_labels, dtype=torch.long, device=entity_embs.device)
            current_pos += length

        # Prefilter 적용 (문장 단위로)
        #& entity_prefilter_loss
        scores, entity_prefilter_loss = self.entity_prefilter(
            entity_embs,  # [num_sents, max_len, D]
            entity_mask,  # [num_sents, max_len]
            labels=padded_labels  # [num_sents, max_len]
        )
        
        #! 조정 필요
        entity_filter_threshold = 0.5
        keep_mask = (scores > entity_filter_threshold).float()  # [num_sents, max_len]

        return {
            # "entity_embs": entity_embs,  # [num_sents, max_len, D]
            "keep_mask": keep_mask,      # [num_sents, max_len]
            "prefilter_loss": entity_prefilter_loss
        }
    # all_node_emb, keep_ids, num_relation, bidirectional=True, use_node_type=True, keep_mask=None
    def construct_graph(self, entity_embs, rel_emb, keep_mask, actual_entity_length):
        # entity_emb = entity_embs[:actual_entity_length]
        # print("entity_emb shape:", entity_emb.shape)
        # print("rel_emb shape:", rel_emb.shape)
        # all_node_emb = torch.cat([entity_emb, rel_emb], dim=0)
        # num_relation = rel_emb.size(0)
        # keep_ids = (keep_mask == 1).nonzero(as_tuple=True)[0].tolist()
        # flat_keep_mask = keep_mask.tolist()
        # 1. 각 문장의 실제 길이만큼만 잘라내고 연결
        entity_embs_list = []
        keep_mask_list = []
        
        for i, length in enumerate(actual_entity_length):
            length = length.item()
            # 현재 문장의 실제 길이만큼만 잘라내기
            entity_embs_list.append(entity_embs[i, :length])  # [length, D]
            keep_mask_list.append(keep_mask[i, :length])  # [length]
        
        # 모든 문장 연결
        flat_entity_embs = torch.cat(entity_embs_list, dim=0)  # [total_length, D]
        flat_keep_mask = torch.cat(keep_mask_list, dim=0)  # [total_length]
        
        # 2. 그래프 구성
        all_node_emb = torch.cat([flat_entity_embs, rel_emb], dim=0)  # [total_length + R, D]
        num_relation = rel_emb.size(0)
        
        # keep mask 처리
        keep_ids = (flat_keep_mask == 1).nonzero(as_tuple=True)[0].tolist()
        flat_keep_mask = flat_keep_mask.tolist()

        graph = build_bipartite_graph(all_node_emb, keep_ids, num_relation, bidirectional=True, keep_mask=flat_keep_mask)
        
        print("graph: ", graph)
        
        return graph
        
class Token_Level_Graph_Pooling(nn.Module):
    def __init__(self, bert_model_name, hidden_size, device=None):
        super(Token_Level_Graph_Pooling, self).__init__()
        self.hidden_size = hidden_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.bert_model_name = bert_model_name
        self.edge_types = ["can_form_subject_of", "can_form_object_of", "can_form_compound_with"]
        self.edgetype_rep = EdgeTypeRep(self.edge_types, lm_model_name=bert_model_name, freeze_bert=True).to(self.device)
        self.triple_scorer = BERTTripleScorer(hidden_size=768).to(self.device)
        self.pooling_threshold = 0.1

        self.tripleset_encoder = TripleSetEncoder(hidden_size=hidden_size).to(self.device)
        self.text_encoder = TextEncoder(model_name=bert_model_name).to(self.device)
        
        # self.sample_size = sample_size
        self.sample_size = 100
        self.cos = nn.CosineSimilarity(dim=-1).to(self.device)

        # validity aware aggregation
        self.update_entity_embedding = EntityEmbeddingUpdater(hidden_dim=hidden_size).to(self.device)
        self.update_relation_embedding = RelationEmbeddingUpdater(hidden_dim=hidden_size).to(self.device)


    def process_token_triples(self, graph):
        # 엣지 타입 인덱스 생성 (0, 1, 2)
        edge_type_indices = torch.arange(len(self.edge_types), device=self.device)
        # 엣지 타입 임베딩 생성
        edge_type_emb = self.edgetype_rep(edge_type_indices)

        # 노드 타입에 따라 임베딩 분리
        all_node_emb = graph.x  # 전체 노드 임베딩
        entity_mask = (graph.node_type == 0)  # 엔티티(토큰) 노드
        relation_mask = (graph.node_type == 1)  # relation 노드
        actual_entity_length = entity_mask.sum().item()  # 엔티티 노드의 실제 개수
        
        # Triple 후보 추출
        # 엣지가 모두 삭제된 경우 예외처리
        edge_index = graph.edge_index
        if edge_index is None or edge_index.numel() == 0 or edge_index.dim() != 2 or edge_index.size(0) != 2:
            print("[Warning] edge_index가 비어있거나 shape이 올바르지 않습니다. 트리플 추출을 스킵합니다.")
            return graph, [], None, [], None, [], None, [], []
        triples = extract_candidate_triples(graph)

        # 먼저 CLS 벡터를 얻고
        triple_cls_vectors = self.triple_scorer.encode_token_triples(
            triples, 
            token_embeddings=all_node_emb[entity_mask],
            relation_embeddings=all_node_emb[relation_mask],
            actual_entity_length=actual_entity_length,
            edge_type_emb=edge_type_emb
        )

        # CLS 벡터를 이용해서 스코어 계산
        token_level_scores = self.triple_scorer(
            cls_vectors=triple_cls_vectors  # 이미 얻은 CLS 벡터만 전달
        )

        if len(token_level_scores) > 0:
            print(f"\n점수 통계:")
            print(f"- 최소값: {token_level_scores.min().item():.4f}")
            print(f"- 최대값: {token_level_scores.max().item():.4f}")
            print(f"- 평균값: {token_level_scores.mean().item():.4f}")
            print(f"- 중앙값: {token_level_scores.median().item():.4f}")
            
        # score로 top/bottom split & 엣지 삭제
        top_triples, top_scores, bottom_triples, bottom_scores, top_indices, bottom_indices = split_triples_by_score(
            triples,
            token_level_scores,
            threshold=self.pooling_threshold,
            return_indices=True
        )
        print(f"\n트리플 분할 결과:")
        print(f"- 상위 트리플 수: {len(top_triples)}")
        print(f"- 하위 트리플 수: {len(bottom_triples)}")

        # 엣지 삭제 전 상태 저장
        num_edges_before = graph.edge_index.size(1)
            
        graph = remove_edges_of_bottom_token_triples(
            graph,
            bottom_triples
        )
            
        # 엣지 삭제 후 상태 확인
        num_edges_after = graph.edge_index.size(1)
        print(f"\n엣지 삭제 결과:")
        print(f"- 삭제 전 엣지 수: {num_edges_before}")
        print(f"- 삭제 후 엣지 수: {num_edges_after}")
        print(f"- 삭제된 엣지 수: {num_edges_before - num_edges_after}")

        print("====================top scores: ", top_scores)

        return graph, triples, triple_cls_vectors, top_triples, top_scores, bottom_triples, bottom_scores, top_indices, bottom_indices
            
    
    def token_llm_alignment(self, 
                      input_text,
                      relation_types,
                      triple_cls_vectors,
                      top_triples,
                      top_scores, 
                      bottom_triples, 
                      bottom_scores, 
                      top_indices, 
                      bottom_indices):
        """
        Args:
            graph: 그래프 데이터
            top_triples: 상위 점수 트리플 리스트
            top_scores: 상위 트리플 점수
            bottom_triples: 하위 점수 트리플 리스트
            bottom_scores: 하위 트리플 점수
            top_indices: 상위 트리플 인덱스
            bottom_indices: 하위 트리플 인덱스
        """
        input_text_embs = []
        for sentences in input_text:
            flattened_text = " ".join(sent for sent in sentences)
            input_text_emb = self.text_encoder(flattened_text)  # [H]
            input_text_embs.append(input_text_emb)
        
        input_text_embs = torch.stack(input_text_embs, dim=0).to(self.device)  # [B, H]
        

        input_text_emb = input_text_embs.mean(dim=0)  # [H]
        
        # 샘플링
        if len(top_triples) == 0:
            print("Warning: No top triples found! Using only bottom triples.")
            sim_top = torch.tensor(-1.0).to(self.device)  # 기본값으로 -1 설정
            top_vector = torch.zeros(self.hidden_size).to(self.device)  # 제로 벡터
        else:
            top_triples, top_scores, sampled_top_idx = sample_triples(
                top_triples, top_scores, self.sample_size, return_indices=True)
            sampled_top_cls = triple_cls_vectors[[top_indices[i] for i in sampled_top_idx]]
            top_vector = self.tripleset_encoder(sampled_top_cls)
            sim_top = self.cos(input_text_emb.unsqueeze(0), top_vector.unsqueeze(0))

        if len(bottom_triples) == 0:
            print("Warning: No bottom triples found! Using only top triples.")
            sim_btm = torch.tensor(-1.0).to(self.device)  # 기본값으로 -1 설정
            bottom_vector = torch.zeros(self.hidden_size).to(self.device)  # 제로 벡터
        else:
            bottom_triples, bottom_scores, sampled_bottom_idx = sample_triples(
                bottom_triples, bottom_scores, self.sample_size, return_indices=True)
            sampled_bottom_cls = triple_cls_vectors[[bottom_indices[i] for i in sampled_bottom_idx]]  # [sample_size, H]
            bottom_vector = self.tripleset_encoder(sampled_bottom_cls)
            sim_btm = self.cos(input_text_emb.unsqueeze(0), bottom_vector.unsqueeze(0))

   
        print(f"코사인 유사도 (top/bottom): {sim_top.item():.4f}/{sim_btm.item():.4f}")
            
        # Triple verbalization
        # 노드 ID를 토큰으로 매핑
        node_id_to_token = node_id_to_token_map(
            entity_tokens=[token for sent in input_text for token in sent],  # 모든 문장의 토큰을 flatten,
            relation_types=relation_types
        )
            
        # 트리플 인덱스를 텍스트로 변환
        top_triples_text = convert_id_triples_to_text(top_triples, node_id_to_token)
        bottom_triples_text = convert_id_triples_to_text(bottom_triples, node_id_to_token)
            
        summary_top = verbalize_triples(top_triples_text, level='token')
        summary_btm = verbalize_triples(bottom_triples_text, level='token')
        
        # LLM 선호도 평가
        preferred = ask_llm_preference(
            input_text=input_text,
            summary_a=summary_top,
            summary_b=summary_btm
        )
            
        print(f"LLM 선호도: {preferred}")
            
        # Preference learning loss 계산
        token_level_llm_loss = preference_learning_loss(sim_top, sim_btm, preferred, input_text_embs)
        # token_level_llm_losses.append(token_level_llm_loss)
            
        print(f"LLM Loss: {token_level_llm_loss.item():.4f}")
        print("="*50)

        return token_level_llm_loss
        
    def validity_aware_aggregation(self, graph, triples, triple_cls_vectors, top_indices):
        """
        엣지 타입에 따라 다르게 노드를 업데이트하는 Validity-Aware Aggregation을 수행합니다.
        
        Args:
            graph: 그래프 데이터
            triples: List[(head_id, edge_type, tail_id)] - 전체 트리플 리스트
            triple_cls_vectors: [num_triples, hidden_size] - 트리플별 CLS 벡터
            top_indices: List[int] - 상위 트리플의 인덱스 리스트
        """
        device = graph.x.device
        # 상위 트리플의 CLS 벡터 추출
        top_cls_vectors = triple_cls_vectors[top_indices].to(device)  # [K, D]
        # 노드 타입에 따른 마스크 생성
        entity_mask = (graph.node_type == 0)  # 엔티티(토큰) 노드
        relation_mask = (graph.node_type == 1)  # relation 노드
        # 업데이트할 노드 임베딩 초기화 (복사본 생성)
        updated_node_emb = graph.x.clone()

        # 1. 모든 entity 노드(head) 업데이트: head entity node가 포함된 triple의 cls 벡터를 모아 weighted average
        for node_id in range(graph.x.size(0)):
            if entity_mask[node_id]:
                # 해당 entity가 head로 등장하는 top triple의 cls 벡터 모으기
                selected_cls = []
                for i, idx in enumerate(top_indices):
                    h, _, _ = triples[idx]
                    if h == node_id:
                        # print("%%%%% head node update: ", node_id)
                        selected_cls.append(top_cls_vectors[i])
                if len(selected_cls) > 0:
                    cls_tensor = torch.stack(selected_cls, dim=0)  # [K, D]
                    weights = torch.softmax(torch.ones(len(selected_cls), device=device), dim=0)  # uniform weight
                    aggregated = torch.sum(cls_tensor * weights.unsqueeze(1), dim=0)  # [D]
                    updated_node_emb[node_id] = aggregated

        # 2. 모든 relation 노드 업데이트: 단순 linear layer transformation
        updated_node_emb = self.update_relation_embedding(
            node_emb=updated_node_emb,
            node_type=graph.node_type
        )

        # 최종 업데이트된 임베딩을 그래프에 반영
        graph.x = updated_node_emb

        return graph


class Token_to_Span_Composition(nn.Module):
    def __init__(self, bert_model_name, hidden_size, device=None):
        super(Token_to_Span_Composition, self).__init__()
        self.hidden_size = hidden_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Span Prefilter
        self.span_prefilter = SpanFilter(hidden_size, dropout=0.1).to(self.device)

    def compose(self, graph, lengths, keep_mask, sentences, ner_spans, actual_length, rel_emb):
        """
        토큰 레벨 그래프를 스팬 레벨로 확장합니다.
        """
        print(f"\n=== Span-level Processing ===")

        #* 1. Candidate Span 생성
        keep_mask_list = []
        current_pos = 0
        for i, length in enumerate(lengths):
            length = length.item()
            # 현재 문장의 실제 길이만큼만 잘라내기
            keep_mask_list.append(keep_mask[i, :length])  # [length]
            current_pos += length
        
        # 모든 문장 연결
        flat_keep_mask = torch.cat(keep_mask_list, dim=0)  # [total_length]

        candidate_spans = get_candidate_spans(
            sentences=sentences,  # 현재 문장들
            window_size=5,  # 최대 스팬 길이
            keep_mask=flat_keep_mask  # prefilter 결과 반영
        )
        
        #* 2. 정답 라벨 생성 (ner에 있으면 정답 1, 없으면 0)
        span_labels = generate_span_labels(
            candidate_spans=candidate_spans,
            ner_spans=ner_spans  # NER 정보
        )
        
        print(f"생성된 span 후보 수: {len(candidate_spans)}")
        print(f"Positive span 수: {sum(span_labels)}")
        
        #* 3. 스팬 임베딩 생성 (내부 토큰은 mean pooling으로 구현)
        # 토큰 노드만 추출 (node_type == 0)
        token_mask = (graph.node_type == 0)
        token_embeddings = graph.x[token_mask]  # [num_tokens, D]
        
        
        span_embs = get_span_embeddings(
            token_embeddings=token_embeddings,  # 토큰 노드의 임베딩만 사용
            candidate_spans=candidate_spans
        )  # [N_spans, D]
        # 여기에서 유효하지 않은 스팬들이 걸러짐. 
        
        if span_embs is None:
            print("스킵: 유효한 span이 없습니다.")
            return None, None, None, None, None
        
        # 텐서로 변환
        span_labels_tensor = torch.tensor(span_labels).unsqueeze(0).to(self.device)  # [1, N_spans]
        
        #* 4. Span Node Prefilter
        span_scores, span_prefilter_loss = self.span_prefilter(
            span_embs.unsqueeze(0),  # [1, N_spans, D]
            labels=span_labels_tensor
        )
        
        # threshold로 필터링
        candidate_spans, selected_indices = filter_candidate_spans(
            span_scores,
            candidate_spans,
            threshold=0.2  # 0.5 이상의 점수를 가진 스팬만 선택
        )
        
        # 필터링된 span 임베딩만 유지
        span_embs = span_embs[selected_indices]
        
        print(f"필터링 후 span 수: {len(candidate_spans)}")
        
        if len(candidate_spans) == 0:
            print("스킵: 필터링 후 유효한 span이 없습니다.")
            return None, None, None, None, None
        
        #* 5. Node Index 재정의
        num_tokens = token_embeddings.size(0)  # 실제 토큰 수
        num_relations = rel_emb.size(0)  # relation 노드 수
        
        token_ids = list(range(num_tokens))  # token node ids
        relation_ids = list(range(num_tokens, num_tokens + num_relations))  # relation node ids
        span_ids = list(range(num_tokens + num_relations, 
                            num_tokens + num_relations + len(candidate_spans)))  # span node ids
        
        #* 6. 스팬 노드의 relation 엣지 연결
        token_rel_edges = extract_token_relation_edges(graph)  # (token_id, rel_id, rel_node_id)
        
        print("token_rel_edges:", token_rel_edges)
        print("edge_index:", graph.edge_index)
        print(f"추출된 token-relation 엣지 수: {len(token_rel_edges)}")
        
        span_rel_edges = create_span_relation_edges(
            token_rel_edges=token_rel_edges,
            candidate_spans=candidate_spans,
            span_ids=span_ids
        )

        #* 7. 그래프 업데이트
        # 노드 임베딩 확장
        graph.x = torch.cat([graph.x, span_embs], dim=0)
        
        # 노드 타입 확장 (span 노드는 타입 2로 설정)
        new_node_types = torch.full((len(span_ids),), 2, 
                                  dtype=graph.node_type.dtype,
                                  device=graph.node_type.device)
        graph.node_type = torch.cat([graph.node_type, new_node_types])
        
        # 엣지 추가
        new_edges = []
        new_edge_types = []
        
        # 양방향 엣지 추가 (semantic type = 2)
        for span_id, rel_id, rel_node_id, edge_type in span_rel_edges:
            # span → relation
            new_edges.append([span_id, rel_node_id])
            new_edge_types.append(2)  # semantic type
            # relation → span
            new_edges.append([rel_node_id, span_id])
            new_edge_types.append(2)  # semantic type
            
            # print(f"Added edges for span {span_id} and relation {rel_node_id}")
        
        if new_edges:  # 새로운 엣지가 있는 경우에만 처리
            # 기존 edge_index와 같은 디바이스에 새로운 텐서 생성
            new_edges = torch.tensor(new_edges, 
                                   dtype=graph.edge_index.dtype,
                                   device=graph.edge_index.device).t()
            new_edge_types = torch.tensor(new_edge_types, 
                                        dtype=graph.edge_type.dtype,
                                        device=graph.edge_type.device)
            
            # 기존 엣지와 통합
            graph.edge_index = torch.cat([graph.edge_index, new_edges], dim=1)
            graph.edge_type = torch.cat([graph.edge_type, new_edge_types])
            
            print("\n=== 엣지 업데이트 ===")
            print(f"기존 엣지 수: {len(graph.edge_type) - len(new_edge_types)}")
            print(f"새로 추가된 엣지 수: {len(new_edge_types)}")
            print(f"전체 엣지 수: {len(graph.edge_type)}")
        
        print(f"\n=== 그래프 업데이트 결과 ===")
        print(f"전체 노드 수: {graph.x.size(0)}")
        print(f"Span 노드 수: {len(span_ids)}")
        print(f"새로 추가된 엣지 수: {len(span_rel_edges) * 2}")  # 양방향이므로 *2
        
        return graph, span_embs, span_ids, candidate_spans, span_prefilter_loss
    
class Span_level_Graph_Pooling(nn.Module):
    def __init__(self, bert_model_name, hidden_size, device=None):
        super(Span_level_Graph_Pooling, self).__init__()
        self.hidden_size = hidden_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.relation_types = ["USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", "COMPARE", "CONJUNCTION", "EVALUATE-FOR"]
        
        # Span Prefilter
        self.span_prefilter = SpanFilter(hidden_size, dropout=0.1).to(self.device)
        
        # Triple Scorer
        self.triple_scorer = BERTTripleScorer(hidden_size=hidden_size).to(self.device)
        self.cos = nn.CosineSimilarity(dim=-1).to(self.device)

        # LLM guidance
        self.tripleset_encoder = TripleSetEncoder(hidden_size=hidden_size).to(self.device)
        self.text_encoder = TextEncoder(model_name=bert_model_name).to(self.device)

        # Linear transformations for entity and relation updates
        self.entity_linear = nn.Linear(hidden_size, hidden_size).to(self.device)
        self.relation_linear = nn.Linear(hidden_size, hidden_size).to(self.device)
        
        # 기타 설정
        self.pooling_threshold = 0.1
        self.sample_size = 100


    def process_span_triples(self, graph, candidate_spans, span_ids, span_embs, rel_emb):
        """
        스팬 레벨 트리플을 처리합니다.
        Args:
            graph: 그래프 데이터
            candidate_spans: 스팬 후보 리스트 [(start, end), ...]
            span_ids: 스팬 노드 ID 리스트
            span_embs: 스팬 임베딩 [num_spans, D]
        """
        # 1. 토큰 노드 ID 추출
        token_mask = (graph.node_type == 0)
        token_ids = torch.where(token_mask)[0].tolist()
        # 2. 스팬-토큰 매핑 생성
        span_id_to_token_indices = {
            span_id: list(range(start, end + 1))
            for span_id, (start, end) in zip(span_ids, candidate_spans)
        }
        # 3. 토큰-관계 엣지 추출 (edge_type까지)
        token_rel_edges = extract_token_relation_edges(graph)  # (token_id, rel_id, rel_node_id, edge_type)
        # 4. 노드-관계 매핑 생성 (token + span 모두)
        from collections import defaultdict
        node_relation_candidates = defaultdict(lambda: defaultdict(set))  # {node_id: {edge_type: set(rel_id)}}
        # token 노드: 직접 연결된 edge_type별 relation
        for token_id, rel_id, rel_node_id, edge_type in token_rel_edges:
            node_relation_candidates[token_id][edge_type].add(rel_id)
        # span 노드: 내부 토큰의 relation 후보를 모아서 edge_type별로 합침
        for span_id, token_indices in span_id_to_token_indices.items():
            for edge_type in [0, 1]:
                rels = set()
                for token_id in token_indices:
                    rels |= node_relation_candidates[token_id][edge_type]
                if rels:
                    node_relation_candidates[span_id][edge_type] = rels
        #* 노드 타입에 따라 임베딩 분리
        # 5. 트리플 후보 생성 (이제 rel_type 포함)
        candidate_triples = extract_span_token_candidate_triples(
            token_ids=token_ids,
            span_ids=span_ids,
            node_relation_candidates=node_relation_candidates,
            span_id_to_token_indices=span_id_to_token_indices,
            relation_types=self.relation_types
        )
        if not candidate_triples:
            print("스킵: 유효한 triple이 없습니다.")
            return None, None, None, None, None, None, None, None, None
        #* 트리플 CLS 벡터 추출 추가
        # 6. Triple 점수 계산
        triple_cls_vectors = self.triple_scorer.encode_span_triples(
            triples=candidate_triples,
            span_embeddings=span_embs,
            token_embeddings=graph.x[token_mask],
            relation_embeddings=rel_emb
        )
        # CLS 벡터를 이용해서 스코어 계산
        span_level_scores = self.triple_scorer(
            cls_vectors=triple_cls_vectors  # 이미 얻은 CLS 벡터만 전달
        )
        # 7. 점수에 따라 트리플 분리
        top_triples, top_scores, bottom_triples, bottom_scores, top_indices, bottom_indices = split_triples_by_score(
            triples=candidate_triples,
            scores=span_level_scores,
            threshold=self.pooling_threshold,
            return_indices=True
        )
        print(f"\n트리플 분할 결과:")
        print(f"- 상위 트리플 수: {len(top_triples)}")
        print(f"- 하위 트리플 수: {len(bottom_triples)}")

        # 8. 하위 트리플의 엣지 삭제
        num_edges_before = graph.edge_index.size(1)
        graph = remove_edges_of_bottom_triples(
            graph,
            bottom_triples
        )
        num_edges_after = graph.edge_index.size(1)
        print(f"\n엣지 삭제 결과:")
        print(f"- 삭제 전 엣지 수: {num_edges_before}")
        print(f"- 삭제 후 엣지 수: {num_edges_after}")
        print(f"- 삭제된 엣지 수: {num_edges_before - num_edges_after}")

        return (candidate_triples, triple_cls_vectors, top_triples, top_scores, 
                bottom_triples, bottom_scores, top_indices, bottom_indices, 
                span_id_to_token_indices)

    def span_llm_alignment(self, 
                     input_text,
                     relation_types,
                     triple_cls_vectors,
                     top_triples,
                     top_scores, 
                     bottom_triples, 
                     bottom_scores, 
                     top_indices, 
                     bottom_indices,
                     span_id_to_token_indices,
                     num_tokens,
                     candidate_spans):
        """
        LLM 기반 선호도 학습을 수행합니다.
        
        Args:
            input_text: 입력 텍스트
            relation_types: 관계 타입 리스트
            triple_cls_vectors: 트리플 CLS 벡터
            top_triples: 상위 점수 트리플
            top_scores: 상위 트리플 점수
            bottom_triples: 하위 점수 트리플
            bottom_scores: 하위 트리플 점수
            top_indices: 상위 트리플 인덱스
            bottom_indices: 하위 트리플 인덱스
            span_id_to_token_indices: Dict[int, List[int]] - 스팬 ID → 토큰 인덱스 매핑
            num_tokens: int - 전체 토큰 수
            candidate_spans: List[Tuple[int, int]] - 스팬 후보 리스트
        """
        input_text_embs = []
        for sentences in input_text:
            flattened_text = " ".join(sent for sent in sentences)
            input_text_emb = self.text_encoder(flattened_text)  # [H]
            input_text_embs.append(input_text_emb)
        
        input_text_embs = torch.stack(input_text_embs, dim=0).to(self.device)  # [B, H]
        

        input_text_emb = input_text_embs.mean(dim=0)  # [H]
        
        # 샘플링
        if len(top_triples) == 0:
            print("Warning: No top triples found! Using only bottom triples.")
            sim_top = torch.tensor(-1.0).to(self.device)  # 기본값으로 -1 설정
            top_vector = torch.zeros(self.hidden_size).to(self.device)  # 제로 벡터
        else:
            top_triples, top_scores, sampled_top_idx = sample_triples(
                top_triples, top_scores, self.sample_size, return_indices=True)
            sampled_top_cls = triple_cls_vectors[[top_indices[i] for i in sampled_top_idx]]
            top_vector = self.tripleset_encoder(sampled_top_cls)
            sim_top = self.cos(input_text_emb.unsqueeze(0), top_vector.unsqueeze(0))

        if len(bottom_triples) == 0:
            print("Warning: No bottom triples found! Using only top triples.")
            sim_btm = torch.tensor(-1.0).to(self.device)  # 기본값으로 -1 설정
            bottom_vector = torch.zeros(self.hidden_size).to(self.device)  # 제로 벡터
        else:
            bottom_triples, bottom_scores, sampled_bottom_idx = sample_triples(
                bottom_triples, bottom_scores, self.sample_size, return_indices=True)
            sampled_bottom_cls = triple_cls_vectors[[bottom_indices[i] for i in sampled_bottom_idx]]  # [sample_size, H]
            bottom_vector = self.tripleset_encoder(sampled_bottom_cls)
            sim_btm = self.cos(input_text_emb.unsqueeze(0), bottom_vector.unsqueeze(0))

   
        print(f"코사인 유사도 (top/bottom): {sim_top.item():.4f}/{sim_btm.item():.4f}")
            
        # Triple verbalization
        # 노드 ID를 토큰으로 매핑 (entity relation)
        # node_id_to_token = node_id_to_token_map(
        #     entity_tokens=[token for sent in input_text for token in sent],  # 모든 문장의 토큰을 flatten,
        #     relation_types=relation_types
        # )
        tokens = [token for sent in input_text for token in sent]
        num_tokens = len(tokens)
        num_relations = len(relation_types)
        span_texts = [" ".join(tokens[start:end+1]) for (start, end) in candidate_spans]

        # entity + relation + span 모두 포함
        # 1. node_id_to_token을 직접 생성 (토큰, 리레이션, 스팬 모두 포함)
        node_id_to_token = {}
        for i in range(num_tokens):
            node_id_to_token[i] = tokens[i]
        for i in range(num_relations):
            node_id_to_token[num_tokens + i] = relation_types[i]
        for i, span_text in enumerate(span_texts):
            node_id_to_token[num_tokens + num_relations + i] = span_text

        print("node_id_to_token: ", node_id_to_token)

        # 트리플 인덱스를 텍스트로 변환
        top_triples_text = convert_span_triples_to_text(top_triples, node_id_to_token, span_id_to_token_indices, num_tokens)
        bottom_triples_text = convert_span_triples_to_text(bottom_triples, node_id_to_token, span_id_to_token_indices, num_tokens)
            
        summary_top = verbalize_triples(top_triples_text, level='span')
        summary_btm = verbalize_triples(bottom_triples_text, level='span')
        
        # LLM 선호도 평가
        preferred = ask_llm_preference(
            input_text=input_text,
            summary_a=summary_top,
            summary_b=summary_btm
        )

        print(f"LLM 선호도: {preferred}")
            
        # Preference learning loss 계산
        token_level_llm_loss = preference_learning_loss(sim_top, sim_btm, preferred, input_text_embs)
        # token_level_llm_losses.append(token_level_llm_loss)
            
        print(f"LLM Loss: {token_level_llm_loss.item():.4f}")
        print("="*50)

        return token_level_llm_loss
    
    def validity_aware_aggregation(self, graph, triples, triple_cls_vectors, top_indices):
        """
        스팬 레벨 Validity-Aware Aggregation.
        각 top triple에 대해 head entity node와 relation node를 동시에 업데이트.
        triple의 cls 벡터를 weighted sum해서 업데이트.

        Args:
            graph: 그래프 데이터
            triples: List[(head_id, rel_id, tail_id)] - 전체 트리플 리스트
            triple_cls_vectors: [num_triples, hidden_size] - 트리플별 CLS 벡터
            top_indices: List[int] - 상위 트리플의 인덱스 리스트
        """
        print(f"\n=== Span-Level Validity-Aware Aggregation ===")
        print(f"전체 트리플 수: {len(triples)}")
        print(f"상위 트리플 인덱스 수: {len(top_indices)}")
        print(f"최대 인덱스 값: {max(top_indices) if top_indices else -1}")

        device = graph.x.device
        # top triple의 cls 벡터만 추출
        top_cls_vectors = triple_cls_vectors[top_indices].to(device)  # [K, D]

        # 노드 타입에 따른 마스크 생성
        entity_mask = (graph.node_type == 0) | (graph.node_type == 2)  # 토큰 or span 노드
        relation_mask = (graph.node_type == 1)  # relation 노드

        # 업데이트할 노드 임베딩 초기화 (복사본 생성)
        updated_node_emb = graph.x.clone()

        # 1. Entity 노드 업데이트
        # head_ids: 각 top triple의 head entity ID를 모은 텐서
        head_ids = torch.tensor([triples[idx][0] for idx in top_indices if idx < len(triples)], 
                              device=device)
        
        # 각 head entity ID별로 등장 횟수를 계산
        unique_heads, head_counts = torch.unique(head_ids, return_counts=True)
        
        # head entity별 가중치 계산 (uniform weights)
        weights = torch.zeros(len(head_ids), device=device)
        for i, head_id in enumerate(head_ids):
            weights[i] = 1.0 / float(head_counts[unique_heads == head_id])
        
        # weighted cls vectors 계산
        weighted_cls = top_cls_vectors * weights.unsqueeze(1)  # [K, D]
        
        # scatter_add로 한번에 업데이트
        entity_update = torch.zeros_like(updated_node_emb)
        entity_update.scatter_add_(0, head_ids.unsqueeze(1).expand(-1, weighted_cls.size(1)), weighted_cls)
        
        # Linear transformation 적용
        entity_update = self.entity_linear(entity_update)
        
        # entity mask에 해당하는 노드만 업데이트
        updated_node_emb[entity_mask] = entity_update[entity_mask]

        # 2. Relation 노드 업데이트: 단순 linear layer transformation
        # relation 노드만 선형 변환
        updated_node_emb[relation_mask] = self.relation_linear(updated_node_emb[relation_mask])

        # 최종 업데이트된 임베딩을 그래프에 반영
        graph.x = updated_node_emb
        return graph

class Prediction(nn.Module):
    def __init__(self, hidden_size: int, device=None):
        super(Prediction, self).__init__()
        self.hidden_size = hidden_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # NER을 위한 임계값
        self.ner_threshold = 0.5
        
        # RE를 위한 임계값
        self.relation_threshold = 0.5
        
        # RE+를 위한 임계값
        self.replus_threshold = 0.6
    
    def predict_ner(self, token_embeddings: torch.Tensor, keep_mask: torch.Tensor) -> List[Tuple[int, int, float]]:
        """
        Token embeddings로부터 Named Entity를 예측합니다.
        
        Args:
            token_embeddings: [num_tokens, hidden_size] - 토큰 임베딩
            keep_mask: [num_tokens] - prefilter 결과 (1: keep, 0: drop)
            
        Returns:
            List[Tuple[int, int, float]] - (start_idx, end_idx, confidence) 형태의 예측된 엔티티 목록
        """
        predicted_entities = []
        current_entity = None
        
        # 연속된 keep_mask=1인 토큰들을 하나의 엔티티로 간주
        for i in range(len(keep_mask)):
            if keep_mask[i] == 1:
                if current_entity is None:
                    current_entity = (i, i, 1.0)  # (start, end, confidence)
                else:
                    current_entity = (current_entity[0], i, current_entity[2])
            elif current_entity is not None:
                if current_entity[2] >= self.ner_threshold:
                    predicted_entities.append(current_entity)
                current_entity = None
        
        # 마지막 엔티티 처리
        if current_entity is not None and current_entity[2] >= self.ner_threshold:
            predicted_entities.append(current_entity)
            
        return predicted_entities

    def predict_relations(
        self,
        triples: List[Tuple[int, int, int]],
        triple_scores: torch.Tensor,
        token_to_text: Dict[int, str],
        relation_types: List[str],
        span_id_to_token_indices: Dict[int, List[int]] = None,
        num_tokens: int = None
    ) -> List[Tuple[str, str, str]]:
        """
        Triple scores를 기반으로 관계를 예측합니다.
        
        Args:
            triples: List[Tuple[int, int, int]] - (head_id, rel_id, tail_id) 형태의 triple 목록
            triple_scores: [num_triples] - 각 triple의 점수
            token_to_text: Dict[int, str] - token ID를 텍스트로 매핑하는 딕셔너리
            relation_types: List[str] - relation type 목록
            span_id_to_token_indices: Dict[int, List[int]] - span ID를 구성하는 token indices 매핑
            num_tokens: int - 실제 토큰의 개수 (스팬 노드 제외)
            
        Returns:
            List[Tuple[str, str, str]] - (head_text, relation_type, tail_text) 형태의 예측된 관계 목록
        """
        predicted_relations = []
        
        # 스팬 정보가 있는 경우 convert_span_triples_to_text 사용
        if span_id_to_token_indices is not None and num_tokens is not None:
            node_id_to_token = {
                "entity": token_to_text,
                "relation": {i: rel for i, rel in enumerate(relation_types)}
            }
            
            for triple, score in zip(triples, triple_scores):
                score_value = score.item() if isinstance(score, torch.Tensor) else score
                if score_value >= self.relation_threshold:
                    head_id, rel_id, tail_id = triple
                    # convert_span_triples_to_text는 (head, rel, tail, rel_type) 형식을 기대
                    converted = convert_span_triples_to_text(
                        [(head_id, rel_id, tail_id, "semantic")],
                        node_id_to_token,
                        span_id_to_token_indices,
                        num_tokens
                    )
                    if converted:  # 변환된 결과가 있는 경우
                        head_text, rel_text, tail_text, _ = converted[0]
                        predicted_relations.append((head_text, rel_text, tail_text))
        
        # 스팬 정보가 없는 경우 기존 방식대로 처리
        else:
            for triple, score in zip(triples, triple_scores):
                score_value = score.item() if isinstance(score, torch.Tensor) else score
                if score_value >= self.relation_threshold:
                    head_id, rel_id, tail_id = triple
                    # 토큰 ID가 token_to_text의 범위 내에 있는 경우만 처리
                    if head_id in token_to_text and tail_id in token_to_text:
                        head_text = token_to_text[head_id]
                        tail_text = token_to_text[tail_id]
                        relation = relation_types[rel_id]
                        predicted_relations.append((head_text, relation, tail_text))
        
        return predicted_relations

    def predict_relations_plus(
        self,
        token_triples: List[Tuple[int, int, int]],
        span_triples: List[Tuple[int, int, int]],
        token_scores: torch.Tensor,
        span_scores: torch.Tensor,
        token_to_text: Dict[int, str],
        span_to_text: Dict[int, str],
        relation_types: List[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Token-level과 Span-level의 triple scores를 통합하여 관계를 예측합니다.
        
        Args:
            token_triples: List[Tuple[int, int, int]] - token-level triples
            span_triples: List[Tuple[int, int, int]] - span-level triples
            token_scores: [num_token_triples] - token-level triple scores
            span_scores: [num_span_triples] - span-level triple scores
            token_to_text: Dict[int, str] - token ID를 텍스트로 매핑
            span_to_text: Dict[int, str] - span ID를 텍스트로 매핑
            relation_types: List[str] - relation type 목록
            
        Returns:
            List[Tuple[str, str, str]] - (entity_text, relation_type, entity_text) 형태의 예측된 관계 목록
        """
        predicted_relations = []
        
        # Token-level relations
        for triple, score in zip(token_triples, token_scores):
            score_value = score.item() if isinstance(score, torch.Tensor) else score
            if score_value >= self.replus_threshold:
                head_id, rel_id, tail_id = triple
                head_text = token_to_text[head_id]
                tail_text = token_to_text[tail_id]
                relation = relation_types[rel_id]
                predicted_relations.append((head_text, relation, tail_text))
        
        # Span-level relations
        for triple, score in zip(span_triples, span_scores):
            score_value = score.item() if isinstance(score, torch.Tensor) else score
            if score_value >= self.replus_threshold:
                head_id, rel_id, tail_id = triple
                # head/tail이 span인 경우 span_to_text 사용
                head_text = span_to_text.get(head_id, token_to_text.get(head_id))
                tail_text = span_to_text.get(tail_id, token_to_text.get(tail_id))
                relation = relation_types[rel_id]
                predicted_relations.append((head_text, relation, tail_text))
        
        # 중복 제거
        predicted_relations = list(set(predicted_relations))
        
        return predicted_relations

# class RePoolPredictor:
#     """
#     샘플/배치 단위로 예측과 loss를 모두 반환하는 통합 predictor.
#     model.py에서 이 클래스를 사용해 예측/로스 결과를 받아 최종 loss를 합산할 수 있음.
#     """
#     def __init__(self, relation_types, device=None):
#         self.relation_types = relation_types
#         self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
#         self.re_predictor = REPredictor(threshold=0.1)
#         self.relation_loss_fn = RelationLoss(relation_types=relation_types, device=self.device)

#     def predict_and_loss(self, sample, graph,
#                          span_top_triples, span_top_scores, 
#                          token_to_text, num_tokens,
#                          gold_relations, sentences,
#                          prefilter_loss, span_prefilter_loss,
#                          token_level_llm_loss, span_level_llm_loss):
#         """
#         예측 결과와 모든 loss를 반환
#         """
#         # 예측
#         relation_indices = [i - num_tokens for i, t in enumerate(graph.node_type) if t == 1]
#         predictions = self.re_predictor.predict(
#             num_tokens=num_tokens,
#             span_triples=span_top_triples if span_top_triples is not None else [],
#             span_scores=span_top_scores if span_top_scores is not None else torch.tensor([]),
#             token_to_text=token_to_text,
#             relation_indices=relation_indices
#         )
#         # gold relation flatten
#         flat_gold_relations = flatten_gold_relations(gold_relations, sentences)
#         # relation loss
#         relation_loss_val = self.relation_loss_fn(
#             predictions,
#             flat_gold_relations
#         )
#         # 최종 loss 합산
#         total_loss = (
#             prefilter_loss +
#             (span_prefilter_loss if span_prefilter_loss is not None else 0.0) +
#             token_level_llm_loss +
#             span_level_llm_loss +
#             relation_loss_val
#         )
#         return {
#             "predictions": predictions,
#             "prefilter_loss": prefilter_loss,
#             "span_prefilter_loss": span_prefilter_loss,
#             "token_level_llm_loss": token_level_llm_loss,
#             "span_level_llm_loss": span_level_llm_loss,
#             "relation_loss": relation_loss_val,
#             "final_loss": total_loss
#         }

#* ================================
if __name__ == "__main__":
    # 1. 데이터 로드
    with open("data/train_sample_scierc.json") as f:
        data = [json.loads(line) for line in f]
    sample = data[0]  # 첫 번째 샘플만 사용

    # 2. 모델 초기화
    dual = Dual_View_BipartiteKG_Construction(bert_model_name="bert-base-uncased", hidden_size=768)
    token_level = Token_Level_Graph_Pooling(bert_model_name="bert-base-uncased", hidden_size=768)
    token_to_span = Token_to_Span_Composition(bert_model_name="bert-base-uncased", hidden_size=768)
    span_level = Span_level_Graph_Pooling(bert_model_name="bert-base-uncased", hidden_size=768)
    relation_types = [
            "USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", 
            "COMPARE", "CONJUNCTION", "EVALUATE-FOR"
        ]

    # 3. Entity Embeddings 생성
    # 문장별 토큰 길이 계산
    lengths = torch.tensor([len(sent) for sent in sample["sentences"]]).to(dual.device)
    token_out = dual.get_entity_embeddings(tokens=sample["sentences"], lengths=lengths)
    entity_embs = token_out["embeddings"]
    entity_mask = token_out["mask"]
    print(f"\n=== Entity Embeddings ===")
    print(f"Entity embeddings shape: {entity_embs.shape}")
    print(f"Entity mask shape: {entity_mask.shape}")
    print(f"Lengths: {lengths}")

    # 4. Relation Embeddings 생성
    # relation id는 relation_types의 인덱스
    relation_ids = torch.arange(len(dual.relation_types)).to(dual.device)
    rel_emb = dual.get_relation_embeddings(relation_ids)
    print(f"\n=== Relation Embeddings ===")
    print(f"Relation embeddings shape: {rel_emb.shape}")

    # 5. Entity Prefilter 적용
    # 라벨 생성 (NER spans 기반)
    prefilter_out = dual.get_entity_prefilter(entity_embs, entity_mask, sample["sentences"], sample["ner"])
    keep_mask = prefilter_out["keep_mask"]
    prefilter_loss = prefilter_out["prefilter_loss"]

    print(f"\n=== Prefilter Results ===")
    print(f"Keep mask shape: {keep_mask.shape}")
    print(f"Prefilter loss: {prefilter_loss.item():.4f}")

    # 6. 이분 그래프 구성
    graph = dual.construct_graph(
        entity_embs=entity_embs,
        rel_emb=rel_emb,
        keep_mask=keep_mask,  
        actual_entity_length=lengths
    )
    print(f"\n=== Final Graph ===")
    print(f"Number of nodes: {graph.x.size(0)}")
    print(f"Number of edges: {graph.edge_index.size(1)}")
    if hasattr(graph, 'node_type'):
        print(f"Node types: {graph.node_type.unique().tolist()}")
    if hasattr(graph, 'edge_type'):
        print(f"Edge types: {graph.edge_type.unique().tolist()}")

    # 7. Token Level Graph Pooling
    graph, triples, triple_cls_vectors, top_triples, top_scores, bottom_triples, bottom_scores, top_indices, bottom_indices = token_level.process_token_triples(graph)

    # top_triples가 없으면 스킵
    if top_triples is None or len(top_triples) == 0:
        print("Warning: No top token-level triples found! Skipping this sample.")
        token_level_llm_loss = torch.tensor(0.0, requires_grad=True)
    else:
        input_text = sample["sentences"]
        token_level_llm_loss = token_level.token_llm_alignment(
            input_text,
            relation_types,
            triple_cls_vectors,
            top_triples,
            top_scores, 
            bottom_triples, 
            bottom_scores, 
            top_indices, 
            bottom_indices)
        # 8. Token Level Validity-Aware Aggregation
        graph = token_level.validity_aware_aggregation(graph, triples, triple_cls_vectors, top_indices)

    # 9. Token to Span Composition
    actual_length = lengths[0].item()  # 첫 번째 문장의 실제 길이
    graph, span_embs, span_ids, candidate_spans, span_prefilter_loss = token_to_span.compose(
        graph=graph,
        lengths=lengths,
        keep_mask=keep_mask, 
        sentences=sample["sentences"],
        ner_spans=sample["ner"],
        actual_length=actual_length,
        rel_emb=rel_emb
    )

    # 10. Span Level Graph Pooling
    if span_ids is not None and candidate_spans is not None and len(candidate_spans) > 0:
        print(f"\n=== Span Level Graph Pooling ===")
        result = span_level.process_span_triples(graph, candidate_spans, span_ids, span_embs, rel_emb)
        if result is None:
            print("Warning: No valid span-level triples found! Skipping this sample.")
            span_level_llm_loss = torch.tensor(0.0, requires_grad=True)
        else:
            candidate_triples, triple_cls_vectors, span_top_triples, span_top_scores, span_bottom_triples, span_bottom_scores, span_top_indices, span_bottom_indices, span_id_to_token_indices = result
            # span_top_triples가 없으면 스킵
            if span_top_triples is None or len(span_top_triples) == 0:
                print("Warning: No top span-level triples found! Skipping this sample.")
                span_level_llm_loss = torch.tensor(0.0, requires_grad=True)
            else:
                # Span Level LLM Alignment
                span_level_llm_loss = span_level.span_llm_alignment(
                    input_text=input_text,
                    relation_types=relation_types,
                    triple_cls_vectors=triple_cls_vectors,
                    top_triples=span_top_triples,
                    top_scores=span_top_scores,
                    bottom_triples=span_bottom_triples,
                    bottom_scores=span_bottom_scores,
                    top_indices=span_top_indices,
                    bottom_indices=span_bottom_indices,
                    span_id_to_token_indices=span_id_to_token_indices,
                    num_tokens=len(span_id_to_token_indices),
                    candidate_spans=candidate_spans
                )
                # span level validity aware aggregation
                graph = span_level.validity_aware_aggregation(graph, candidate_triples, triple_cls_vectors, span_top_indices)

            # === Final Results & Relation Extraction ===
            print(f"\n=== Final Results ===")
            print(f"Token Level LLM Loss: {token_level_llm_loss.item():.4f}")
            print(f"Span Level LLM Loss: {span_level_llm_loss.item():.4f}")

            print(f"\n=== Relation Extraction Prediction ===")
            # 토큰/스팬 텍스트 매핑 생성
            tokens = [token for sent in sample["sentences"] for token in sent]
            token_to_text = {i: token for i, token in enumerate(tokens)}

            num_tokens = len(tokens)
            for rel_idx, rel_type in enumerate(relation_types):
                token_to_text[num_tokens + rel_idx] = rel_type
            num_relations = len(relation_types)
            for span_idx, (start, end) in enumerate(candidate_spans):
                token_to_text[num_tokens + num_relations + span_idx] = " ".join(tokens[start:end+1])
            re_predictor = REPredictor(threshold=0.7)
            relation_indices = [i - num_tokens for i, t in enumerate(graph.node_type) if t == 1]

            predictions = re_predictor.predict(
                num_tokens=num_tokens,
                span_triples=span_top_triples,
                span_scores=span_top_scores,
                token_to_text=token_to_text,
                relation_indices=relation_indices
            )
            print("\nExtracted Relations:")
            for head_text, rel_type, tail_text, confidence in predictions:
                print(f"- {head_text} --[{rel_type}]--> {tail_text} (confidence: {confidence:.4f})")
            # Relation Loss 계산
            score_tensors = []
            if top_scores is not None and isinstance(top_scores, torch.Tensor) and top_scores.numel() > 0:
                score_tensors.append(top_scores)
            if span_top_scores is not None and isinstance(span_top_scores, torch.Tensor) and span_top_scores.numel() > 0:
                score_tensors.append(span_top_scores)

            if len(score_tensors) > 0:
                predicted_scores = torch.cat(score_tensors)
            else:
                predicted_scores = torch.tensor([], dtype=torch.float32)

            relation_loss = RelationLoss(relation_types=relation_types)
            # gold_relations flatten
            relations = sample.get("relations", [])
            print("relations: ", relations)
            flat_gold_relations = flatten_gold_relations(sample.get("relations", []), sample["sentences"])
            print("[DEBUG-prediction] flat_gold_relations:", flat_gold_relations)
            
            relation_loss_val = relation_loss(
                predictions,
                flat_gold_relations
            )
            print("predictions: ", predictions)
            print(f"\nLosses:")
            print(f"Token Level LLM Loss: {token_level_llm_loss.item():.4f}")
            print(f"Span Level LLM Loss: {span_level_llm_loss.item():.4f}")
            print(f"Relation Prediction Loss: {relation_loss_val.item():.4f}")
            print(f"Prefilter Loss: {prefilter_loss.item():.4f}")
            print(f"Span Prefilter Loss: {span_prefilter_loss.item():.4f}")
            total_loss = (
                            prefilter_loss +
                            (span_prefilter_loss if span_prefilter_loss is not None else 0.0) +
                            token_level_llm_loss +
                            span_level_llm_loss +
                            relation_loss_val
                        )
            print(f"Total Loss: {total_loss.item():.4f}")
    else:
        print("Warning: No valid candidate spans found! Skipping span-level processing.")
        span_level_llm_loss = torch.tensor(0.0, requires_grad=True)
        print("\nSkipping Span Level Processing: No valid spans found")