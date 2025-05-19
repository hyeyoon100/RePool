import torch
import torch.nn as nn
import json 
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple
from collections import defaultdict

from data_processor import build_batch
from token_rep import EntityTokenRep, RelationTokenRep   # 토큰/엔티티 임베딩
from generate_labels import generate_token_labels, generate_span_labels # keep label 생성
from prefilter import TokenFilter, SpanFilter  # 엔티티/스팬 필터
from utils import build_bipartite_graph, get_candidate_spans, sample_triples, TextEncoder, create_span_relation_edges, get_span_embeddings, extract_token_relation_edges, filter_candidate_spans
from triplet_scorer import extract_candidate_triples, BERTTripleScorer, split_triples_by_score, remove_edges_of_bottom_triples, node_id_to_token_map, convert_id_triples_to_text, extract_span_token_candidate_triples, convert_span_triples_to_text
from llm_guidance import TripleSetEncoder, verbalize_triples, preference_learning_loss, ask_llm_preference   # LLM-guided alignment/loss
from aggregation import EntityEmbeddingUpdater, RelationEmbeddingUpdater  # aggregation
from prediction import predict_relations
from loss_functions import RelationLoss





class IEModel(nn.Module):
    def __init__(
        self,
        bert_model_name="bert-base-cased",
        hidden_size=768,
        sample_size=100,  # 트리플 샘플링 크기
        device=None,
        relation_types=None  # relation_types 파라미터 추가
    ):
        super().__init__()
        # Device 설정
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # relation types 설정
        self.default_relation_types = [
            "USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", 
            "COMPARE", "CONJUNCTION", "EVALUATE-FOR"
        ]
        self.relation_types = relation_types or self.default_relation_types
        
        # hidden size 저장
        self.hidden_size = hidden_size
        
        # Token representation
        self.entity_token_rep = EntityTokenRep(model_name=bert_model_name, hidden_size=hidden_size).to(self.device)
        self.relation_token_rep = RelationTokenRep(
            relation_types=self.relation_types,  # self.relation_types 사용
            lm_model_name=bert_model_name, 
            freeze_bert=True 
        ).to(self.device)
        
        # Prefilter
        self.entity_prefilter = TokenFilter(hidden_size, dropout=0.1).to(self.device)
        self.span_prefilter = SpanFilter(hidden_size, dropout=0.1).to(self.device)
        
        # Triple scorer
        self.triple_scorer = BERTTripleScorer(hidden_size=768).to(self.device)
        
        # LLM guidance
        self.tripleset_encoder = TripleSetEncoder(hidden_size=hidden_size).to(self.device)
        self.text_encoder = TextEncoder(model_name=bert_model_name).to(self.device)
        
        self.pooling_threshold = 0.3
        self.sample_size = sample_size
        self.cos = nn.CosineSimilarity(dim=-1).to(self.device)

        # Validity-Aware Aggregation
        self.update_entity_embedding = EntityEmbeddingUpdater(hidden_dim=hidden_size).to(self.device)
        self.update_relation_embedding = RelationEmbeddingUpdater(hidden_dim=hidden_size).to(self.device)

        # RE를 위한 임계값 설정
        self.relation_threshold = 0.5
        self.relation_loss = RelationLoss(relation_types=self.relation_types, device=self.device)  # self.relation_types 사용

        print(f"Model initialized on device: {self.device}")
        print(f"Using relation types: {self.relation_types}")

    def to(self, device):
        """
        모델의 device를 변경합니다.
        """
        self.device = device
        return super().to(device)


    def forward(self, batch):
        """
        batch: dict with
            - tokens: List[List[str]]
            - lengths: torch.LongTensor
            - sentences: List[List[str]]  # 문장 단위로 구분된 원본 텍스트
            - ...
        """
        # # 배치별 loss들을 저장할 리스트들 초기화
        # relation_losses = []
        # entity_prefilter_losses = []
        # token_level_llm_losses = []
        # span_prefilter_losses = []
        # span_level_llm_losses = []

        # 입력 텐서들을 device로 이동
        if isinstance(batch["lengths"], list):
            batch["lengths"] = torch.tensor(batch["lengths"], dtype=torch.long)
        batch["lengths"] = batch["lengths"].to(self.device)
        if "relation_ids" in batch:
            batch["relation_ids"] = [ids.to(self.device) if isinstance(ids, torch.Tensor) else ids 
                                   for ids in batch["relation_ids"]]

        # === Input Text Embedding 생성 ===
        input_text_embs = []
        for sentences in batch["sentences"]:
            flattened_text = " ".join(" ".join(sent) for sent in sentences)
            input_text_emb = self.text_encoder(flattened_text)  # [H]
            input_text_embs.append(input_text_emb)
        
        input_text_embs = torch.stack(input_text_embs, dim=0).to(self.device)  # [B, H]
        
        print(f"\n=== Input Text Embedding 생성 ===")
        print(f"배치 크기: {input_text_embs.size(0)}")
        print(f"임베딩 차원: {input_text_embs.size(1)}")

        #* 1. Token-level Entity 임베딩
        # 토큰 들어갈 때 sentences. 쌩으로 들어감 
        token_out = self.entity_token_rep(batch["tokens"], batch["lengths"])
        entity_embs = token_out["embeddings"]        # [B, L, D] <- padding 처리 O
        entity_mask = token_out["mask"]              # [B, L]

        #* 1-2. Relation Embedding 얻기
        # ____ (1) variable-length list padding
        PAD_ID = -1
        rel_ids = batch["relation_ids"]
        max_len = max(len(x) for x in rel_ids)
        padded_rel_ids = [x + [PAD_ID]*(max_len-len(x)) for x in rel_ids]
        relation_ids_tensor = torch.LongTensor(padded_rel_ids)  # [B, max_len]

        # ____ (2) relation 마스크 생성 _ relation 개수 가장 많은 샘플 기준에 맞추어 패딩
        relation_mask = torch.BoolTensor([[i < len(row) for i in range(max_len)] for row in rel_ids])  # [B, max_len]

        # ____ (3) 임베딩 얻기 (PAD_ID는 무시)
        rel_emb = self.relation_token_rep(relation_ids_tensor)  # [B, max_len, D]

        #* 2. Entity Prefilter
        # ___ entity label 생성
        e_labels_list = []
        for sent_list, ner_spans, tokens in zip(batch["sentences"], batch["ner"], batch["tokens"]):
            # sent_list: [num_sent][tokens]
            # tokens: flatten
            label = [0] * len(tokens)
            cur = 0
            for sent_idx, sent in enumerate(sent_list):
                for start, end, _ in ner_spans[sent_idx]:
                    for i in range(start, end+1):
                        # local index to global
                        idx = cur + (i - start)
                        if 0 <= idx < len(tokens):
                            label[idx] = 1
                cur += len(sent)
            e_labels_list.append(label)
        e_labels_tensor = pad_sequence(
                                    [torch.tensor(l, dtype=torch.long) for l in e_labels_list],
                                    batch_first=True,
                                    padding_value=0
                                ).to(entity_embs.device)  # [B, max_len]

        # ___ 필터링 
        scores, entity_prefilter_loss = self.entity_prefilter(entity_embs, entity_mask, labels=e_labels_tensor)    # scores: [B, L], prefilter_loss: scalar (or [B] shape)
        entity_filter_threshold = 0.5
        keep_mask = (scores > entity_filter_threshold).float()   # [B, L], 1 = keep, 0 = drop

        #* 3. Bipartite Graph (엔티티/관계)
        graphs = []
        for b in range(entity_embs.size(0)):  # B
            entity_emb = entity_embs[b]             # [L, D]
            relation_emb_b = rel_emb[b][:len(self.relation_types)]  # 실제 relation만 사용
            
            # 실제 토큰 길이만큼만 사용
            actual_length = batch["lengths"][b].item()
            entity_emb = entity_emb[:actual_length]  # [actual_length, D]
            keep_mask_b = keep_mask[b]              # [L]
            
            all_node_emb = torch.cat([entity_emb, relation_emb_b], dim=0)  # [actual_length+R, D]
            num_relation = len(self.relation_types)  # 항상 고정된 relation type 개수 사용
            relation_emb_b = relation_emb_b[:num_relation]  # 실제 relation만 사용
            keep_ids = (keep_mask_b == 1).nonzero(as_tuple=True)[0].tolist()
            flat_keep_mask = keep_mask_b.tolist()   # [actual_length]
            
            print(f"\n=== Triple Scorer 입력 디버깅 ===")
            print(f"all_node_emb size: {all_node_emb.size()}")
            print(f"actual_length: {actual_length}")
            print(f"num_relation: {num_relation}")

            # 그래프 생성
            graph = build_bipartite_graph(
                all_node_emb=all_node_emb,
                keep_ids=keep_ids,
                num_relation=num_relation,
                bidirectional=True,
                use_node_type=True,
                keep_mask=flat_keep_mask
            )
            graphs.append(graph)
        
            # Triple 후보 추출
            triples = extract_candidate_triples(graph, compound_edge_type=0)
            print(f"추출된 트리플 수: {len(triples)}")

            token_level_scores = self.triple_scorer(
                triples=triples,
                token_embeddings=all_node_emb,
                relation_embeddings=rel_emb[b][:num_relation],
                lengths=actual_length  # lengths 대신 실제 엔티티 토큰 개수 전달
            )
            
            triple_cls_vectors = self.triple_scorer.encode_triples(
                triples, 
                all_node_emb, 
                relation_embeddings=rel_emb[b][:num_relation],  # 여기도 동일하게 수정
                actual_entity_length=actual_length # 실제 토큰으 ㅣ길이
            )
            
            if len(token_level_scores) > 0:
                print(f"\n점수 통계:")
                print(f"- 최소값: {token_level_scores.min().item():.4f}")
                print(f"- 최대값: {token_level_scores.max().item():.4f}")
                print(f"- 평균값: {token_level_scores.mean().item():.4f}")
                print(f"- 중앙값: {token_level_scores.median().item():.4f}")
            
            #* 6. score로 top/bottom split & 엣지 삭제
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
            
            graph = remove_edges_of_bottom_triples(
                graph,
                bottom_triples,
                semantic_type_id=2,
                compound_type_ids=(0, 1)
            )
            
            # 엣지 삭제 후 상태 확인
            num_edges_after = graph.edge_index.size(1)
            print(f"\n엣지 삭제 결과:")
            print(f"- 삭제 전 엣지 수: {num_edges_before}")
            print(f"- 삭제 후 엣지 수: {num_edges_after}")
            print(f"- 삭제된 엣지 수: {num_edges_before - num_edges_after}")
            
            #* 7. LLM-guided Alignment
            print(f"\n=== LLM Alignment ===")
            
            # 샘플링
            top_triples, top_scores, sampled_top_idx = sample_triples(
                top_triples, top_scores, self.sample_size, return_indices=True)
            bottom_triples, bottom_scores, sampled_bottom_idx = sample_triples(
                bottom_triples, bottom_scores, self.sample_size, return_indices=True)
            
            print(f"샘플링된 트리플 수: {len(top_triples)}/{len(bottom_triples)}")
            
            # CLS 벡터 샘플링
            sampled_top_cls = triple_cls_vectors[[top_indices[i] for i in sampled_top_idx]]    # [sample_size, H]
            sampled_bottom_cls = triple_cls_vectors[[bottom_indices[i] for i in sampled_bottom_idx]]  # [sample_size, H]
            
            # Triple set 인코딩
            top_vector = self.tripleset_encoder(sampled_top_cls)      # [H]
            bottom_vector = self.tripleset_encoder(sampled_bottom_cls)  # [H]
            
            # 입력 텍스트와의 코사인 유사도
            input_text_emb = input_text_embs[b].to(self.device)  # [H]
            sim_top = self.cos(input_text_emb.unsqueeze(0), top_vector.unsqueeze(0))
            sim_btm = self.cos(input_text_emb.unsqueeze(0), bottom_vector.unsqueeze(0))
            
            print(f"코사인 유사도 (top/bottom): {sim_top.item():.4f}/{sim_btm.item():.4f}")
            
            # Triple verbalization
            # 노드 ID를 토큰으로 매핑
            node_id_to_token = node_id_to_token_map(
                entity_tokens=batch["tokens"][b],
                relation_types=self.relation_types
            )
            
            # 트리플 인덱스를 텍스트로 변환
            top_triples_text = convert_id_triples_to_text(top_triples, node_id_to_token)
            bottom_triples_text = convert_id_triples_to_text(bottom_triples, node_id_to_token)
            
            summary_top = verbalize_triples(top_triples_text)
            summary_btm = verbalize_triples(bottom_triples_text)
            
            # LLM 선호도 평가
            preferred = ask_llm_preference(
                input_text=batch["sentences"][b],
                summary_a=summary_top,
                summary_b=summary_btm
            )
            
            print(f"LLM 선호도: {preferred}")
            
            # Preference learning loss 계산
            token_level_llm_loss = preference_learning_loss(sim_top, sim_btm, preferred)
            # token_level_llm_losses.append(token_level_llm_loss)
            
            print(f"배치 {b} LLM Loss: {token_level_llm_loss.item():.4f}")
            print("="*50)
            
            #! Validity-Aware Aggregation
            print(f"\n=== Validity-Aware Aggregation ===")
            
            # 1. top triple들의 head id 추출 및 중복 제거
            top_head_ids = [triples[idx][0] for idx in top_indices]
            unique_head_ids = set(top_head_ids)
            print(f"unique_head_ids: {unique_head_ids}")
            
            # 2. 엔티티 임베딩 업데이트
            updated_entity_emb = entity_emb[b, :actual_length].clone()  # [actual_length, D]
            
            # device 확인 및 이동
            device = graph.x.device
            updated_entity_emb = updated_entity_emb.to(device)
            
            # 차원 확인 및 수정
            if updated_entity_emb.dim() == 1:
                # [actual_length] -> [actual_length, hidden_size]
                updated_entity_emb = updated_entity_emb.unsqueeze(-1).expand(-1, self.hidden_size).clone()
                # expand 연산은 메모리를 공유하기 때문에 새로운 메모리 할당 필요 (clone)
            elif updated_entity_emb.size(1) != self.hidden_size:
                # [actual_length, 1] -> [actual_length, hidden_size]
                updated_entity_emb = updated_entity_emb.expand(-1, self.hidden_size).clone()
            
            print(f"Initial updated_entity_emb shape: {updated_entity_emb.shape}")
            print(f"Device 확인:")
            print(f"- graph.x: {graph.x.device}")
            print(f"- updated_entity_emb: {updated_entity_emb.device}")
            
            # top triple의 CLS 벡터 추출 및 device 이동
            top_cls_vectors = triple_cls_vectors[top_indices].to(device)  # [K, D]
            
            # 3. 각 head_id에 대해 업데이트 적용
            for head_id in unique_head_ids:
                updated_emb = self.update_entity_embedding(
                    entity_emb=updated_entity_emb.clone(),  # 복사본 전달
                    cls_embeddings=top_cls_vectors,
                    triple_indices=top_indices,
                    triples=triples,
                    target_head_id=head_id
                )  # [D]
                
                
                # 해당 위치의 임베딩 업데이트 (contiguous 메모리로 변환)
                updated_entity_emb[head_id] = updated_emb.clone().contiguous()
            
            print(f"Final updated_entity_emb shape: {updated_entity_emb.shape}")
            
            # 엔티티 임베딩 업데이트 반영
            num_entity = actual_length
            graph.x[:num_entity] = updated_entity_emb
            
            # relation 임베딩 업데이트
            updated_relation_embedding = self.update_relation_embedding(
                node_emb=graph.x,
                node_type=graph.node_type
            )
            
            # 최종 업데이트된 임베딩을 그래프에 반영
            graph.x = updated_relation_embedding
            
            print(f"엔티티 노드 {len(unique_head_ids)}개 업데이트 완료")
            print(f"전체 노드 임베딩 크기: {updated_relation_embedding.size()}")
            print("="*50)

            #! Span-level Processing
            print(f"\n=== Span-level Processing ===")
            
            #* 1. Candidate Span 생성
            keep_mask_list = keep_mask[b, :actual_length].cpu().tolist()  # token-level prefilter 결과
            candidate_spans = get_candidate_spans(
                sentences=batch["sentences"][b],  # 현재 배치의 문장들
                window_size=5,  # 최대 스팬 길이
                keep_mask=keep_mask_list  # prefilter 결과 반영
            )
            
            #* 2. 정답 라벨 생성 (ner에 있으면 정답 1, 없으면 0)
            span_labels = generate_span_labels(
                candidate_spans=candidate_spans,
                ner_spans=batch["ner"][b]  # 현재 배치의 NER 정보
            )
            
            print(f"생성된 span 후보 수: {len(candidate_spans)}")
            print(f"Positive span 수: {sum(span_labels)}")
            
            #* 3. 스팬 임베딩 생성 (내부 토큰은 mean pooling으로 구현)
            span_embs = get_span_embeddings(
                token_embeddings=updated_entity_emb,  # 업데이트된 토큰 임베딩 사용
                candidate_spans=candidate_spans
            )  # [N_spans, D]
            if span_embs is None:
                print("스킵: 유효한 span이 없습니다.")
                continue  # 현재 배치 아이템을 건너뛰고 다음으로 진행
            
            # 텐서로 변환
            span_labels_tensor = torch.tensor(span_labels).unsqueeze(0).to(self.device)  # [1, N_spans]
            
            #* 4. Span Node Prefilter
            span_scores, span_prefilter_loss = self.span_prefilter(
                span_embs.unsqueeze(0),  # [1, N_spans, D]
                labels=span_labels_tensor
            )
            
            # topk로 필터링
            candidate_spans, top_indices = filter_candidate_spans(
                span_scores,
                candidate_spans,
                topk_ratio=0.4  #! 상위 40%만 유지 _ threshold로 변경
            )
            
            # 필터링된 span 임베딩만 유지
            span_embs = span_embs[top_indices]
            
            print(f"필터링 후 span 수: {len(candidate_spans)}")
            print(f"Span 임베딩 크기: {span_embs.shape}")
            
            #* 5. Node Index 재정의 (bipartite 구조: token -> relation -> span 순서)
            token_ids = list(range(actual_length))  # token node ids
            relation_ids = list(range(actual_length, actual_length + rel_emb[b].size(0)))  # relation node ids
            span_ids = list(range(actual_length + rel_emb[b].size(0), 
                                actual_length + rel_emb[b].size(0) + len(candidate_spans)))  # span node ids
            
            #* 6. 스팬 노드의 relation 엣지 연결
            token_rel_edges = extract_token_relation_edges(graph)  # (token_id, rel_id, rel_node_id)
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
            for span_id, rel_id, rel_node_id in span_rel_edges:
                # span → relation
                new_edges.append([span_id, rel_node_id])
                new_edge_types.append(2)  # semantic type
                # relation → span
                new_edges.append([rel_node_id, span_id])
                new_edge_types.append(2)  # semantic type
            
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
            
            print(f"\n=== 그래프 업데이트 결과 ===")
            print(f"전체 노드 수: {graph.x.size(0)}")
            print(f"Span 노드 수: {len(span_ids)}")
            print(f"새로 추가된 엣지 수: {len(span_rel_edges) * 2}")  # 양방향이므로 *2

            #* 8. Span-level Triple 생성
            # span_id -> 포함된 token indices 매핑 생성
            # span_id_to_token_indices = {
            #     span_ids[i]: list(range(start, end + 1))
            #     for i, (start, end) in enumerate(candidate_spans)
            # }
            span_id_to_token_indices = {}
            for i, (span_id, (start, end)) in enumerate(zip(span_ids, candidate_spans)):
                span_id_to_token_indices[span_id] = list(range(start, end + 1))
            
            print(f"span_id_to_token_indices 매핑 크기: {len(span_id_to_token_indices)}")
            print(f"span_embeddings 크기: {span_embs.size(0)}")
            
            # span_id -> 가능한 relation ids 매핑 생성
            span_relation_candidates = defaultdict(set)
            for span_id, rel_id, rel_node_id in span_rel_edges:
                if span_id in span_id_to_token_indices:  # 필터링된 span만 포함
                    span_relation_candidates[span_id].add(rel_id)
            
            print(f"span_relation_candidates 매핑 크기: {len(span_relation_candidates)}")
            
            # Triple 후보 생성 (token-token, token-span, span-span)
            span_level_triples = extract_span_token_candidate_triples(
                token_ids=token_ids,
                span_ids=span_ids,
                span_relation_candidates=span_relation_candidates,
                span_id_to_token_indices=span_id_to_token_indices,
                relation_types=self.relation_types
            )
            # print('span_level_triples: ', span_level_triples)
            
            #* 9. Triple Scoring
            if not span_level_triples:
                print("Warning: No valid span-level triples found")
                # 빈 트리플인 경우 다음 배치로 넘어감
                continue

            print("\n=== Span-level Triple 처리 ===")
            span_level_scores = []
            span_level_cls_vectors_list = []
            valid_triple_indices = []  # 유효한 트리플의 원본 인덱스 저장

            for i, triple in enumerate(span_level_triples):
                # encode_span_triples는 한 번에 하나의 트리플만 처리
                cls_vector = self.triple_scorer.encode_span_triples(
                    [triple],  # 단일 트리플
                    token_embeddings=updated_entity_emb,
                    span_embeddings=span_embs,
                    relation_embeddings=rel_emb[b],
                    num_tokens=actual_length
                )
                
                # 유효한 결과만 저장
                if cls_vector.size(0) > 0:  # 빈 텐서가 아닌 경우
                    span_level_cls_vectors_list.append(cls_vector)
                    score = self.triple_scorer.score_span_triples(
                        [triple],
                        token_embeddings=updated_entity_emb,
                        span_embeddings=span_embs,
                        relation_embeddings=rel_emb[b],
                        num_tokens=actual_length
                    )
                    span_level_scores.append(score.item())
                    valid_triple_indices.append(i)

            # 결과 통합
            if span_level_cls_vectors_list:
                span_level_cls_vectors = torch.cat(span_level_cls_vectors_list, dim=0) # 개별적으로 처리된 CLS 벡터를 하나의 텐서로 결합해줌
                span_level_scores = torch.tensor(span_level_scores, device=self.device)
            else:
                print("Warning: No valid triples found")
                continue

            print(f"원본 트리플 수: {len(span_level_triples)}")
            print(f"유효한 트리플 수: {len(valid_triple_indices)}")
            print(f"CLS 벡터 크기: {span_level_cls_vectors.size()}")

            # Split with valid indices
            span_level_top_triples, top_scores, span_level_bottom_triples, bottom_scores, top_indices, bottom_indices = split_triples_by_score(
                [span_level_triples[i] for i in valid_triple_indices],  # 유효한 트리플만 사용
                span_level_scores,
                threshold=self.pooling_threshold,
                return_indices=True
            )

            # Sampling
            span_level_top_triples, top_scores, sampled_top_idx = sample_triples(
                span_level_top_triples, top_scores, self.sample_size, return_indices=True)
            span_level_bottom_triples, bottom_scores, sampled_bottom_idx = sample_triples(
                span_level_bottom_triples, bottom_scores, self.sample_size, return_indices=True)

            # CLS 벡터 샘플링 - 이제 인덱스가 span_level_cls_vectors와 일치
            sampled_top_cls = span_level_cls_vectors[[top_indices[i] for i in sampled_top_idx]]
            sampled_bottom_cls = span_level_cls_vectors[[bottom_indices[i] for i in sampled_bottom_idx]]
            
            # 입력 텍스트와의 코사인 유사도
            # input_text_emb = input_text_embs[b_idx].to(self.device)  # [H]
            # tripleset_encoder의 출력을 사용
            top_vector = self.tripleset_encoder(sampled_top_cls)      # [H]
            bottom_vector = self.tripleset_encoder(sampled_bottom_cls)  # [H]
            
            # 이제 차원이 맞음: [1, H] vs [1, H]
            sim_top = self.cos(input_text_emb.unsqueeze(0), top_vector.unsqueeze(0))
            sim_btm = self.cos(input_text_emb.unsqueeze(0), bottom_vector.unsqueeze(0))
            
            print(f"\n=== 코사인 유사도 ===")
            print(f"Top Triple Set: {sim_top.item():.4f}")
            print(f"Bottom Triple Set: {sim_btm.item():.4f}")
            
            # Triple verbalization
            node_id_to_token = node_id_to_token_map(
                entity_tokens=batch["tokens"][b],
                relation_types=self.relation_types
            )
            
            top_triples_text = convert_span_triples_to_text(
                triples=span_level_top_triples, 
                node_id_to_token=node_id_to_token,
                span_id_to_token_indices=span_id_to_token_indices,
                num_tokens=actual_length
            )
            
            bottom_triples_text = convert_span_triples_to_text(
                triples=span_level_bottom_triples, 
                node_id_to_token=node_id_to_token,
                span_id_to_token_indices=span_id_to_token_indices,
                num_tokens=actual_length
            )
            
            summary_top = verbalize_triples(top_triples_text)
            summary_btm = verbalize_triples(bottom_triples_text)
            
            # LLM 선호도 평가
            preferred = ask_llm_preference(
                input_text=batch["sentences"][b],
                summary_a=summary_top,
                summary_b=summary_btm
            )
            
            print(f"\n=== LLM 선호도 ===")
            print(f"선호도: {preferred}")
            
            # Preference learning loss 계산
            span_level_llm_loss = preference_learning_loss(sim_top, sim_btm, preferred)
            # span_level_llm_losses.append(span_level_llm_loss)
            
            print(f"배치 {b} Span-level LLM Loss: {span_level_llm_loss.item():.4f}")

            #! Span-level Validity-Aware Aggregation
            print(f"\n=== Span-level Validity-Aware Aggregation ===")

            # 1. top triple들의 head id 추출 및 중복 제거
            span_top_head_ids = [span_level_triples[idx][0] for idx in top_indices]
            span_unique_head_ids = set(span_top_head_ids)

            # 2. 엔티티/스팬 임베딩 업데이트
            updated_node_emb = graph.x.clone().to(self.device)  # [num_total_nodes, D]

            # 차원 확인 및 수정
            if updated_node_emb.dim() == 1:
                updated_node_emb = updated_node_emb.unsqueeze(-1).expand(-1, self.hidden_size).clone()
            elif updated_node_emb.size(1) != self.hidden_size:
                updated_node_emb = updated_node_emb.expand(-1, self.hidden_size).clone()

            print(f"Initial updated_node_emb shape: {updated_node_emb.shape}")
            print(f"Device 확인:")
            print(f"- updated_node_emb: {updated_node_emb.device}")
            print(f"- graph.x: {graph.x.device}")

            # top triple의 CLS 벡터 추출
            top_cls_vectors = span_level_cls_vectors[top_indices].to(self.device)  # [K, D]

            # 3. 각 head_id에 대해 업데이트 적용
            for head_id in span_unique_head_ids:
                updated_emb = self.update_entity_embedding(
                    entity_emb=updated_node_emb.clone(),  # 복사본 전달
                    cls_embeddings=top_cls_vectors,
                    triple_indices=top_indices,
                    triples=span_level_triples,
                    target_head_id=head_id
                )  # [D]
                
                # 해당 위치의 임베딩 업데이트 (contiguous 메모리로 변환)
                updated_node_emb[head_id] = updated_emb.clone().contiguous()

            print(f"Final updated_node_emb shape: {updated_node_emb.shape}")

            # 4. 업데이트된 임베딩을 그래프에 반영
            graph.x = updated_node_emb.to(graph.x.device)  # 그래프의 디바이스에 맞춰줌

            # 5. relation 임베딩 업데이트
            updated_relation_embedding = self.update_relation_embedding(
                node_emb=graph.x,
                node_type=graph.node_type.to(self.device)  # node_type도 디바이스 맞춰줌
            )

            # 6. 최종 업데이트된 임베딩을 그래프에 반영
            graph.x = updated_relation_embedding.to(graph.x.device)  # 다시 그래프의 디바이스에 맞춰줌

            print(f"노드 {len(span_unique_head_ids)}개 업데이트 완료")
            print(f"전체 노드 임베딩 크기: {updated_relation_embedding.size()}")
            print("="*50)

            # Relation Prediction 수행
            predicted_relations = predict_relations(
                model=self, 
                batch={
                    "tokens": [batch["tokens"][b]],
                    "lengths": batch["lengths"][b].unsqueeze(0),
                    "relation_ids": [batch["relation_ids"][b]],
                    "relation_types": self.relation_types
                }
            )

            # 현재 배치의 예측 결과
            current_predictions = predicted_relations[0]

            print(f"\n=== Relation Prediction 결과 ===")
            print(f"예측된 관계 수: {len(current_predictions)}")
            for h_span, rel_type, t_span in current_predictions:
                print(f"Head: {h_span}, Relation: {rel_type}, Tail: {t_span}")

            # Relation Loss 계산 (한 번만)
            if "relations" in batch and len(batch["relations"][b]) > 0:
                current_relation_loss = self.relation_loss.compute_relation_loss(
                    triples=triples,
                    triple_scores=token_level_scores,
                    gold_relations=batch["relations"][b]
                )
            else:
                current_relation_loss = torch.tensor(0.0).to(self.device)
            
            # relation_losses.append(current_relation_loss)
            relation_loss = current_relation_loss
            print(f"Relation Loss: {current_relation_loss.item()}")

            # 다른 loss들도 저장
            # entity_prefilter_losses.append(entity_prefilter_loss)
            # token_level_llm_losses.append(token_level_llm_loss)
            # span_prefilter_losses.append(span_prefilter_loss)

            final_loss = entity_prefilter_loss + token_level_llm_loss + span_prefilter_loss + span_level_llm_loss + relation_loss
            print(f"Final Loss: {final_loss.item()}")
        # # 전체 loss 계산
        # batch_losses = []
        # for b in range(len(graphs)):
        #     print(f"\n=== 그래프 {b}의 Loss 계산 ===")
            
        #     current_losses = []
            
        #     # 각 loss component 추가 - 0차원 텐서 처리 수정
        #     if isinstance(entity_prefilter_losses[b], torch.Tensor):
        #         if entity_prefilter_losses[b].dim() > 0:
        #             current_losses.append(entity_prefilter_losses[b].mean())
        #         else:
        #             current_losses.append(entity_prefilter_losses[b])
        #     else:
        #         current_losses.append(torch.tensor(entity_prefilter_losses[b], device=self.device))

        #     # 나머지 loss들도 동일한 방식으로 처리
        #     for loss in [token_level_llm_losses[b], span_prefilter_losses[b], span_level_llm_losses[b], relation_losses[b]]:
        #         if isinstance(loss, torch.Tensor):
        #             if loss.dim() > 0:
        #                 current_losses.append(loss.mean())
        #             else:
        #                 current_losses.append(loss)
        #         else:
        #             current_losses.append(torch.tensor(loss, device=self.device))
            
        #     # loss 통계 출력
        #     print(f"Entity Prefilter Loss: {current_losses[0].item()}")
        #     print(f"Token Level LLM Loss: {current_losses[1].item()}")
        #     print(f"Span Prefilter Loss: {current_losses[2].item()}")
        #     print(f"Span Level LLM Loss: {current_losses[3].item()}")
        #     print(f"Relation Loss: {current_losses[4].item()}")
            
        #     # 현재 그래프의 평균 loss 계산
        #     graph_loss = torch.stack(current_losses).mean()
        #     batch_losses.append(graph_loss)

        # # 전체 배치의 평균 loss
        # final_loss = torch.stack(batch_losses).mean()
        # print(f"\n배치 평균 Loss: {final_loss.item():.4f}")

        return {
            "graphs": graphs,
            "final_loss": final_loss
        }






#* ===================
# relation_types = ["used-for", "feature-of", "hyponym-of", "part-of", "compare", "conjunction", "evaluate-for"]
# # relation types = relation types + ["COMPOUND", "COMPOUND_INV"]


# with open("data/train_sample_scierc.json") as f:
#         data = [json.loads(line) for line in f]

# model = IEModel()


# # 데이터셋 읽기 (json line 형식)
# samples = []
# with open("data/train_sample_scierc.json") as f:
#     samples = [json.loads(line) for line in f]

# output = model(samples)
# batch = build_batch(samples)
# output = model(batch)

# sample = data[0]  # 첫 문서만 테스트
# sentences = sample["sentences"]
# ner_spans = sample["ner"]
# model = IEModel()
# model(batch)