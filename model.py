import torch
import torch.nn as nn
from typing import List, Dict, Any

from modules import (
    Dual_View_BipartiteKG_Construction,
    Token_Level_Graph_Pooling,
    Token_to_Span_Composition,
    Span_level_Graph_Pooling,
    Prediction
)
from prediction import NERPredictor, REPredictor, REPlusPredictor, predict_relations

class RePoolModel(nn.Module):
    def __init__(
        self,
        bert_model_name: str = "bert-base-cased",
        hidden_size: int = 768,
        sample_size: int = 100,
        device: torch.device = None,
        relation_types: List[str] = None
    ):
        super().__init__()
        
        # Device 설정
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Relation types 설정
        self.default_relation_types = [
            "USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", 
            "COMPARE", "CONJUNCTION", "EVALUATE-FOR"
        ]
        self.relation_types = relation_types or self.default_relation_types
        
        # 모듈 초기화
        self.dual_view = Dual_View_BipartiteKG_Construction(
            bert_model_name=bert_model_name,
            hidden_size=hidden_size,
            device=self.device
        )
        
        self.token_level = Token_Level_Graph_Pooling(
            bert_model_name=bert_model_name,
            hidden_size=hidden_size,
            device=self.device
        )
        
        self.token_to_span = Token_to_Span_Composition(
            bert_model_name=bert_model_name,
            hidden_size=hidden_size,
            device=self.device
        )
        
        self.span_level = Span_level_Graph_Pooling(
            bert_model_name=bert_model_name,
            hidden_size=hidden_size,
            device=self.device
        )

        # Prediction 모듈 초기화
        self.id2label = {i: label for i, label in enumerate(self.relation_types)}
        self.ner_predictor = NERPredictor(id2label=self.id2label)
        self.re_predictor = REPredictor(threshold=0.5)
        self.replus_predictor = REPlusPredictor(threshold=0.6, id2label=self.id2label)
        
        self.repool_predictor = None  # 나중에 relation_types가 정해진 후 초기화
        
        # 기타 설정
        self.hidden_size = hidden_size
        self.sample_size = sample_size
        
        print(f"Model initialized on device: {self.device}")
        print(f"Using relation types: {self.relation_types}")

    def to(self, device):
        """모델의 device를 변경합니다."""
        self.device = device
        self.dual_view = self.dual_view.to(device)
        self.token_level = self.token_level.to(device)
        self.token_to_span = self.token_to_span.to(device)
        self.span_level = self.span_level.to(device)
        return super().to(device)

    def process_single_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """단일 샘플을 robust하게 처리합니다."""
        # 1. Entity/Relation Embeddings 생성
        lengths = torch.tensor([len(sent) for sent in sample["sentences"]]).to(self.device)
        token_out = self.dual_view.get_entity_embeddings(sample["sentences"], lengths)
        entity_embs = token_out["embeddings"]
        entity_mask = token_out["mask"]
        relation_ids = torch.arange(len(self.relation_types)).to(self.device)
        rel_emb = self.dual_view.get_relation_embeddings(relation_ids)

        # 2. Entity Prefilter
        prefilter_out = self.dual_view.get_entity_prefilter(
            entity_embs, entity_mask, sample["sentences"], sample["ner"]
        )
        keep_mask = prefilter_out["keep_mask"]
        prefilter_loss = prefilter_out["prefilter_loss"]

        # 3. 이분 그래프 구성
        graph = self.dual_view.construct_graph(
            entity_embs=entity_embs,
            rel_emb=rel_emb,
            keep_mask=keep_mask,
            actual_entity_length=lengths
        )

        # 4. Token Level Processing
        graph, triples, triple_cls_vectors, top_triples, top_scores, bottom_triples, bottom_scores, top_indices, bottom_indices = self.token_level.process_token_triples(graph)

        # 5. Token Level LLM Alignment & Aggregation
        if top_triples is None or len(top_triples) == 0 or triple_cls_vectors is None or len(top_indices) == 0:
            token_level_llm_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
        else:
            token_level_llm_loss = self.token_level.token_llm_alignment(
                input_text=sample["sentences"],
                relation_types=self.relation_types,
                triple_cls_vectors=triple_cls_vectors,
                top_triples=top_triples,
                top_scores=top_scores,
                bottom_triples=bottom_triples,
                bottom_scores=bottom_scores,
                top_indices=top_indices,
                bottom_indices=bottom_indices
            )
            graph = self.token_level.validity_aware_aggregation(graph, triples, triple_cls_vectors, top_indices)

        # 6. Token to Span Composition
        actual_length = lengths[0].item()
        graph, span_embs, span_ids, candidate_spans, span_prefilter_loss = self.token_to_span.compose(
            graph=graph,
            lengths=lengths,
            keep_mask=keep_mask,
            sentences=sample["sentences"],
            ner_spans=sample["ner"],
            actual_length=actual_length,
            rel_emb=rel_emb
        )
        if span_prefilter_loss is None:
            span_prefilter_loss = torch.tensor(0.0, requires_grad=True, device=self.device)

        # 전체 노드에 대한 token_to_text 매핑 생성
        tokens = [token for sent in sample["sentences"] for token in sent]
        token_to_text = {i: token for i, token in enumerate(tokens)}
        num_tokens = len(tokens)
        for rel_idx, rel_type in enumerate(self.relation_types):
            token_to_text[num_tokens + rel_idx] = rel_type

        # 7. Span Level Processing (if valid spans exist)
        span_level_outputs = None
        predictions = []
        if span_ids is not None and candidate_spans is not None and len(candidate_spans) > 0 and span_embs is not None:
            result = self.span_level.process_span_triples(graph, candidate_spans, span_ids, span_embs, rel_emb)
            if result is None:
                span_level_llm_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                span_top_triples = span_top_scores = span_id_to_token_indices = None
            else:
                (candidate_triples,triple_cls_vectors, span_top_triples, span_top_scores, 
                 span_bottom_triples, span_bottom_scores, span_top_indices, 
                 span_bottom_indices, span_id_to_token_indices) = result
                if span_top_triples is None or len(span_top_triples) == 0 or triple_cls_vectors is None or len(span_top_indices) == 0:
                    span_level_llm_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
                else:
                    span_level_llm_loss = self.span_level.span_llm_alignment(
                        input_text=sample["sentences"],
                        relation_types=self.relation_types,
                        triple_cls_vectors=triple_cls_vectors,
                        top_triples=span_top_triples,
                        top_scores=span_top_scores,
                        bottom_triples=span_bottom_triples,
                        bottom_scores=span_bottom_scores,
                        top_indices=span_top_indices,
                        bottom_indices=span_bottom_indices,
                        span_id_to_token_indices=span_id_to_token_indices,
                        num_tokens=num_tokens,
                        candidate_spans=candidate_spans
                    )
                    graph = self.span_level.validity_aware_aggregation(
                        graph,
                        candidate_triples,
                        triple_cls_vectors,
                        span_top_indices
                    )
            # 8. 예측 및 loss 통합 처리
            #   relation_indices = [i - num_tokens for i, t in enumerate(graph.node_type) if t == 1]
            try:
                if self.repool_predictor is None:
                    from modules import RePoolPredictor
                    self.repool_predictor = RePoolPredictor(self.relation_types, device=self.device)
                outputs = self.repool_predictor.predict_and_loss(
                    sample=sample,
                    graph=graph,
                    span_top_triples=span_top_triples,
                    span_top_scores=span_top_scores,
                    token_to_text=token_to_text,
                    num_tokens=num_tokens,
                    gold_relations=sample.get("relations", []),
                    sentences=sample["sentences"],
                    prefilter_loss=prefilter_loss,
                    span_prefilter_loss=span_prefilter_loss,
                    token_level_llm_loss=token_level_llm_loss,
                    span_level_llm_loss=span_level_llm_loss
                )
            except Exception as e:
                print(f"[Warning] RePoolPredictor 예외 발생: {e}")
                outputs = {
                    "predictions": [],
                    "relation_loss": torch.tensor(0.0, requires_grad=True, device=self.device),
                    "prefilter_loss": prefilter_loss,
                    "span_prefilter_loss": span_prefilter_loss,
                    "token_level_llm_loss": token_level_llm_loss,
                    "span_level_llm_loss": span_level_llm_loss,
                    "final_loss": prefilter_loss + (span_prefilter_loss if span_prefilter_loss is not None else 0.0) + token_level_llm_loss + span_level_llm_loss
                }
            span_level_outputs = {
                "span_level_llm_loss": span_level_llm_loss,
                "relation_loss": outputs["relation_loss"],
                "predictions": outputs["predictions"],
                "graph": graph,
                "span_triples": span_top_triples,
                "span_scores": span_top_scores,
                "span_id_to_token_indices": span_id_to_token_indices,
                "span_prefilter_loss": span_prefilter_loss
            }
        else:
            span_level_llm_loss = torch.tensor(0.0, requires_grad=True, device=self.device)
            relation_loss_val = torch.tensor(0.0, requires_grad=True, device=self.device)
            predictions = []

        return {
            "graph": graph,
            "entity_prefilter_loss": prefilter_loss,
            "span_prefilter_loss": span_prefilter_loss,
            "token_level_llm_loss": token_level_llm_loss,
            "span_level_outputs": span_level_outputs,
            "predictions": {
                "re": outputs["predictions"] if span_level_outputs is not None else []
            },
            "final_loss": outputs["final_loss"] if span_level_outputs is not None else (
                prefilter_loss + (span_prefilter_loss if span_prefilter_loss is not None else 0.0) + token_level_llm_loss + span_level_llm_loss
            )
        }

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """배치 단위 처리를 robust하게 수행합니다."""
        batch_size = len(batch["sentences"])
        batch_outputs = []
        for i in range(batch_size):
            sample = {
                "sentences": batch["sentences"][i],
                "tokens": batch["tokens"][i] if "tokens" in batch else batch["sentences"][i],
                "lengths": batch["lengths"][i] if isinstance(batch["lengths"], list) else batch["lengths"][i].item(),
                "ner": batch["ner"][i] if "ner" in batch else None,
                "relation_ids": batch["relation_ids"][i] if "relation_ids" in batch else None,
                "relations": batch["relations"][i] if "relations" in batch else None
            }
            outputs = self.process_single_sample(sample)
            batch_outputs.append(outputs)
            print(f"[Batch {i}] Loss: {outputs.get('final_loss', 'N/A')}")
        # Loss 취합
        def to_tensor(val, device):
            if isinstance(val, torch.Tensor):
                return val
            else:
                return torch.tensor(val, device=device)

        total_entity_prefilter_loss = torch.stack(
            [to_tensor(out["entity_prefilter_loss"], self.device) for out in batch_outputs]
        ).mean()
        total_span_prefilter_loss = torch.stack(
            [to_tensor(out["span_prefilter_loss"], self.device) for out in batch_outputs]
        ).mean()
        total_token_level_llm_loss = torch.stack(
            [to_tensor(out["token_level_llm_loss"], self.device) for out in batch_outputs]
        ).mean()
        # Span level loss와 relation loss (있는 경우만)
        span_level_losses = []
        relation_losses = []
        for out in batch_outputs:
            if out["span_level_outputs"] is not None:
                span_level_losses.append(to_tensor(out["span_level_outputs"]["span_level_llm_loss"], self.device))
                relation_losses.append(to_tensor(out["span_level_outputs"]["relation_loss"], self.device))
        total_span_level_llm_loss = torch.stack(span_level_losses).mean() if span_level_losses else torch.tensor(0.0).to(self.device)
        total_relation_loss = torch.stack(relation_losses).mean() if relation_losses else torch.tensor(0.0).to(self.device)
        # 최종 Loss 계산 (prefilter, span prefilter 포함)
        final_loss = (
            total_entity_prefilter_loss +
            total_span_prefilter_loss +
            total_token_level_llm_loss +
            total_span_level_llm_loss +
            total_relation_loss
        )
        
        return {
            "batch_outputs": batch_outputs,
            "final_loss": final_loss,
            "entity_prefilter_loss": total_entity_prefilter_loss,
            "span_prefilter_loss": total_span_prefilter_loss,
            "token_level_llm_loss": total_token_level_llm_loss,
            "span_level_llm_loss": total_span_level_llm_loss,
            "relation_loss": total_relation_loss
        }