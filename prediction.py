from typing import List, Dict, Tuple
import torch
from utils import build_bipartite_graph
from triplet_scorer import extract_candidate_triples

# --------------------------
# 1. NER Prediction
# --------------------------
class NERPredictor:
    def __init__(self, id2label: Dict[int, str]):
        self.id2label = id2label  # entity type index → label

    def predict(self,
                entity_emb: torch.Tensor,
                ner_classifier: torch.nn.Module,
                token_span_map: Dict[int, Tuple[int, int]]
                ) -> List[List[Tuple[int, int, str]]]:
        """
        Returns:
            ner_predictions: List of [start, end, label]
        """
        ner_logits = ner_classifier(entity_emb)  # [N_entity, num_types]
        ner_preds = ner_logits.argmax(dim=-1).tolist()

        results = []
        for ent_id, type_id in enumerate(ner_preds):
            if type_id == 0:  # skip 'O'
                continue
            start, end = token_span_map[ent_id]
            label = self.id2label[type_id]
            results.append((start, end, label))
        return results


# --------------------------
# 2. RE Prediction
# --------------------------
class REPredictor:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def predict(
        self,
        num_tokens: int,
        span_triples: list,           # (head_id, rel_id, tail_id, r_type)
        span_scores: torch.Tensor,    # [num_span_triples]
        token_to_text: dict,          # 노드 id → 텍스트
        relation_indices: list        # relation node 인덱스 리스트
    ) -> list:
        """
        r(두 번째 값)이 relation node 인덱스(사전정의 7개)인 경우만 예측
        """
        if not span_triples or span_scores is None or len(span_scores) == 0:
            return []
        predictions = []
        for (h, r, t, r_type), score in zip(span_triples, span_scores):
            if r not in relation_indices:
                print(f"Warning: Relation index {r} is not in relation_indices")
                continue
            score_value = score.item() if isinstance(score, torch.Tensor) else score
            if score_value < self.threshold:
                continue
            head_text = token_to_text.get(h, f"UNK_{h}")
            rel_text = token_to_text.get(r + num_tokens, f"UNK_{r}")
            tail_text = token_to_text.get(t, f"UNK_{t}")
            
            predictions.append((head_text, rel_text, tail_text, score_value))
        predictions.sort(key=lambda x: x[3], reverse=True)
        return predictions


# --------------------------
# 3. RE+ Prediction
# --------------------------
class REPlusPredictor:
    def __init__(self, threshold: float = 0.0, id2label: Dict[int, str] = None):
        self.threshold = threshold
        self.id2label = id2label  # entity type index → label

    def predict(self,
                triples: List[Tuple[int, int, int, str]],
                triple_scores: torch.Tensor,
                token_span_map: Dict[int, Tuple[int, int]],
                ner_preds: List[int]
                ) -> List[Tuple[Tuple[int, int], str, str, Tuple[int, int], str]]:
        """
        Returns:
            re+ predictions: List of (h_span, h_type, rel_type, t_span, t_type)
        """
        results = []
        for (h, _, t, r_type), score in zip(triples, triple_scores.tolist()):
            if score < self.threshold:
                continue
            h_span = token_span_map[h]
            t_span = token_span_map[t]
            h_type = self.id2label[ner_preds[h]]
            t_type = self.id2label[ner_preds[t]]
            results.append((h_span, h_type, r_type, t_span, t_type))
        return results


def predict_relations(model, batch):
    """
    배치에서 관계를 예측합니다.
    Args:
        model: RePoolModel 인스턴스
        batch: 입력 배치 데이터
    Returns:
        List[List[Tuple[str, str, str, float]]] - 배치별 (head_text, relation_type, tail_text, confidence) 리스트
    """
    # 모델을 평가 모드로 설정
    model.eval()
    
    with torch.no_grad():
        # Token representation
        token_out = model.entity_token_rep(batch["tokens"], batch["lengths"])
        entity_embs = token_out["embeddings"]
        entity_mask = token_out["mask"]
        
        # Relation Embedding
        PAD_ID = -1
        rel_ids = batch["relation_ids"]
        max_len = max(len(x) for x in rel_ids)
        padded_rel_ids = [x + [PAD_ID]*(max_len-len(x)) for x in rel_ids]
        relation_ids_tensor = torch.LongTensor(padded_rel_ids).to(model.device)
        rel_emb = model.relation_token_rep(relation_ids_tensor)
        
        # Entity Prefiltering
        scores, _ = model.entity_prefilter(entity_embs, entity_mask)
        keep_mask = (scores > 0.5).float()
        
        # 배치별 예측 결과를 저장할 리스트
        batch_predictions = []
        
        for b in range(entity_embs.size(0)):
            # 그래프 생성 및 트리플 추출
            entity_emb = entity_embs[b]
            actual_length = batch["lengths"][b].item()
            keep_mask_b = keep_mask[b]
            
            # 그래프 생성
            all_node_emb = torch.cat([entity_emb[:actual_length], rel_emb[b][:len(model.relation_types)]], dim=0)
            keep_ids = (keep_mask_b[:actual_length] == 1).nonzero(as_tuple=True)[0].tolist()
            
            graph = build_bipartite_graph(
                all_node_emb=all_node_emb,
                keep_ids=keep_ids,
                num_relation=len(model.relation_types),
                bidirectional=True,
                use_node_type=True,
                keep_mask=keep_mask_b[:actual_length].tolist()
            )
            
            # 트리플 추출 및 점수 계산
            triples = extract_candidate_triples(graph)
            if not triples:
                batch_predictions.append([])
                continue
            
            scores = model.triple_scorer(
                triples=triples,
                token_embeddings=all_node_emb,
                relation_embeddings=rel_emb[b][:len(model.relation_types)],
                lengths=actual_length
            )
            
            # 텍스트 매핑 생성
            token_to_text = {i: token for i, token in enumerate(batch["tokens"][b][:actual_length])}
            
            # REPredictor를 사용하여 예측
            predictor = REPredictor(threshold=0.5)
            relation_indices = [i for i, t in enumerate(graph.node_type) if t == 1]
            predictions = predictor.predict(
                span_triples=triples,
                span_scores=scores,
                token_to_text=token_to_text,
                relation_indices=relation_indices
            )
            
            batch_predictions.append(predictions)
    
    return batch_predictions
