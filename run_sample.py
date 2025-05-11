import torch
import json
from torch.nn.utils.rnn import pad_sequence

from token_rep import EntityTokenRep, RelationTokenRep
from entnode_prefilter import EntityNodeFilter
from data_preprocess import generate_token_labels
from utils import get_topk_candidates, build_bipartite_graph, TextEncoder
from triplet_scorer import BERTTripleScorer, extract_candidate_triples, node_id_to_token_map, convert_id_triples_to_text, split_triples_by_score
from llm_guidance import TripleSetEncoder,verbalize_triples, dpo_alignment_loss, ask_llm_preference


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === 데이터 로딩 ===
with open("data/train_sample_scierc.json") as f:
        data = [json.loads(line) for line in f]

relation_types = ["used-for", "feature-of", "hyponym-of", "part-of", "compare", "conjunction", "evaluate-for"]


# === 데이터 추출 ===
sample = data[0]  # 첫 문서만 테스트
sentences = sample["sentences"]
ner_spans = sample["ner"]

# input text 임베딩
flattened_sentences = [" ".join(s) for s in sentences]
text_encoder = TextEncoder("bert-base-cased").to(device)
input_text_embedding = text_encoder(flattened_sentences)

# 개별 토큰
entity_tokens = [tok for sent in sentences for tok in sent]  # flat list
# assert len(entity_tokens) == entity_emb.size(0)

# === 라벨 생성 ===
labels = generate_token_labels(sentences, ner_spans)  # List[List[int]]
lengths = torch.tensor([len(s) for s in sentences])


# === 모델 구성 ===
token_model = EntityTokenRep(model_name="bert-base-cased")
relation_model = RelationTokenRep(num_relations=len(relation_types),
                                embedding_dim=768,
                                method="pretrained_lm",  # 또는 "onehot"
                                relation_names=relation_types
                            )
filter_model = EntityNodeFilter(hidden_size=768)

# === Token 임베딩 ===
out = token_model(sentences, lengths)
embeddings, mask = out["embeddings"], out["mask"]

flat_entity_emb = embeddings[mask.bool()]  # → [N_entity, D]

# 라벨을 텐서로 만들 때 문장 별 길이에 맞게 패딩 처리 
label_tensor = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=0)
flat_label_tensor = label_tensor[mask.bool()]  # [N_entity]


# === Relation node 임베딩 ===
rel_emb = relation_model()


# === Entity Node Pre-Filtering 실행 ===
#! Loss
scores, prefilter_loss = filter_model(flat_entity_emb.unsqueeze(0), labels=flat_label_tensor.unsqueeze(0))

#~ candidate 추출해서 그걸로 bipartite graph 구성할 때 쓴 코드
# top_k = 7  # 또는 min(L, max_top_k) + add_top_k 같은 방식
# candidates: [B, L, D], label_tensor: [B, L], mask: [B, L], idx: [B, L]
# candidate_embeddings, candidate_labels, candidate_masks, candidate_indices = [
#     get_topk_candidates(scores, tensor, topk=top_k)[0] for tensor in [
#         embeddings, label_tensor, mask, torch.arange(embeddings.size(1)).unsqueeze(0).repeat(embeddings.size(0), 1)
#         # 필터링 스코어(scores)를 기준으로 상위 top_k개의 토큰을 여러 개의 텐서에서 동시에 추출하는 리스트 컴프리헨션
#         # 리스트 컴프리헨션의 결과를 분리하여 candidate_embeddings, candidate_labels, candidate_masks, candidate_indices 각각 저장
#     ]
# ]

#~ original tokens에서 탈락한 노드는 keep_mask 할당해서 엣지 생성 제한하는 형태
# scores: [B, L]
topk = 7
sorted_idx = torch.argsort(scores, dim=1, descending=True)
keep_mask = torch.zeros_like(scores)  # [B, L]
keep_mask.scatter_(1, sorted_idx[:, :topk], 1)  # 상위 topk만 1로 유지


# === bidirectional complete bipartite graph 생성 ===
# top-k만 relation과 edge 생성, 탈락된 토큰은 isolate node로 존재
# entity_emb = embeddings[0]
keep_ids = (keep_mask[0] == 1).nonzero(as_tuple=True)[0].tolist()
data = build_bipartite_graph(flat_entity_emb, keep_ids, rel_emb, bidirectional=True)

# Output:
# Data(x=[17, 768], edge_index=[2, 140], edge_type=[140], node_type=[17])

triples = extract_candidate_triples(data)
node_id_to_token = node_id_to_token_map(entity_tokens, relation_types)
text_triples = convert_id_triples_to_text(triples, node_id_to_token)


scorer = BERTTripleScorer("bert-base-cased")
token_level_scores = scorer(text_triples)

top_triples, top_scores, bottom_triples, bottom_scores = split_triples_by_score(
    text_triples,
    token_level_scores,
    top_k=10
)

print("🔼 Top triples:")
for t, s in zip(top_triples, top_scores):
    print(f"{t} → {s.item():.4f}")

print("🔽 Bottom triples:")
for t, s in zip(bottom_triples[:5], bottom_scores[:5]):  # 5개만
    print(f"{t} → {s.item():.4f}")


tripleset_encoder = TripleSetEncoder("bert-base-cased").to(device)

# top_triples, bottom_triples 입력
top_vector = tripleset_encoder.encode_triple_set(top_triples)      # [H]
bottom_vector = tripleset_encoder.encode_triple_set(bottom_triples)  # [H]

# cosine similarity with input_text_embedding
cos = torch.nn.CosineSimilarity(dim=-1)
sim_top = cos(input_text_embedding, top_vector)
sim_btm = cos(input_text_embedding, bottom_vector)

print("Cosine similarity to input text:")
print("Top triple set →", sim_top.item())
print("Bottom triple set →", sim_btm.item())


# verbalize
summary_top = verbalize_triples(top_triples)
summary_btm = verbalize_triples(bottom_triples)

print("================")
print("summary_top: ", summary_top)
print("summary_btm: ", summary_btm)

# LLM으로 선호도 평가 받기
preferred = ask_llm_preference(flattened_sentences, summary_top, summary_btm)

# DPO-style loss 계산
loss = dpo_alignment_loss(sim_top, sim_btm, preferred)