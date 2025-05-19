import torch
import json
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

from token_rep import EntityTokenRep, RelationTokenRep
from prefilter import TokenFilter, SpanFilter
from generate_labels import generate_token_labels, generate_span_labels
from utils import build_bipartite_graph, TextEncoder, sample_triples, get_candidate_spans, get_span_embeddings, filter_candidate_spans, extract_token_relation_edges
from triplet_scorer import BERTTripleScorer, extract_candidate_triples, node_id_to_token_map, convert_id_triples_to_text, split_triples_by_score, extract_span_token_candidate_triples, remove_edges_of_bottom_triples
from llm_guidance import TripleSetEncoder,verbalize_triples, preference_learning_loss, ask_llm_preference
from aggregation import EntityEmbeddingUpdater, RelationEmbeddingUpdater

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# === 데이터 로딩 ===
with open("data/train_sample_scierc.json") as f:
        data = [json.loads(line) for line in f]

relation_types = ["used-for", "feature-of", "hyponym-of", "part-of", "compare", "conjunction", "evaluate-for"]
num_relations = len(relation_types)

# === 데이터 추출 ===
sample = data[0]  # 첫 문서만 테스트
sentences = sample["sentences"]
ner_spans = sample["ner"]

# input text 임베딩
# flattened_sentences = [" ".join(s) for s in sentences]
# input_text = " ".join([" ".join(sent) for sent in sentences])
flattened_sentences = " ".join(" ".join(sent) for sent in sentences)

# 전체 텍스트의 임베딩
text_encoder = TextEncoder("bert-base-cased").to(device)
input_text_embedding = text_encoder(flattened_sentences)

# 개별 토큰
entity_tokens = [tok for sent in sentences for tok in sent]  # flat list
# assert len(entity_tokens) == entity_emb.size(0)

# === 토큰 라벨 생성 ===
# (해당 토큰이 ner에 포함되면 1, 포함 안되면 0) -> prefilter의 정답값으로 사용함
labels = generate_token_labels(sentences, ner_spans)  # List[List[int]]
lengths = torch.tensor([len(s) for s in sentences]).to(device)


# === 모델 구성 === (__init__)
token_model = EntityTokenRep(model_name="bert-base-cased").to(device)
relation_model = RelationTokenRep(
                    relation_types=relation_types,                     # relation 이름 리스트
                    lm_model_name="bert-base-uncased",                 # 사용할 LM 이름
                    freeze_bert=True                                   # 임베딩 고정 여부 (True면 학습 안 함)
                ).to(device)
token_filter = TokenFilter(hidden_size=768).to(device)
span_filter = SpanFilter(hidden_size=768).to(device)

#! hidden_size config에 정의된 거 사용하게 바꾸기?
# hidden_size = config.hidden_size  # 일반적으로 768
# scorer = BERTTripleScorer(bert_model, hidden_size=hidden_size).to(device)
scorer = BERTTripleScorer(hidden_size=768).to(device)
tripleset_encoder = TripleSetEncoder().to(device)


# === Token 임베딩 ===
out = token_model(sentences, lengths)
embeddings, mask = out["embeddings"], out["mask"]

flat_entity_emb = embeddings[mask.bool()]  # [N, D]

# 라벨을 텐서로 만들 때 문장 별 길이에 맞게 패딩 처리 
label_tensor = pad_sequence([torch.tensor(l) for l in labels], batch_first=True, padding_value=0).to(device)
flat_label_tensor = label_tensor[mask.bool()]  # [N_entity].to(device)


# === Relation node 임베딩 ===
rel_ids = torch.arange(num_relations).to(device)  # shape: [num_relations]
rel_emb = relation_model(rel_ids)


# === Entity Node Pre-Filtering 실행 ===
#! Loss
scores, prefilter_loss = token_filter(flat_entity_emb.unsqueeze(0), labels=flat_label_tensor.unsqueeze(0))

#~ original tokens에서 탈락한 노드는 keep_mask 할당해서 엣지 생성 제한하는 형태
# scores: [B, L]
topk = 12
sorted_idx = torch.argsort(scores, dim=1, descending=True)
keep_mask = torch.zeros_like(scores)  # [B, L]
keep_mask.scatter_(1, sorted_idx[:, :topk], 1)  # 상위 topk만 1로 유지

# flat_keep_mask = keep_mask[mask.bool()]    # [N], 1이면 keep, 0이면 drop
# flat_keep_mask = keep_mask.view(-1)[mask.view(-1).bool()]
flat_keep_mask = keep_mask.view(-1)[:flat_entity_emb.size(0)]  # [N]


# Drop된 토큰 위치를 제로 벡터로 masking
# 스팬 임베딩 생성 시 탈락된 토큰이 포함되지 않도록 하기 위함
flat_entity_emb = flat_entity_emb * flat_keep_mask.unsqueeze(-1)

all_node_emb = torch.cat([flat_entity_emb, rel_emb], dim=0)  # [N + R, D]

# === bidirectional complete bipartite graph 생성 ===
# top-k만 relation과 edge 생성, 탈락된 토큰은 isolate node로 존재
# entity_emb = embeddings[0]
keep_ids = (keep_mask[0] == 1).nonzero(as_tuple=True)[0].tolist()
data = build_bipartite_graph(
                            all_node_emb, 
                            keep_ids, 
                            num_relation=rel_emb.size(0), 
                            bidirectional=True, 
                            keep_mask=flat_keep_mask.squeeze(0).tolist()
                            )


# Output:
# Data(x=[17, 768], edge_index=[2, 140], edge_type=[140], node_type=[17])

#* Token-level 
triples = extract_candidate_triples(data, compound_edge_type=0)
# node_id_to_token = node_id_to_token_map(entity_tokens, relation_types) # entity node, relation node id 값 부여
# text_triples = convert_id_triples_to_text(triples, node_id_to_token)

print('======triples 리스트: ', triples[900:])

print('triples 개수: ', len(triples))

# token_level_scores = scorer(text_triples)
# triple_cls_vectors = scorer.encode_triples(text_triples).to(device)  # [N, H] _ Triple별 CLS 벡터 추출

#! triples가 임베딩으로 버트에 입력되도록 바꿔주어야 함. 이거는 아마 숫자 그대로를 텍스트로 했을 것 같음.
token_level_scores = scorer(triples, all_node_emb)
triple_cls_vectors = scorer.encode_triples(triples, all_node_emb).to(device)  # [N, H] _ Triple별 CLS 벡터 추출


top_triples, top_scores, bottom_triples, bottom_scores, top_indices, bottom_indices = split_triples_by_score(
    # text_triples,
    triples,
    token_level_scores,
    threshold=0.25,
    return_indices=True  # top/bottom triple의 인덱스도 반환하도록
)

# 엣지 개수: 함수 호출 전
num_edges_before = data.edge_index.size(1)
# num_edges_bf = data.edge_mask.sum().item()  # True인 개수만 카운트


# bottom 트리플 엣지 삭제
data = remove_edges_of_bottom_triples(data, bottom_triples,
                                semantic_type_id=2,
                                compound_type_ids=(0, 1))

# 엣지 개수: 함수 호출 후
num_edges_after = data.edge_index.size(1)
# num_edges_af = data.edge_mask.sum().item()  # True인 개수만 카운트


print(f"# of edges before pruning: {num_edges_before}")
print(f"# of edges after pruning:  {num_edges_after}")
# print(f"# of edges_mask before pruning: {num_edges_bf}")
# print(f"# of edges_mask after pruning:  {num_edges_af}")


print("🔼 Top triples:")
for t, s in zip(top_triples[:5], top_scores):
    print(f"{t} → {s.item():.4f}")

print("🔽 Bottom triples:")
for t, s in zip(bottom_triples[:5], bottom_scores[:5]):  # 5개만
    print(f"{t} → {s.item():.4f}")

# #! top-k 제외 나머지 엣지는 삭제해주기
# # 저장할 edge 정보
# token_ids = list(range(len(flat_entity_emb)))  # token node ids
# keep_edges = set()

# for h, r, t in top_triples:
#     rel_node_id = r  # rel_id를 rel node id로 변환
#     keep_edges.add((h, rel_node_id))  # entity → relation
#     keep_edges.add((rel_node_id, t))  # relation → entity

# # print(type(src), type(tgt))
# for e in keep_edges:
#     print(type(e[0]), type(e[1]))

# # edge filtering
# new_edge_index = []
# new_edge_type = []

# for i in range(data.edge_index.size(1)):
#     src = data.edge_index[0, i].item()
#     tgt = data.edge_index[1, i].item()
#     rel = data.edge_type[i].item()
#     print(f"CHECKING edge ({src}, {tgt}) → keep? { (src, tgt) in keep_edges }")

#     if (src, tgt) in keep_edges:
#         new_edge_index.append([src, tgt])
#         new_edge_type.append(rel)

# print("new_edge_index): ", new_edge_index) 

# # 재구성
# data.edge_index = torch.tensor(new_edge_index, dtype=torch.long).t().contiguous()
# data.edge_type = torch.tensor(new_edge_type, dtype=torch.long)

# print("======== 재구성")
# print(data.edge_index)

# 각 top-k, bottom triples에서 일부만 랜덤하게 샘플링
sample_size = 10  # 나중에 비율 형태로 변경

top_triples, top_scores, sampled_top_idx = sample_triples(top_triples, top_scores, sample_size, return_indices=True)
bottom_triples, bottom_scores, sampled_bottom_idx = sample_triples(bottom_triples, bottom_scores, sample_size, return_indices=True)

# === 샘플링된 인덱스를 원래 triple_cls_vectors의 인덱스로 매핑 ===
# e.g., top_indices = [12, 35, 46, ...] → sampled_top_idx = [1, 3, 5]
sampled_top_cls = triple_cls_vectors[[top_indices[i] for i in sampled_top_idx]]  # [sample_size, H]
sampled_bottom_cls = triple_cls_vectors[[bottom_indices[i] for i in sampled_bottom_idx]]  # [sample_size, H]


# 이제 encode_triple_set에 넣기
top_vector = tripleset_encoder(sampled_top_cls)
bottom_vector = tripleset_encoder(sampled_bottom_cls)

# top_triples, bottom_triples 입력
# top_vector = tripleset_encoder.encode_triple_set(top_triples)      # [H]
# bottom_vector = tripleset_encoder.encode_triple_set(bottom_triples)  # [H]

# cosine similarity with input_text_embedding
cos = torch.nn.CosineSimilarity(dim=-1)
sim_top = cos(input_text_embedding, top_vector).to(device)
sim_btm = cos(input_text_embedding, bottom_vector).to(device)

# print("Cosine similarity to input text:")
# print("Top triple set →", sim_top.item())
# print("Bottom triple set →", sim_btm.item())


# verbalize _ 단순 이어붙이는 형태?
summary_top = verbalize_triples(top_triples)
summary_btm = verbalize_triples(bottom_triples)

# print("================")
# print("summary_top: ", summary_top)
# print("summary_btm: ", summary_btm)

# LLM으로 선호도 평가 받기
# "A" 또는 "B"
preferred = ask_llm_preference(flattened_sentences, summary_top, summary_btm)

# DPO-style loss 계산
#! token-lvel preference learning loss <- 수정 필요함
tok_pf_loss = preference_learning_loss(sim_top, sim_btm, preferred)


#! Validity-Aware Aggregation
# 1. top triple들의 head id 추출
top_head_ids = [triples[idx][0] for idx in top_indices]  # h from (h, r, t, type)

# 2. 중복 제거
unique_head_ids = set([triples[i][0] for i in top_indices])

# 3. 임베딩 복사본 생성 (업데이트될 것)
update_entity_embedding = EntityEmbeddingUpdater(hidden_dim=flat_entity_emb.size(1)).to(device)
updated_entity_emb = flat_entity_emb.clone()

top_cls_vectors = triple_cls_vectors[top_indices] 

# 4. 각 head_id에 대해 업데이트 적용
#! 일단 지금은 weighted sum에서 weight를 유니폼하게 1로 주는데, 이거를 score 기반으로 weight 할당?
for head_id in unique_head_ids:
    updated_entity_emb[head_id] = update_entity_embedding(
        entity_emb=flat_entity_emb,
        cls_embeddings=top_cls_vectors,     # shape: [K, D]
        triple_indices=top_indices,            # 길이 K
        triples=triples,                       # full triple list
        target_head_id=head_id
    )

update_relation_embedding = RelationEmbeddingUpdater(hidden_dim=rel_emb.size(1)).to(device)

# 전체 노드 임베딩: [num_entity + num_relation, D]
updated_relation_embedding = update_relation_embedding(rel_emb, node_type)




#* Span-level

# # span-level 노드 생성 (connected 엣지 기준, window size 지정, 개인노드-스팬노드 )
# spans = generate_span_candidates(sentences)
# print('spans: ', spans)

# # span pre-filter
# span_labels = 


#* token-to-span composition layer
# Candidate Span 생성
# 문장별로 window size 이내의 모든 (start, end) 생성
keep_mask_list = flat_keep_mask.view(-1).tolist()
candidate_spans = get_candidate_spans(sentences, window_size=5, keep_mask=keep_mask_list)

# 정답 라벨 (ner에 있으면 정답 1, 없으면 0)
span_labels = generate_span_labels(candidate_spans, ner_spans)  # [N_spans]

print('candidate_spans: ', candidate_spans[:20])
print('span_labels: ', span_labels[:10])

# 스팬 임베딩 생성 (내부 토큰은 일단 mean pooling으로 구현)
span_embs = get_span_embeddings(flat_entity_emb, candidate_spans)  # [N_spans, D]
span_labels_tensor = torch.tensor(span_labels).unsqueeze(0).to(device)
# span_ids = list(range(len(candidate_spans)))

# Span Node Prefilter
#! span prefilter Loss
span_scores, span_loss = span_filter(span_embs.unsqueeze(0), labels=span_labels_tensor)
candidate_spans, topk_indices = filter_candidate_spans(span_scores, candidate_spans, topk_ratio=0.4)


# === Node Index 재정의 (bipartite 구조 _ token node -> relation node -> span node 인덱스 순서)
token_ids = list(range(len(entity_tokens)))  # token node ids
relation_ids = list(range(len(token_ids), len(token_ids) + len(relation_types)))
span_ids = list(range(len(token_ids) + len(relation_types), len(token_ids) + len(relation_types) + len(candidate_spans)))

# === span_id -> 포함된 token_id 매핑
# {155: [110, 111]} : 155번 노드(스팬)은 토큰 110, 111로 구성되어있음을 의미
span_id_to_token_indices = {
    span_ids[i]: list(range(start, end + 1))
    for i, (start, end) in enumerate(candidate_spans)
}
print("span_id_to_token_indices:", span_id_to_token_indices)


# === 스팬 노드의 relation 엣지 연결 (내부 토큰이 연결된 rel만)
span_rel_edges = []  # (span_id, rel_id, rel_node_id)
token_rel_edges = extract_token_relation_edges(data)  #! (token_id, rel_id, rel_node_id)

for span_id, token_list in span_id_to_token_indices.items():
    for token_id in token_list:
        for t_id, rel_id, rel_node_id in token_rel_edges:
            if token_id == t_id:
                span_rel_edges.append((span_id, rel_id, rel_node_id))

# 중복 방지 _
# span 내부에 동일한 relation이 여러 번 등장할 경우 중복된 (span_id, rel_id, rel_node_id)가 여러 번 append되기 때문
span_rel_edges = list(set(span_rel_edges))

# === 사전 처리: span_id → rel_id mapping 만들기
# 스팬 노드가 어떤 relation 타입들과 연결될 수 있는지
span_relation_candidates = defaultdict(set)
for span_id, rel_id, rel_node_id in span_rel_edges:
    span_relation_candidates[span_id].add(rel_id)

print("span_relation_candidates: ", span_relation_candidates)

# === candidate triple 생성 (token-span, span-span)
total_node_ids = token_ids + span_ids  # 현재는 relation node는 제외
total_span_ids = span_ids

# span_id_to_token_indices 는 리턴받을 필요 X?
triples = extract_span_token_candidate_triples(
    token_ids=token_ids,
    span_ids=span_ids,
    span_relation_candidates=span_relation_candidates,
    span_id_to_token_indices=span_id_to_token_indices,
    relation_types=relation_types
)

print('span_triples: ', triples)










# # 필터링 된 스팬만 가지고 인덱스 부여 (-> 트리플 조합 생성 시 인덱스 필요)
# span_labels = [span_labels[i] for i in topk_indices]
# # span_ids = [span_ids[i] for i in topk_indices]
# span_embs = span_embs.squeeze(0)[topk_indices]  # [N, D]

# #! 스팬 노드 생성하여 그래프에 추가해주는 과정 추가



# # ===여기부터는 token-level과 동일한 플로우

# token_ids = list(range(len(flat_entity_emb)))  
# # span node ID는 token 노드 ID 다음부터 _ 스팬도 노드로 
# span_ids = list(range(len(token_ids), len(token_ids) + len(candidate_spans)))

# # 스팬을 구성하고 있는 토큰의 인덱스 정보를 리스트 형태로 담음
# span_id_to_token_indices = {
#     span_ids[i]: list(range(start, end + 1))
#     for i, (start, end) in enumerate(candidate_spans)
# }

# print("span_id_to_token_indices: ", span_id_to_token_indices)

# #! 정의해줘야함 _ token level에서 엣지 풀링 정보 갱신하는 것도.. 
# token_rel_edges = extract_token_relation_edges(data)  # 또는 저장된 token-level 엣지

# triples, span_id_to_token_indices = extract_span_token_candidate_triples_topk_pooling(
#     token_ids=token_ids,
#     span_ids=span_ids,
#     candidate_spans=candidate_spans,
#     span_id_to_token_indices=span_id_to_token_indices,
#     token_rel_edges=token_rel_edges,
#     relation_types=relation_types
# )

# token_level_scores = scorer(text_triples)
# triple_cls_vectors = scorer.encode_triples(text_triples).to(device)  # [N, H] _ Triple별 CLS 벡터 추출

# top_triples, top_scores, bottom_triples, bottom_scores, top_indices, bottom_indices = split_triples_by_score(
#     text_triples,
#     token_level_scores,
#     top_k=13,
#     return_indices=True  # top/bottom triple의 인덱스도 반환하도록
# )


# # === 임베딩 통합
# node_embs = torch.cat([flat_entity_emb, span_embs], dim=0)

# # === triple CLS 임베딩 추출
# triple_cls_vectors = scorer.encode_triples(
#     span_triples, node_embs, relation_emb
# ).to(device)

# # === triple scoring
# span_level_scores = scorer.score_triple_vectors(triple_cls_vectors)

# # === top-k/bottom-k 분리
# top_triples, top_scores, bottom_triples, bottom_scores, top_indices, bottom_indices = split_triples_by_score(
#     span_triples,
#     span_level_scores,
#     top_k=13,
#     return_indices=True
# )
