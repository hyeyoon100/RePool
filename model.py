

#* EntityNode Embedding




#* Relation Node Embedding
# One-hot 방식
onehot_rel_emb = RelationEmbedding(num_relations=20, embedding_dim=128, method="onehot")

# Pretrained LM 방식
rel_names = ["Work_For", "Located_In", "OrgBased_In", "Live_In", ...]
lm_rel_emb = RelationEmbedding(num_relations=len(rel_names), embedding_dim=768, method="pretrained_lm", relation_names=rel_names)


#* EntityNode Pre-filter
# entity_rep: [B, L, D]
# entity_mask: [B, L] (1 for valid tokens, 0 for pad)
# ner_labels: [B, L] (1 for tokens inside entity spans, 0 otherwise)

filter_layer = EntityNodeFilter(hidden_size=768)
score, loss = filter_layer(token_embeddings=entity_rep, labels=ner_labels, mask=entity_mask)

# score: [B, L] — sigmoid score (0~1), higher means more likely part of entity
# loss: BCE loss over valid tokens
