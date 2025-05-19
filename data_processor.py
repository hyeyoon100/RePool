import json

def build_batch(samples, batch_size=32, relation_types=None):
    """
    samples: List[dict]  # JSON에서 읽은 샘플 리스트
    batch_size: int  # 배치 크기
    relation_types: Optional[List[str]]  # 전체 relation type 리스트 (미리 집계 가능)
    """
    # 기본 relation types 설정
    default_relation_types = [
        "USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", 
            "COMPARE", "CONJUNCTION", "EVALUATE-FOR"
    ]
    relation_types = relation_types or default_relation_types
    relation2id = {r: i for i, r in enumerate(relation_types)}

    sentences_list = []
    tokens_list = []
    lengths_list = []
    relation_ids_list = []
    ner_list = []

    for sample in samples:
        tokens = [tok for sent in sample["sentences"] for tok in sent]
        tokens_list.append(tokens)
        sentences_list.append(sample["sentences"])
        lengths_list.append(len(tokens))

        # relation_ids 처리 - relations 키가 없거나 비어있을 경우 빈 리스트 사용
        if "relations" in sample and sample["relations"]:
            rel_ids = [relation2id[r[-1]] for rels in sample["relations"] for r in rels]
            relation_ids_list.append(list(set(rel_ids)))
        else:
            relation_ids_list.append([])

        # ner 처리 - ner 키가 없을 경우 빈 리스트 사용
        ner_list.append(sample.get("ner", []))

    # 배치 크기만큼 데이터 자르기
    if len(samples) > batch_size:
        tokens_list = tokens_list[:batch_size]
        sentences_list = sentences_list[:batch_size]
        lengths_list = lengths_list[:batch_size]
        relation_ids_list = relation_ids_list[:batch_size]
        ner_list = ner_list[:batch_size]

    batch = {
        "tokens": tokens_list,
        "sentences": sentences_list,
        "lengths": lengths_list,
        "relation_ids": relation_ids_list,
        "relation_types": relation_types,
        "ner": ner_list,
        "batch_size": len(tokens_list)
    }
    return batch
