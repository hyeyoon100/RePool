


#* EntityNode 에 labeling _ 해당 토큰이 NER(span)에 포함되는지 binary label 부여
def generate_token_labels(sentences, ner_spans):
    """
    sentences: List[List[str]] — 한 문서의 문장별 토큰 리스트
    ner_spans: List[List[Tuple[int, int, str]]] — 각 문장에 대한 NER span 정보

    return: List[List[int]] — 문장별 token 단위 binary label (1 = part of entity, 0 = not)
    """
    token_labels = []
    offset = 0
    for sent_tokens in sentences:
        label = [0] * len(sent_tokens)
        token_labels.append(label)

    for span_list in ner_spans:
        for start, end, _ in span_list:
            # global index → 문장 index + local index 변환
            cur = 0
            for sent_idx, sent in enumerate(sentences):
                sent_len = len(sent)
                if start >= cur and end < cur + sent_len:
                    for i in range(start - cur, end - cur + 1):
                        token_labels[sent_idx][i] = 1
                    break
                cur += sent_len
    return token_labels
