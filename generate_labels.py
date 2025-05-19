#* EntityNode 에 labeling _ 해당 토큰이 NER(span)에 포함되는지 binary label 부여
def generate_token_labels(sentences, ner_spans):
    """
    Args:
        sentences: List[List[str]] - 문장별 토큰 리스트
        ner_spans: List[List[Tuple[int, int, str]]] - 문장별 NER span 정보
    Returns:
        labels: List[int] - 전체 토큰에 대한 binary label (1: entity token, 0: non-entity token)
        total_len: int - 전체 토큰 수
    """
    # 토큰 flatten
    flat_tokens = [tok for sent in sentences for tok in sent]
    total_len = len(flat_tokens)
    
    # 각 문장 길이 누적합 계산
    sent_lens = [len(s) for s in sentences]
    offsets = [0]
    for l in sent_lens[:-1]:
        offsets.append(offsets[-1] + l)
        
    # 0으로 초기화
    label = [0] * total_len
    
    # span labeling
    for sent_idx, span_list in enumerate(ner_spans):
        offset = offsets[sent_idx]  # 현재 문장의 시작 오프셋
        
        for start, end, _ in span_list:
            # 현재 문장 내에서의 start, end를 전체 문서에서의 인덱스로 변환
            global_start = offset + start
            global_end = offset + end
            
            # 범위 체크 후 라벨링
            if global_start < total_len:
                for i in range(global_start, min(global_end + 1, total_len)):
                    label[i] = 1
    
    return label, total_len


#* SpanEntityNode에 labeling _ 해당 스팬이 NER인지 binary label 부여
def generate_span_labels(candidate_spans, ner_spans):
    """
    Args:
        candidate_spans: List[Tuple[int, int]] — global start, end
        ner_spans: List[List[Tuple[int, int, str]]] — sentence-wise span annotations (global indices)
    Returns:
        labels: List[int] — 1 if exact match with gold span, else 0
    """
    gold_spans = set((start, end) for sent_spans in ner_spans for start, end, _ in sent_spans)
    labels = [1 if (start, end) in gold_spans else 0 for (start, end) in candidate_spans]
    return labels
