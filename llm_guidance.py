import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple 
from openai import OpenAI
import os
from dotenv import load_dotenv
import random

# .env 파일 로드
load_dotenv()

# 환경 변수 읽기
api_key = os.getenv('openai.api_key')
client = OpenAI(api_key=api_key)

#* [CLS_triple_i]를 입력으로 받아 트리플셋 전체를 대표하는 [CLS_graph] 벡터를 출력함
# class TripleSetEncoder(nn.Module):
#     def __init__(self, model_name: str = "bert-base-cased"):
#         super().__init__()
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.encoder = AutoModel.from_pretrained(model_name)

#     def encode_triple_set(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
#         """
#         Converts a list of triples into a single [CLS] embedding.
#         Input format: "h r t [SEP] h r t [SEP] ..."
#         """
#         triple_strings = [" ".join([h, r, t]) for h, r, t in triples]
#         joined_text = " [SEP] ".join(triple_strings)

#         encoded = self.tokenizer(
#             joined_text,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             add_special_tokens=True  # adds [CLS] and final [SEP]
#         )

#         input_ids = encoded['input_ids'].to(self.encoder.device)
#         attention_mask = encoded['attention_mask'].to(self.encoder.device)

#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
#         cls_embedding = outputs.last_hidden_state[:, 0]  # [B, H], first token is [CLS]
#         return cls_embedding.squeeze(0)  # [H]


class TripleSetEncoder(nn.Module):
    def __init__(self, hidden_size=768, n_layers=2, n_heads=8, dropout=0.1):
        super().__init__()

        # Learnable CLS token [1, 1, H]
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))

        # Transformer Encoder Layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True  # input: [B, T, H]
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, cls_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cls_embeddings: Tensor of shape [N, H] (N = number of triples)
        Returns:
            Tensor of shape [H], the triple-set representation
        """
        N, H = cls_embeddings.size()
        device = cls_embeddings.device

        # Add CLS token in front of the sequence
        # [1, 1, H] → [1, N+1, H]
        cls_token_expanded = self.cls_token.expand(1, -1, -1).to(device) # [1, 1, H]

        # [CLS] + triple_1 + triple_2 + ... 형태로 입력 구성
        triple_sequence = torch.cat([cls_token_expanded, cls_embeddings.unsqueeze(0)], dim=1)  # [1, N+1, H]

        # Pass through Transformer Encoder
        encoded = self.encoder(triple_sequence)  # [1, N+1, H]

        # Return only the [CLS] token
        return encoded[:, 0, :].squeeze(0)  # [H]





def verbalize_triple(h: str, r: str, t: str, r_type: str) -> str:
    """
    Convert a single (head, relation, tail) triple to a natural language sentence.
    """
    relation_map = {
        'USED-FOR': 'is used for',
        'PART-OF': 'is part of',
        'FEATURE-OF': 'has the feature of',
        'COMPARE': 'is compared to',
        'EVALUATE-FOR': 'is evaluated for',
        'HYPONYM-OF': 'is a kind of',
        'CONJUNCTION': 'and',
    }
    if r_type == "semantic":
        if r in relation_map:
            return f"{h} {relation_map[r]} {t}"
        else:
            return f"{h} {r} {t}."
    elif r_type == "compound":
        return f"{h} is compound with {t}"
    else:
        return f"{h} {r} {t}."

def verbalize_token_triple(h: str, edge_type: str, t: str) -> str:
    edge_type_map = {
        'can_form_subject_of': 'can be the subject of',
        'can_form_object_of': 'can be the object of',
        'can_form_compound_with': 'can form a compound with'
    }
    
    if edge_type in edge_type_map:
        return f"{h} {edge_type_map[edge_type]} {t}"
    else:
        return f"{h} {edge_type} {t}"



def verbalize_triples(triples: List[Tuple[str, str, str, str]], level: str) -> str:
    """
    Convert a list of (head, relation, tail) triples to a single verbalized paragraph.
    """
    print('triples in : ', level, triples[:3])
    
    if level == 'token':
        sentences = [verbalize_token_triple(h, r, t) for (h, r, t) in triples] #  
    elif level == 'span': # span
        sentences = [verbalize_triple(h, r, t, r_type) for (h, r, t, r_type) in triples]

    return " ".join(sentences)


def preference_learning_loss(sim_a, sim_b, preferred: str, input_text_emb):
    # 목표: preferred (A)의 점수를 높이고, non-preferred (B)의 점수를 낮춤
    """
    Compute preference learning loss:
    L = -log(σ(T·sim_pos - T·sim_neg))
    
    Args:
        sim_a: similarity score for summary A
        sim_b: similarity score for summary B
        preferred: 'A' or 'B' indicating which summary is preferred
    
    """
    if preferred == 'A':
        sim_pos = sim_a
        sim_neg = sim_b
    else:  # preferred == 'B'
        sim_pos = sim_b
        sim_neg = sim_a

    # Input text embedding과 similarity score의 곱
    weighted_sim_pos = sim_pos * torch.mean(input_text_emb)  # 스칼라 곱셈
    weighted_sim_neg = sim_neg * torch.mean(input_text_emb)  # 스칼라 곱셈
    
    # 차이 계산
    diff = weighted_sim_pos - weighted_sim_neg

    #* llm loss 계속 같은 값이 나옴. 체크하기
    
    # Loss 계산 (mean 제거)
    loss = -torch.log(torch.sigmoid(diff + 1e-8))
    
    return loss

def ask_llm_preference(input_text: List[List[str]], summary_a: str, summary_b: str) -> str:
    """
    Prompt an LLM to select which summary (A or B) better aligns with the input text.
    
    Args:
        input_text: List[List[str]] - 문장별로 나뉜 토큰 리스트
        summary_a: str - 첫 번째 요약
        summary_b: str - 두 번째 요약
    """
    # 토큰 리스트를 문장으로 변환
    flattened_text = []
    for sentence_tokens in input_text:
        sentence = " ".join(sentence_tokens)
        flattened_text.append(sentence)
    
    # 문장들을 하나의 텍스트로 합침
    final_text = " ".join(flattened_text)
    print("final text for LLM: ", final_text)

    # ======= 실제 LLM 프롬프트 호출 코드 (비용 문제로 주석처리) =======
    prompt = f"""
    Given the following input text:
    {final_text}
    
    Which of the following two summaries better matches the input text?
    A: {summary_a}
    B: {summary_b}
    
    Answer with 'A' or 'B' only.
    """
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1,
    )
    answer = response.choices[0].message.content.strip()
    print('======llm answer: ', answer)
    # ===========================================================

    # 임시로 랜덤 선택 (실제 LLM 호출 대신)
    # answer = random.choice(["A", "B"])
    # print('random answer: ', answer)

    return answer