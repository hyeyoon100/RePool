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

class TripleSetEncoder(nn.Module):
    def __init__(self, model_name: str = "bert-base-cased"):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)

    def encode_triple_set(self, triples: List[Tuple[str, str, str]]) -> torch.Tensor:
        """
        Converts a list of triples into a single [CLS] embedding.
        Input format: "h r t [SEP] h r t [SEP] ..."
        """
        triple_strings = [" ".join([h, r, t]) for h, r, t in triples]
        joined_text = " [SEP] ".join(triple_strings)

        encoded = self.tokenizer(
            joined_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True  # adds [CLS] and final [SEP]
        )

        input_ids = encoded['input_ids'].to(self.encoder.device)
        attention_mask = encoded['attention_mask'].to(self.encoder.device)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [B, H], first token is [CLS]
        return cls_embedding.squeeze(0)  # [H]


def verbalize_triple(h: str, r: str, t: str) -> str:
    """
    Convert a single (head, relation, tail) triple to a natural language sentence.
    """
    relation_map = {
        'used-for': 'is used for',
        'part-of': 'is part of',
        'feature-of': 'has the feature of',
        'compare': 'is compared to',
        'evaluate-for': 'is evaluated for',
        'hyponym-of': 'is a kind of',
        'conjunction': 'and',
    }

    if r in relation_map:
        if r == 'conjunction':
            return f"{h} and {t} are mentioned together."
        return f"{h} {relation_map[r]} {t}."
    else:
        return f"{h} {r} {t}."  # fallback: 관계 그대로 사용



def verbalize_triples(triples: List[Tuple[str, str, str]]) -> str:
    """
    Convert a list of (head, relation, tail) triples to a single verbalized paragraph.
    """
    sentences = [verbalize_triple(h, r, t) for (h, r, t) in triples]
    return " ".join(sentences)


def preference_learning_loss(sim_a, sim_b, preferred: str, tau=0.1):
    """
    Compute DPO-style preference loss between two cosine similarities.
    """
    # cosine similarity 점수 자체가 1 또는 0에 가까워지도록 학습함
    sims = torch.stack([sim_a, sim_b]) / tau
    label = torch.tensor(0 if preferred == "A" else 1, device=sims.device)
    return nn.functional.cross_entropy(sims.unsqueeze(0), label.unsqueeze(0))


# Dummy version (no real LLM call) — for actual use, plug in OpenAI or Huggingface

def ask_llm_preference(input_text: str, summary_a: str, summary_b: str) -> str:
    """
    Prompt an LLM to select which summary (A or B) better aligns with the input text.
    """
    print("==== in llm ====")
    print('llm input_text: ', input_text)
    print('sum_a: ', summary_a)
    print('sum_b: ', summary_b)

    prompt = f"""
        You are given an input text and two generated summaries.

        Input Text:
        "{input_text}"

        Summary A:
        "{summary_a}"

        Summary B:
        "{summary_b}"

        Question:
        Which summary better captures the information in the input text?

        Answer with only one letter: A or B.
        No explanation. Only output 'A' or 'B'.
    """
    #    Question:
    #     Which summary (A or B) better captures the information in the Input Text? Please answer "A" or "B" only.
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "user", "content": prompt}
    ],
    temperature=0  # deterministic 답변을 유도
    )

    answer = response.choices[0].message.content.strip()
    print('llm answer: ', answer)

    # answer = random.choice(["A", "B"])
    # print('random answer: ', answer)

    return answer

    # A: prefer. B: disprefer
    #! 자동화된 방향 선택 _ bottom을 prefer로 선택하면 순서 바꿔줌
    # if preferred == "A":
    #     loss = dpo_loss(sim_top, sim_bot)
    # else:
    #     loss = dpo_loss(sim_bot, sim_top)

    # return random.choice(["A", "B"])


#! 저번에 짜둔 것. 현 모델에 맞추어 개선하기
# openai.api_key = "YOUR_OPENAI_API_KEY"

# def ask_llm_preference(input_text, summary_a, summary_b):
#     prompt = f"""
# You are given an input text and two generated summaries.

# Input Text:
# "{input_text}"

# Summary A:
# "{summary_a}"

# Summary B:
# "{summary_b}"

# Question:
# Which summary (A or B) better captures the information in the Input Text? Please answer "A" or "B" only.
# """
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0  # deterministic 답변을 유도
#     )

#     answer = response['choices'][0]['message']['content'].strip()
    # return answer