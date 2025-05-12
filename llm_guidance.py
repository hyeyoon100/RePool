import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import List, Tuple



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


def verbalize_triples(triples: List[Tuple[str, str, str]]) -> str:
    """
    Convert triples into a textual summary for LLM input.
    Example: ("is", "used-for", "evaluation") → "is used for evaluation."
    """
    return " ".join([f"{h} {r.replace('_', ' ')} {t}." for h, r, t in triples])


def preference_learning_loss(sim_a, sim_b, preferred: str, tau=0.1):
    """
    Compute DPO-style preference loss between two cosine similarities.
    """
    # cosine similarity 점수 자체가 1 또는 0에 가까워지도록 학습
    sims = torch.stack([sim_a, sim_b]) / tau
    label = torch.tensor(0 if preferred == "A" else 1, device=sims.device)
    return nn.functional.cross_entropy(sims.unsqueeze(0), label.unsqueeze(0))


# Dummy version (no real LLM call) — for actual use, plug in OpenAI or Huggingface
import random
def ask_llm_preference(input_text: str, summary_a: str, summary_b: str) -> str:
    """
    Prompt an LLM to select which summary (A or B) better aligns with the input text.
    For now, returns random choice for testing.
    """
    # TODO: Replace with actual LLM call
    prompt = f"""
        You are given an input text and two generated summaries.

        Input Text:
        "{input_text}"

        Summary A:
        "{summary_a}"

        Summary B:
        "{summary_b}"

        Question:
        Which summary (A or B) better captures the information in the Input Text? Please answer "A" or "B" only.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0  # deterministic 답변을 유도
    )

    answer = response['choices'][0]['message']['content'].strip()
    # return answer

    # A: prefer. B: disprefer
    #! 자동화된 방향 선택 _ bottom을 prefer로 선택하면 순서 바꿔줌
    # if preferred == "A":
    #     loss = dpo_loss(sim_top, sim_bot)
    # else:
    #     loss = dpo_loss(sim_bot, sim_top)

    return random.choice(["A", "B"])


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