from typing import List
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer


#* Entity Node 를 위한 토큰 임베딩
class EntityTokenRep(nn.Module):
    def __init__(self, model_name: str = "bert-base-cased", fine_tune: bool = True, hidden_size: int = 768):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.fine_tune = fine_tune
        self.bert_hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Linear(self.bert_hidden_size, hidden_size) if hidden_size != self.bert_hidden_size else None

    def forward(self, tokens: List[List[str]], lengths: torch.Tensor):
        token_embeddings, mask = self.compute_word_embedding_split_by_sentence(tokens)

        if self.projection:
            token_embeddings = self.projection(token_embeddings)

        return {"embeddings": token_embeddings, "mask": mask}
        # # Tokenize each sentence separately with is_split_into_words=True
        # batch_encodings = self.tokenizer(
        #     tokens,
        #     is_split_into_words=True,
        #     return_tensors="pt",
        #     padding=True,
        #     truncation=False, # True? False?
        #     return_attention_mask=True
        # )

        # input_ids = batch_encodings["input_ids"].to(self.encoder.device)
        # attention_mask = batch_encodings["attention_mask"].to(self.encoder.device)

        # with torch.set_grad_enabled(self.fine_tune):
        #     outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        #     hidden_states = outputs.last_hidden_state  # [B, T, H]

        # batch_embeddings = []
        # for i, input_tokens in enumerate(tokens):
        #     word_ids = batch_encodings.word_ids(batch_index=i)
        #     seen = set()
        #     token_vectors = []

        #     for j, word_id in enumerate(word_ids):
        #         if word_id is None or word_id in seen:
        #             continue
        #         seen.add(word_id)
        #         token_vectors.append(hidden_states[i, j])  # first subtoken for each token

        #     token_vectors = torch.stack(token_vectors)  # [L_i, H]
        #     batch_embeddings.append(token_vectors)

        # token_embeddings = pad_sequence(batch_embeddings, batch_first=True)  # [B, L, H]
        # if self.projection:
        #     token_embeddings = self.projection(token_embeddings)

        # # Generate mask
        # B = len(lengths)
        # max_length = lengths.max()
        # mask = (torch.arange(max_length).unsqueeze(0).repeat(B, 1).to(lengths.device) < lengths.unsqueeze(1)).long()

        # return {"embeddings": token_embeddings, "mask": mask}
    
    #* 문장 단위로 나누어서 처리
    def compute_word_embedding_split_by_sentence(self, tokens: List[List[str]]) -> torch.Tensor:
        all_embeddings = []

        for sent in tokens:
            encoding = self.tokenizer(
                [sent],
                is_split_into_words=True,
                return_tensors="pt",
                padding=True,
                truncation=False,
                return_attention_mask=True
            )

            input_ids = encoding["input_ids"].to(self.encoder.device)
            attention_mask = encoding["attention_mask"].to(self.encoder.device)

            with torch.set_grad_enabled(self.fine_tune):
                outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
                hidden_states = outputs.last_hidden_state[0]  # [T, H]

            word_ids = encoding.word_ids(batch_index=0)
            seen = set()
            token_vectors = []

            for j, word_id in enumerate(word_ids):
                if word_id is None or word_id in seen:
                    continue
                seen.add(word_id)
                token_vectors.append(hidden_states[j])  # 첫 subtoken만

            if token_vectors:
                sent_embedding = torch.stack(token_vectors)  # [L_i, H]
                all_embeddings.append(sent_embedding)

        # [B, L, H]
        token_embeddings = pad_sequence(all_embeddings, batch_first=True)

        # 실제 길이에 맞는 mask 생성
        lengths = [emb.size(0) for emb in all_embeddings]
        max_length = max(lengths)
        mask = (torch.arange(max_length).unsqueeze(0).repeat(len(lengths), 1).to(token_embeddings.device)
                < torch.tensor(lengths).unsqueeze(1).to(token_embeddings.device)).long()

        return token_embeddings, mask



class RelationTokenRep(nn.Module):
    def __init__(self, num_relations, embedding_dim, method="onehot", lm_model_name="bert-base-uncased", relation_names=None):
        super().__init__()
        self.method = method
        self.embedding_dim = embedding_dim

        if method == "onehot":
            self.embedding = nn.Embedding(num_relations, embedding_dim)
        elif method == "pretrained_lm":
            assert relation_names is not None, "relation_names list is required for pretrained_lm method"
            self.tokenizer = AutoTokenizer.from_pretrained(lm_model_name)
            self.encoder = AutoModel.from_pretrained(lm_model_name)
            with torch.no_grad():
                encoded = self.tokenizer(relation_names, return_tensors='pt', padding=True, truncation=True)
                outputs = self.encoder(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS]
            self.embedding = nn.Parameter(cls_embeddings, requires_grad=True)  # learnable after init
        else:
            raise ValueError(f"Unsupported relation embedding method: {method}")

    def forward(self, relation_ids=None):
        if self.method == "onehot":
            return self.embedding(relation_ids)  # [batch_size, embedding_dim]
        elif self.method == "pretrained_lm":
            return self.embedding  # [num_relations, embedding_dim]
