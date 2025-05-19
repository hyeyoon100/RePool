import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class CustomBertEmbeddings(nn.Module):
    def __init__(self, config, type_vocab_size=4):
        super().__init__()
        # self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Position embedding 제거
        # self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)

        # Custom segmentation embedding (e.g. [HEAD], [REL], [TAIL])
        self.segment_embeddings = nn.Embedding(type_vocab_size, config.hidden_size)

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_embs, seg_ids=None):
        # input_embeds = self.token_embeddings(input_ids) # token embeddings
        # seq_length = input_embs.size(1)

        if seg_ids is None:
            seg_ids = torch.zeros_like(input_embs.size(0), input_embs.size(1), dtype=torch.long, device=input_embs.device) 
            # seg_ids = torch.zeros_like(input_ids)  # default to segment 0
        seg_embeds = self.segment_embeddings(seg_ids).to(input_embs.device)

        embeddings = input_embs + seg_embeds # No position embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings



class OneLayerBertModel(nn.Module):
    def __init__(self, pretrained_model_name='bert-base-cased', seg_vocab_size=4):
        # seg_vocab_size=4 : CLS, HEAD, REL, TAIL _ 커스텀하기에 CLS에도 id=0 부여해줘야함
        super().__init__()
        config = BertConfig.from_pretrained(pretrained_model_name)
        config.num_hidden_layers = 1

        # self.bert = BertModel(config)
        # full_model = BertModel.from_pretrained(pretrained_model_name)
        # # Create empty 1-layer model _ custom embedding으로 대체
        # self.config = config
        # self.bert.embeddings = CustomBertEmbeddings(config, seg_vocab_size=seg_vocab_size)
        # self.bert.embeddings.token_embeddings.load_state_dict(full_model.embeddings.token_embeddings.state_dict())
        # self.bert.encoder.layer[0].load_state_dict(full_model.encoder.layer[0].state_dict())
        # self.bert.pooler.load_state_dict(full_model.pooler.state_dict())

        # Custom segmentation embedding (no position embedding)
        self.embeddings = CustomBertEmbeddings(config, type_vocab_size=seg_vocab_size)

        # BERT 전체 모델에서 0번째 레이어, pooler만 복사해옴
        full_model = BertModel.from_pretrained(pretrained_model_name)
        self.encoder_layer = full_model.encoder.layer[0]
        # self.pooler = full_model.pooler


        self.hidden_size = config.hidden_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def forward(self, input_embs, seg_ids=None, attention_mask=None):
        """
        input_embs: [batch_size, seq_len, hidden_dim] — 이미 token embedding이 완료된 상태
        seg_ids:    [batch_size, seq_len]             — 각 토큰의 segmentation ID
        attention_mask: [batch_size, seq_len]         — 1 (keep) / 0 (pad)
        """
        embeddings = self.embeddings(input_embs, seg_ids=seg_ids) # segment embedding 더해줌

        if attention_mask is None:
            attention_mask = torch.ones(embeddings.size()[:2], device=embeddings.device)
        extended_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0 # attention_mask 수동 확장 처리?
        
        # 1개 encoder layer만 통과
        encoder_output = self.encoder_layer(embeddings, attention_mask=extended_mask)[0]

        # pooler 출력
        # pooled_output = self.pooler(encoder_output)
        
        
        return encoder_output[:, 0, :]  # → [CLS] 위치의 contextualized 벡터