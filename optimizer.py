import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

def get_optimizer(model, lr=1e-5, weight_decay=0.01, warmup_steps=1000):
    """
    모델의 optimizer와 learning rate scheduler를 반환
    
    Args:
        model: 학습할 모델
        lr: 학습률
        weight_decay: weight decay 계수
        warmup_steps: warmup 스텝 수
    
    Returns:
        optimizer: AdamW optimizer
        scheduler: LR scheduler with warmup
    """
    # 1. Optimizer 설정
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)

    # 2. Learning rate scheduler with warmup
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler 