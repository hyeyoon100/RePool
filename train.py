import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import json
import os
import logging
from datetime import datetime
from pathlib import Path

from model import RePoolModel
from data_loader import SciERCDataset, collate_fn

# 토크나이저 병렬처리 비활성화
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='train.log',
    filemode='w'
)
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, optimizer, device, epoch):
    """한 에폭 학습을 수행합니다."""
    model.train()
    total_loss = 0
    total_entity_prefilter_loss = 0
    total_token_level_llm_loss = 0
    total_span_level_llm_loss = 0
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 배치 데이터를 device로 이동
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # Forward pass
        outputs = model(batch)
        loss = outputs["final_loss"]

        # 배치별 loss 로그로 남기기
        logger.info(f"Epoch {epoch+1} Batch {batch_idx+1} Loss: {loss.item():.4f}")
        # 필요하다면 세부 loss도 함께
        logger.info(
            f"Entity: {outputs['entity_prefilter_loss'].item():.4f}, "
            f"Token LLM: {outputs['token_level_llm_loss'].item():.4f}, "
            f"Span LLM: {outputs['span_level_llm_loss'].item():.4f}"
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Loss 누적
        total_loss += loss.item()
        total_entity_prefilter_loss += outputs["entity_prefilter_loss"].item()
        total_token_level_llm_loss += outputs["token_level_llm_loss"].item()
        total_span_level_llm_loss += outputs["span_level_llm_loss"].item()
        
        # Progress bar 업데이트
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'entity_loss': f'{outputs["entity_prefilter_loss"].item():.4f}',
            'token_llm_loss': f'{outputs["token_level_llm_loss"].item():.4f}',
            'span_llm_loss': f'{outputs["span_level_llm_loss"].item():.4f}'
        })
    
    # 평균 loss 계산
    num_batches = len(train_loader)
    avg_losses = {
        'total_loss': total_loss / num_batches,
        'entity_prefilter_loss': total_entity_prefilter_loss / num_batches,
        'token_level_llm_loss': total_token_level_llm_loss / num_batches,
        'span_level_llm_loss': total_span_level_llm_loss / num_batches
    }
    
    return avg_losses

def evaluate(model, eval_loader, device):
    """모델을 평가합니다."""
    model.eval()
    total_predictions = []
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            outputs = model(batch)
            total_loss += outputs["final_loss"].item()
            
            # 예측 결과 수집
            for batch_output in outputs["batch_outputs"]:
                predictions = batch_output["predictions"]
                total_predictions.extend(predictions["re"])  # RE 예측만 수집
    
    avg_loss = total_loss / len(eval_loader)
    return avg_loss, total_predictions

def main():
    # 하이퍼파라미터 설정
    config = {
        # 모델 설정
        'bert_model_name': 'bert-base-cased',
        'hidden_size': 768,
        'sample_size': 100,
        
        # 학습 설정
        'learning_rate': 2e-5,
        'batch_size': 8,
        'num_epochs': 2,
        'num_workers': 4,
        
        # 데이터 설정
        'data_dir': 'data/SciERC',  # 데이터 디렉토리
        'train_file': 'train.json',
        'dev_file': 'dev.json',
        'test_file': 'test.json',
        
        # 저장 설정
        'save_dir': 'checkpoints',
        'save_every': 1,  # n 에폭마다 저장
    }
    
    # Device 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 데이터 로딩
    logger.info("Loading datasets...")
    # data_dir = Path(config['data_dir'])
    data_dir = Path("data/SciERC")
    
    train_dataset = SciERCDataset(data_dir / config['train_file'])
    eval_dataset = SciERCDataset(data_dir / config['dev_file'])
    test_dataset = SciERCDataset(data_dir / config['test_file'])
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Dev dataset size: {len(eval_dataset)}")
    logger.info(f"Test dataset size: {len(test_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        collate_fn=collate_fn
    )
    
    # 모델 초기화
    logger.info("Initializing model...")
    model = RePoolModel(
        bert_model_name=config['bert_model_name'],
        hidden_size=config['hidden_size'],
        sample_size=config['sample_size'],
        device=device,
        relation_types=train_dataset.relation_types
    ).to(device)
    
    # 옵티마이저 설정
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # 체크포인트 디렉토리 생성
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    # 학습 루프
    logger.info("Starting training...")
    best_eval_loss = float('inf')
    
    for epoch in range(config['num_epochs']):
        # 학습
        train_losses = train_epoch(model, train_loader, optimizer, device, epoch)
        logger.info(f"Epoch {epoch+1} training losses: {train_losses}")
        
        # 평가
        eval_loss, predictions = evaluate(model, eval_loader, device)
        logger.info(f"Epoch {epoch+1} evaluation loss: {eval_loss:.4f}")
        
        # 체크포인트 저장
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            checkpoint_path = save_dir / f"best_model_epoch_{epoch+1}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'eval_loss': eval_loss,
                'config': config
            }, checkpoint_path)
            logger.info(f"Saved best model checkpoint to {checkpoint_path}")
        
        # 예측 결과 저장
        predictions_path = save_dir / f"predictions_epoch_{epoch+1}.json"
        with open(predictions_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"Saved predictions to {predictions_path}")

if __name__ == "__main__":
    main()