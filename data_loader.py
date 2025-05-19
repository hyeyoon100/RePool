import json
import torch
from torch.utils.data import Dataset, DataLoader

class SciERCDataset(Dataset):
    def __init__(self, path, relation_types=None):
        # 기본 relation types 설정
        self.default_relation_types = [
            "USED-FOR", "FEATURE-OF", "HYPONYM-OF", "PART-OF", 
            "COMPARE", "CONJUNCTION", "EVALUATE-FOR"
        ]
        self.relation_types = relation_types or self.default_relation_types
        self.relation2id = {r: i for i, r in enumerate(self.relation_types)}
        
        # 데이터 로드
        with open(path, 'r') as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 토큰 처리
        tokens = [token for sent in item["sentences"] for token in sent]
        
        # relation_ids 처리
        relation_ids = []
        if "relations" in item and item["relations"]:
            rel_ids = [self.relation2id[r[-1]] for rels in item["relations"] for r in rels]
            relation_ids = list(set(rel_ids))
            
        return {
            "tokens": tokens,
            "sentences": item["sentences"],
            "ner": item.get("ner", []),
            "relations": item.get("relations", []),
            "relation_ids": relation_ids,
            "length": len(tokens)
        }

def collate_fn(batch):
    """배치 데이터를 모델이 처리할 수 있는 형태로 변환"""
    
    # 배치 데이터 수집
    tokens_list = [item['tokens'] for item in batch]
    sentences_list = [item['sentences'] for item in batch]
    ner_list = [item['ner'] for item in batch]
    relation_ids_list = [item['relation_ids'] for item in batch]
    lengths_list = [item['length'] for item in batch]
    
    # lengths를 텐서로 변환
    lengths_tensor = torch.LongTensor(lengths_list)
    
    # 배치 딕셔너리 생성
    batch_dict = {
        "tokens": tokens_list,
        "sentences": sentences_list,
        "lengths": lengths_tensor,
        "relation_ids": relation_ids_list,
        "ner": ner_list,
        "batch_size": len(batch)
    }
    
    return batch_dict

def get_dataloader(file_path, batch_size=32, shuffle=True, num_workers=4):
    """데이터로더 생성 헬퍼 함수"""
    dataset = SciERCDataset(file_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
