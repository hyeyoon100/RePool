from evaluator import Evaluator
from loss_functions import flatten_gold_relations
import json

# 예측 결과 불러오기
with open("checkpoints/predictions_epoch_1.json") as f:
    all_outs = json.load(f)  # 예: [[...], [...], ...] 또는 [...]

# gold label 준비
with open("data/SciERC/test.json") as f:
    gold_data = [json.loads(line) for line in f]
all_true = [flatten_gold_relations(sample["relations"], sample["sentences"]) for sample in gold_data]

# predictions가 flat 리스트라면 리스트로 한 번 더 감싸기
if all_outs and isinstance(all_outs[0], (str, list, tuple)) and not isinstance(all_outs[0], list):
    all_outs = [all_outs]

# 평가
evaluator = Evaluator(all_true, all_outs)
output_str, f1 = evaluator.evaluate()
print(output_str)