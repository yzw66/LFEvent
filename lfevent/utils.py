"""Checkpoint, batch and reproducibility helpers."""
# 通用辅助函数：文件安全写入、权重哈希、分批规则和训练梯度检查。
import hashlib
import json
import math
from pathlib import Path
import torch

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 先写临时文件，再替换目标文件，避免中途退出后留下不完整的 JSON。
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)



def sha256(path):
    # 对整个文件求哈希，用于确认数据、代码或冻结编码器文件是否一致。
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()



def save_checkpoint(path, state):
    # checkpoint 也采用临时文件替换的写法；这里不改变传入的权重或优化器状态。
    temporary = path.with_name(path.name + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)



def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))



def tensor_digest(state):
    # 按名称排序后对张量内容、类型和形状求哈希，比较模型权重而非文件序列化细节。
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()



def batches(count, size, order=None):
    if count < 2 or size < 2:
        raise ValueError("At least two samples and batch_size>=2 are required")
    order = list(range(count)) if order is None else order
    result = [order[i:i + size] for i in range(0, count, size)]
    # 不丢弃尾样本；若最后仅剩 1 条，将它并入前一批，避免 BatchNorm 和对比损失的单样本问题。
    # 例如 129 条、size=64 时，实际分为 64+65，而不是 64+64+1。
    if len(result[-1]) == 1:
        result[-2].extend(result.pop())
    return result



def head_state(model):
    # 仅提取非文本编码器部分的权重和 BN 状态；冻结编码器由单独文件保存。
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            if not k.startswith("text_encoder.")}



def load_heads(model, state):
    # 允许缺失冻结编码器的键，但训练头缺失或出现多余键时立即报错。
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(not k.startswith("text_encoder.") for k in result.missing_keys):
        raise ValueError(f"Checkpoint structure differs: {result}")



def gradient_evidence(model):
    # 检查首个训练批次中注意力、门控、融合层、投影器和预测器是否都获得非零有限梯度。
    groups = ["augmented_fusion.attention", "augmented_fusion.fusion", "projector", "predictor"]
    groups += [f"augmented_fusion.gates.{i}" for i in range(4)]
    groups += [f"cmap.{p}" for p in ["q", "k", "v", "o"]]
    result = {}
    for group in groups:
        squared = sum(float(p.grad.detach().square().sum()) for name, p in model.named_parameters()
                      if name.startswith(group + ".") and p.grad is not None)
        result[group] = math.sqrt(squared)
    if any(not math.isfinite(v) or v == 0 for v in result.values()):
        raise RuntimeError(f"A required module has zero/nonfinite gradient: {result}")
    if any(p.grad is not None for p in model.text_encoder.parameters()):
        # MiniLM 是冻结的，即使训练时开启 Dropout，其参数也不应产生梯度。
        raise RuntimeError("Frozen SBERT unexpectedly received gradients")
    return result
