"""Evaluate an LFEvent checkpoint with structural-entropy-guided clustering."""
# 独立评估入口：指定 checkpoint、数据集和 S1/S2，输出指标及可复核的聚类中间结果。
import argparse
import json
from pathlib import Path

from lfevent.data import FourSourceInputs, audit_setting
from lfevent.engine import REPO, evaluate, load_controlled_model, original_encodings

import numpy as np
import torch
from transformers import AutoTokenizer


def main(args):
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    frames, _ = audit_setting(args.data_root.resolve(), args.dataset, args.setting)
    # 直接使用指定分割的原始行顺序；source_row 用于把预测结果对应回 CSV 行号。
    frame = frames[args.split].copy()
    frame.insert(0, "source_row", np.arange(len(frame)))
    # 同时恢复训练头和配套冻结编码器；加载器会检查编码器文件哈希。
    model = load_controlled_model(args.checkpoint.resolve(), args.device)
    tokenizer = AutoTokenizer.from_pretrained(model.config.encoder, local_files_only=True)
    inputs = FourSourceInputs(frame, args.dataset, args, tokenizer, {})
    # 推理只编码原文+时间，增强文本不会传入聚类特征计算。
    text = original_encodings(model, inputs, args.device)
    # 保持与训练期间验证/测试相同的批次大小和图构建方式。
    args.batch_size, args.graph = 64, "reference"
    output = args.output or REPO / "runs/evaluation" / f"{args.dataset}_{args.setting}_{args.split}"
    metrics = evaluate(model, text, inputs.images, frame, output.resolve(), args)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", choices=["NED", "CrisisMMD"], required=True)
    parser.add_argument("--setting", choices=["S1", "S2"], required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--data-root", type=Path, default=REPO / "data/Event_data")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    main(parser.parse_args())
