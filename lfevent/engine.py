"""LFEvent training, validation, checkpoint selection and evaluation."""
# 核心流程：读现有分割 -> 四路文本在线训练 -> 每轮验证 -> 选最佳 checkpoint -> 测试一次。
import argparse
from dataclasses import asdict
import hashlib
import json
import logging
import math
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import cut_tree, linkage
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score
import torch
from transformers import AutoTokenizer

from .model import ControlledLFEvent, LFEventConfig, implementation_choices
from .data import FourSourceInputs, audit_setting
from .clustering import cosine_knn_graph, reference_graph, structural_entropy_curve
from .utils import batches, gradient_evidence, head_state, load_heads, read, save_checkpoint, sha256, write_json

REPO = Path(__file__).resolve().parent.parent
# 当前 S1/S2 共用的训练配置；其他结构和损失参数见 LFEventConfig。
RECIPE = {"learning_rate": .01, "gradient_clip": 1.}
# 这些代码文件的哈希写入训练协议，断点续训时会检查协议是否一致。
SOURCES = ["train.py", "evaluate.py", "lfevent/__init__.py", "lfevent/model.py", "lfevent/initialization.py",
           "lfevent/data.py", "lfevent/clustering.py", "lfevent/utils.py", "lfevent/engine.py",
           "prepare_data.py", "extract_image_features.py", "download_models.py", "requirements.txt"]


@torch.no_grad()
def original_encodings(model, inputs, device):
    # 仅缓存评估阶段的原始文本句向量；eval 下冻结编码器输出确定，不缓存训练 Dropout 的输出。
    model.eval()
    return torch.cat([model.text_encoder(**inputs.tokens(indices, 0, device)).cpu()
                      for indices in batches(len(inputs.images), 64)])



def load_controlled_model(checkpoint_path, device="cpu"):
    path = Path(checkpoint_path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("model_class") != "ControlledLFEvent":
        raise ValueError("Expected a ControlledLFEvent checkpoint")
    # 训练头与冻结编码器分开保存；相对路径允许读取 selected.pth 或每轮 checkpoint。
    frozen = path.parent / state["frozen_encoder_file"]
    if sha256(frozen) != state["frozen_encoder_sha256"]:
        raise ValueError("Frozen encoder checkpoint hash differs")
    model = ControlledLFEvent(LFEventConfig(**state["model_config"]))
    model.text_encoder.load_state_dict(torch.load(frozen, map_location="cpu", weights_only=True))
    load_heads(model, state["state_dict"])
    return model.to(device).eval()



@torch.inference_mode()
def evaluate(model, text, images, frame, destination, args):
    model.eval()
    # 保留输入行顺序和 batch=64 的分批规则；输出转 float64 供图构建和 Ward 使用。
    features = torch.cat([model.cluster_features(text[idx].to(args.device), images[idx].to(args.device)).cpu()
                          for idx in batches(len(frame), args.batch_size)]).numpy().astype(np.float64)
    if not np.isfinite(features).all():
        raise FloatingPointError("Nonfinite inference features")
    # 当前入口使用 reference：Pearson 1NN 加共同 hashtag/@mention 的边。
    # 兼容的 paper 分支使用余弦 3NN，当前训练和评估入口没有启用它。
    edges, weights = (cosine_knn_graph(features, 3) if args.graph == "paper"
                      else reference_graph(features, frame.text, 1))
    tree = linkage(features, method="ward")
    curve = structural_entropy_curve(tree, edges, weights, len(frame))
    # curve[K-1] 是 K 个簇时的结构熵；从 N 到 1 搜索，平分时优先取较大的 K。
    count = len(frame) - int(np.argmin(curve[::-1]))
    predicted = cut_tree(tree, n_clusters=[count]).ravel()
    # 真值标签仅用于以下外部指标，不参与上方图构建、Ward 或选簇过程。
    metrics = {"ARI": float(adjusted_rand_score(frame.label, predicted)),
               "NMI": float(normalized_mutual_info_score(frame.label, predicted)),
               "AMI": float(adjusted_mutual_info_score(frame.label, predicted)),
               "samples": len(frame), "true_events": int(frame.label.nunique()), "predicted_events": count,
               "structural_entropy": float(curve[count - 1]), "graph_edges": len(edges)}
    metrics["mean_score"] = sum(metrics[k] for k in ["ARI", "NMI", "AMI"]) / 3
    destination.mkdir(parents=True, exist_ok=True)
    # 同时保存特征、图、层次树和熵曲线，便于独立复核指标及最终选簇结果。
    write_json(destination / "metrics.json", metrics)
    pd.DataFrame({"source_row": frame.source_row, "image_name": frame.image_name,
                  "label_true": frame.label, "label_pred": predicted}).to_csv(destination / "predictions.csv", index=False)
    np.save(destination / "features.npy", features)
    np.save(destination / "ward_linkage.npy", tree)
    np.savez(destination / "graph.npz", edges=edges, weights=weights)
    pd.DataFrame({"n_clusters": np.arange(1, len(curve) + 1), "entropy": curve}).to_csv(
        destination / "entropy_curve.csv", index=False)
    return metrics



def worker(args):
    # 一个 worker 只负责一个数据集的一种设置，例如 NED/S1。
    dataset, setting = args.worker
    case = args.output / f"{dataset}_{setting}"
    case.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("high")
    recipe = RECIPE
    frames, audit = audit_setting(args.data_root, dataset, setting)
    if args.smoke:
        # 快速运行检查使用每份 CSV 的前若干行，不用于报告全量数据性能。
        frames = {part: frame.iloc[:257 if part == "train" else 129].copy().reset_index(drop=True)
                  for part, frame in frames.items()}
    for frame in frames.values():
        frame.insert(0, "source_row", np.arange(len(frame)))
    config = LFEventConfig()
    choices = implementation_choices(config)
    # 基类默认策略与实际训练类有区别，这里记录实际采用的原文+时间、Dropout 和两层预测器。
    choices.update({"text_input": "Original remains literal text+time; type/theme/caption independently encoded.",
        "encoder_policy": "Reference frozen MiniLM; training dropout ON; online encoding each training batch.",
        "predictor": "Reference two-layer Linear/BatchNorm/ReLU/Linear, unchanged including initialization.",
        "projection_head": "Reference three-layer projector, unchanged including initialization."})
    protocol = {"dataset": dataset, "setting": setting, "model": choices,
        "components": ["training_loss", "augmented_text_fusion", "CMAP"],
        "implementation": "LFEvent", "seed": 230, "epochs": args.epochs, "batch_size": 64,
        "optimizer": {"name": "SGD", "momentum": .9, "weight_decay": 1e-4, **recipe},
        "max_length": 512, "graph": "reference", "graph_detail": "Pearson 1NN + shared hashtag/@mention edges",
        "clustering": "Ward on raw predictor outputs; minimum SE over all N cuts",
        "selection": "Max validation mean(ARI,NMI,AMI); earliest tie; test only after selection",
        "smoke_only": args.smoke, "input_sha256": {k: v["sha256"] for k, v in audit["splits"].items()},
        "source_sha256": {source: sha256(REPO / source) for source in SOURCES}}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    # 协议包含数据与源代码哈希，防止把不同配置的训练结果混在同一输出目录。
    protocol["fingerprint"] = fingerprint
    if (case / "protocol.json").exists():
        if not args.resume or read(case / "protocol.json")["fingerprint"] != fingerprint:
            raise ValueError(f"Existing output/configuration differs: {case}")
    write_json(case / "protocol.json", protocol)
    write_json(case / "split_audit.json", audit)
    if args.resume and (case / "result.json").exists():
        return
    # 固定 Python、NumPy、PyTorch 和 CUDA 的随机种子；样本打乱另外使用独立生成器。
    random.seed(230)
    np.random.seed(230)
    torch.manual_seed(230)
    torch.cuda.manual_seed_all(230)
    model = ControlledLFEvent(config).to(args.device)
    write_json(case / "initial_controls.json", model.initial_controls)
    tokenizer = AutoTokenizer.from_pretrained(config.encoder, local_files_only=True)
    cache = {}
    logging.info("%s/%s: preparing four-source training with frozen MiniLM", dataset, setting)
    train = FourSourceInputs(frames["train"], dataset, args, tokenizer, cache)
    val = FourSourceInputs(frames["val"], dataset, args, tokenizer, cache)
    val_text = original_encodings(model, val, args.device)
    frozen_path = case / "frozen_encoder.pth"
    # 每组实验只存一份冻结编码器；每轮 checkpoint 只存训练头，避免重复保存大量权重。
    if not frozen_path.exists():
        save_checkpoint(frozen_path, {k: v.detach().cpu() for k, v in model.text_encoder.state_dict().items()})
    frozen_hash = sha256(frozen_path)
    optimizer = torch.optim.SGD(model.parameters(), lr=recipe["learning_rate"], momentum=.9, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(230)
    first_epoch, history, best_score, best_epoch = 1, [], -math.inf, 0
    last = case / "last.pth.tar"
    if args.resume and last.exists():
        # 完整恢复权重、优化器、历史记录及各随机状态，使下一轮沿用原训练进度。
        state = torch.load(last, map_location="cpu", weights_only=False)
        assert state["fingerprint"] == fingerprint and state["frozen_encoder_sha256"] == frozen_hash
        load_heads(model, state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        history, first_epoch = state["history"], state["epoch"] + 1
        best_score, best_epoch = state["best_score"], state["best_epoch"]
        generator.set_state(state["shuffle_rng"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"], args.device)
    eval_args = argparse.Namespace(device=args.device, batch_size=64, graph="reference")
    for epoch in range(first_epoch, args.epochs + 1):
        model.train()
        assert model.text_encoder.training
        started, count, steps, clipped, max_norm = time.monotonic(), 0, 0, 0, 0.
        total = {"loss": 0., "contrastive": 0., "modal": 0.}
        order = torch.randperm(len(train.images), generator=generator).tolist()
        for indices in batches(len(train.images), 64, order):
            optimizer.zero_grad(set_to_none=True)
            # 四路文本分别经过冻结 MiniLM；训练时 Dropout 开启，因此每批都在线编码。
            # 标签未传入模型或损失；图像特征与四路文本按相同 indices 保持配对。
            tokens = [train.tokens(indices, source, args.device) for source in range(4)]
            outputs = model(tokens, train.images[indices].to(args.device))
            losses = model.loss(outputs)
            if not all(torch.isfinite(value) for value in losses.values()):
                raise FloatingPointError(f"Nonfinite loss epoch {epoch} step {steps}")
            losses["loss"].backward()
            if epoch == 1 and steps == 0:
                write_json(case / "first_step_gradients.json", gradient_evidence(model))
            # 所有可训练模块的总梯度范数裁剪到 1；返回值 norm 是裁剪前的范数。
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),
                recipe["gradient_clip"] if recipe["gradient_clip"] is not None else math.inf, error_if_nonfinite=True))
            max_norm = max(max_norm, norm)
            clipped += int(recipe["gradient_clip"] is not None and norm > recipe["gradient_clip"])
            optimizer.step()
            # 按实际批次大小加权，最后一批可能含 65 条，避免计算 epoch 均值时产生偏差。
            for key in total:
                total[key] += float(losses[key].detach()) * len(indices)
            count += len(indices)
            steps += 1
        assert all(torch.isfinite(p).all() for p in model.parameters())
        # 验证前保存随机状态，验证后恢复，避免验证过程改变下一轮训练的随机序列。
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(args.device)
        metrics = evaluate(model, val_text, val.images, frames["val"],
            case / "evaluations" / f"epoch_{epoch:03d}" / "val", eval_args)
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, args.device)
        if metrics["mean_score"] > best_score:
            # 严格大于才更新，因此验证集均分相同时保留更早的 epoch。
            best_score, best_epoch = metrics["mean_score"], epoch
        record = {"epoch": epoch, **{k: v / count for k, v in total.items()},
            "samples": count, "steps": steps, "seconds": time.monotonic() - started,
            "max_gradient_norm": max_norm, "clipped_steps": clipped, "val": metrics}
        history.append(record)
        checkpoint = {"epoch": epoch, "state_dict": head_state(model), "model_config": asdict(config),
            "model_class": "ControlledLFEvent", "fingerprint": fingerprint,
            "frozen_encoder_file": "../frozen_encoder.pth", "frozen_encoder_sha256": frozen_hash}
        (case / "checkpoints").mkdir(exist_ok=True)
        save_checkpoint(case / "checkpoints" / f"epoch_{epoch:03d}.pth", checkpoint)
        save_checkpoint(last, {**checkpoint, "frozen_encoder_file": "frozen_encoder.pth",
            "optimizer": optimizer.state_dict(), "history": history, "best_score": best_score,
            "best_epoch": best_epoch, "shuffle_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(args.device)})
        write_json(case / "history.json", history)
        logging.info("%s/%s epoch %d/%d: L=%.5f L1=%.5f L2=%.5f val=%.5f seconds=%.1f", dataset, setting,
            epoch, args.epochs, record["loss"], record["contrastive"], record["modal"], metrics["mean_score"], record["seconds"])
    # 全部 epoch 完成后，加载验证集选出的最佳权重，再运行测试集；测试指标不用于挑选模型。
    selected = torch.load(case / "checkpoints" / f"epoch_{best_epoch:03d}.pth", map_location="cpu", weights_only=True)
    load_heads(model, selected["state_dict"])
    selected["frozen_encoder_file"] = "frozen_encoder.pth"
    save_checkpoint(case / "selected.pth", selected)
    test = FourSourceInputs(frames["test"], dataset, args, tokenizer, cache)
    test_text = original_encodings(model, test, args.device)
    metrics = evaluate(model, test_text, test.images, frames["test"], case / "selected_test", eval_args)
    result = {"dataset": dataset, "setting": setting, "seed": 230, "selected_epoch": best_epoch,
        "selection_val_mean": best_score, "smoke_only": args.smoke, **metrics}
    write_json(case / "result.json", result)
    logging.info("Selected %s/%s epoch%d test=%s", dataset, setting, best_epoch, json.dumps(metrics))
