"""Existing CSV splits, four independent text sources and pooled image inputs."""
# 本模块沿用已有 CSV 分割和增强字段；不重新划分数据，也不在训练时调用增强模型。
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .utils import sha256

EXPECTED = {"NED": (40, {"S1": 30, "S2": 20}), "CrisisMMD": (7, {"S1": 5, "S2": 4})}
# KEY 用于识别相同样本；EXPECTED 记录事件总数及各设置的训练已知事件数。
KEY = ["text", "image_name", "time", "label"]


def dataset_directory(root, dataset):
    # 代码中的 CrisisMMD 名称对应磁盘上的 MMD 目录。
    return root / ("MMD" if dataset == "CrisisMMD" else "NED")



def read_split(path):
    # 缺失的增强字段保留为空字符串，不补造 type/theme/caption 内容。
    frame = pd.read_csv(path).fillna("")
    required = set(KEY + ["caption", "theme", "type"])
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: {required - set(frame.columns)}")
    return frame



def audit_setting(root, dataset, setting):
    # S1 使用 train/val/test.csv；S2 使用 train2/val2/test2.csv。
    base = dataset_directory(root, dataset) / "text/Qwen2-VL-7B-Instruct"
    suffix = "" if setting == "S1" else "2"
    paths = {part: base / f"{part}{suffix}.csv" for part in ["train", "val", "test"]}
    frames = {part: read_split(path) for part, path in paths.items()}
    full = read_split(base / "data.csv")
    total_classes, known_by_setting = EXPECTED[dataset]
    known = set(frames["train"].label)
    all_classes = set(full.label)
    if len(known) != known_by_setting[setting] or len(all_classes) != total_classes:
        raise ValueError(f"Unexpected class counts for {dataset}/{setting}")
    # 检查三份分割是否含完全相同的样本，并另行记录不同帖子复用图片的情况。
    row_sets = {part: set(map(tuple, frame[KEY].values)) for part, frame in frames.items()}
    image_sets = {part: set(frame.image_name) for part, frame in frames.items()}
    row_overlap, image_overlap = {}, {}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        name = f"{a}_{b}"
        row_overlap[name] = len(row_sets[a] & row_sets[b])
        image_overlap[name] = len(image_sets[a] & image_sets[b])
    if any(row_overlap.values()):
        raise ValueError(f"Identical sample leakage in {dataset}/{setting}: {row_overlap}")
    for part in ["val", "test"]:
        if set(frames[part].label) != all_classes:
            raise ValueError(f"{dataset}/{setting}/{part} does not contain all classes")
    union = pd.concat(list(frames.values()), ignore_index=True)
    if len(union) != len(full):
        raise ValueError(f"Split total differs from data.csv for {dataset}/{setting}")
    # 已知事件按 train/val/test=80%/10%/10%，未知事件只分配到 val/test，各占 50%。
    # 检查已有分割的比例，允许每类取整带来的少量样本差异。
    proportions = {}
    for label, count in union.label.value_counts().sort_index().items():
        counts = {part: int(frame.label.eq(label).sum()) for part, frame in frames.items()}
        targets = {"train": .8, "val": .1, "test": .1} if label in known else {"train": 0, "val": .5, "test": .5}
        if any(abs(counts[part] - count * targets[part]) > 2 for part in counts):
            raise ValueError(f"Invalid split proportions for {dataset}/{setting}/class {label}")
        proportions[str(label)] = counts
    full_rows = Counter(map(tuple, full[KEY].values))
    split_rows = Counter(map(tuple, union[KEY].values))
    notes = []
    # 以分割 CSV 为实际输入；data.csv 与分割合并结果若有差异，仅记录在审计信息中。
    if any(image_overlap.values()):
        notes.append("Different posts reuse images across splits; supplied assignments are preserved.")
    if full_rows != split_rows:
        notes.append("data.csv differs from the union of the supplied split CSVs; split CSVs are authoritative.")
    audit = {
        "dataset": dataset, "setting": setting,
        "known_classes": sorted(map(int, known)), "unseen_classes": sorted(map(int, all_classes - known)),
        "splits": {part: {"path": str(paths[part]), "sha256": sha256(paths[part]), "rows": len(frame),
                          "classes": int(frame.label.nunique()),
                          "empty_semantic_fields": {c: int(frame[c].eq("").sum()) for c in ["caption", "theme", "type"]}}
                   for part, frame in frames.items()},
        "class_counts": proportions, "identical_sample_overlap": row_overlap, "shared_images": image_overlap,
        "full_csv_sha256": sha256(base / "data.csv"),
        "full_csv_rows_not_in_splits": sum((full_rows - split_rows).values()),
        "split_rows_not_in_full_csv": sum((split_rows - full_rows).values()), "notes": notes,
    }
    return frames, audit



class Inputs:
    # 原版数据容器：保存原文、拼接增强文本、图像及标签；当前四路流程只复用原文和图像。
    def __init__(self, tokenizer, original, enhanced, images, labels):
        self.tokenizer = tokenizer
        self.original = original
        self.enhanced = enhanced
        self.images = images
        self.labels = labels

    def tokens(self, indices, enhanced=False, device="cuda:0"):
        source = self.enhanced if enhanced else self.original
        items = [{key: values[i] for key, values in source.items()} for i in indices]
        return self.tokenizer.pad(items, padding=True, return_tensors="pt").to(device)



def prepare_inputs(frame, dataset, root, tokenizer, max_length, image_cache):
    # 沿用原来的字面字符串拼接，不额外插入分隔符；原始文本分支包含 time。
    original = (frame.text.astype(str) + frame.time.astype(str)).tolist()
    enhanced = (frame.text.astype(str) + frame.type.astype(str) + frame.theme.astype(str)
                + frame.caption.astype(str) + frame.time.astype(str)).tolist()
    original_tokens = tokenizer(original, truncation=True, max_length=max_length)
    enhanced_tokens = tokenizer(enhanced, truncation=True, max_length=max_length)
    image_base = dataset_directory(root, dataset) / "Images"
    images = []
    for row in frame.itertuples():
        # 多图片字段沿用第一个文件名，将图片后缀替换为 .pt，读取已提取的 768 维全局特征。
        image_name = Path(str(row.image_name).split(",")[0]).with_suffix(".pt")
        if dataset == "NED":
            # NED 图片按事件目录存储；这里的 label 只用于定位文件，不作为训练监督信号。
            path = image_base / f"E{int(row.label) + 1}" / image_name
        else:
            if image_name.parts[0] == "data_image":
                image_name = Path(*image_name.parts[1:])
            path = image_base / image_name
        key = str(path)
        if key not in image_cache:
            # .detach() 去除原特征可能携带的梯度标记；相同图片在本次 worker 中只读取一次。
            image = torch.load(path, map_location="cpu", weights_only=True).detach().float().reshape(-1)
            if image.shape != (768,) or not torch.isfinite(image).all():
                raise ValueError(f"Invalid image embedding: {path}: {image.shape}")
            image_cache[key] = image
        images.append(image_cache[key])
    return Inputs(tokenizer, original_tokens, enhanced_tokens, torch.stack(images), frame.label.to_numpy(dtype=int))



class FourSourceInputs:
    def __init__(self, frame, dataset, args, tokenizer, image_cache):
        # 四路依次为 [原文+时间, type, theme, caption]；每一路单独截断，最长 512 个 token。
        original = prepare_inputs(frame, dataset, args.data_root, tokenizer, 512, image_cache)
        self.tokenizer, self.images = tokenizer, original.images
        self.sources = [original.original] + [tokenizer(frame[column].astype(str).tolist(),
            truncation=True, max_length=512) for column in ["type", "theme", "caption"]]

    def tokens(self, indices, source, device):
        # source 指定哪一路文本；按当前批次最长文本动态补齐，indices 同时用于配对图像。
        tokens = self.sources[source]
        items = [{key: values[i] for key, values in tokens.items()} for i in indices]
        return self.tokenizer.pad(items, padding=True, return_tensors="pt").to(device)
