"""从完整样本表和固定分割清单恢复训练所需的 CSV；不复制图片或特征。"""
import argparse
import csv
import hashlib
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = {"NED": "NED", "CrisisMMD": "MMD"}
ID_FIELDS = ("text", "image_name", "time")
RAW_FIELDS = ID_FIELDS + ("label",)
REQUIRED = set(ID_FIELDS) | {"label", "type", "theme", "caption"}


def sample_id(dataset, row):
    # 不使用图片名单独标识样本；保留文本、时间的原始值，不去空格、不改大小写。
    identity = [dataset] + [row[field] for field in ID_FIELDS]
    content = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    prefix = "ned" if dataset == "NED" else "crisis"
    return prefix + "_" + hashlib.sha256(content).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if not fields or len(fields) != len(set(fields)):
            raise ValueError(f"CSV 缺少表头或存在重复列：{path}")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError(f"CSV 存在列数不一致的记录：{path}")
    return fields, rows


def load_samples(metadata_dir, dataset, raw_dir=None):
    """将公开增强表与用户本地原始字段按内容 ID 合并，不按行号猜测对应关系。"""
    metadata_path = Path(metadata_dir) / f"{dataset}.csv"
    fields, samples = read_csv(metadata_path)
    if not (REQUIRED | {"sample_id"}).issubset(fields) or not samples:
        raise ValueError(f"样本表缺少必要字段或为空：{metadata_path}")
    if raw_dir is not None:
        raw_path = Path(raw_dir) / f"{dataset}.csv"
        raw_fields, raw_samples = read_csv(raw_path)
        if not set(RAW_FIELDS).issubset(raw_fields) or not raw_samples:
            raise ValueError(f"本地原始样本表需要 text,image_name,time,label：{raw_path}")
        raw_index = {}
        for row in raw_samples:
            key = sample_id(dataset, row)
            if row.get("sample_id") and row["sample_id"] != key:
                raise ValueError(f"本地原始样本的 sample_id 与内容不一致：{raw_path}")
            if key in raw_index:
                raise ValueError(f"本地原始样本表存在重复 sample_id：{key}")
            raw_index[key] = row
        merged = []
        for row in samples:
            key = row["sample_id"]
            if key not in raw_index:
                raise ValueError(f"本地原始样本表缺少匹配样本：{dataset} / {key}；"
                                 "请核对原文、图片路径和时间的原始字符串，不能按行号匹配。")
            raw = raw_index[key]
            for field in RAW_FIELDS:
                if row[field] and row[field] != raw[field]:
                    raise ValueError(f"样本表与本地原始字段冲突：{key} / {field}")
            # tags 可从本地原文确定性恢复；公开表不包含帖子中的 #hashtag 或 @mention。
            tags = str(re.findall(r"[#@]\w+", raw["text"]))
            merged.append({**row, **{field: raw[field] for field in RAW_FIELDS}, "tags": tags})
        samples = merged
    for row in samples:
        if any(not row[field] for field in RAW_FIELDS):
            raise ValueError(f"{metadata_path} 的原始字段尚未补齐。请按 data/README.md 准备本地 CSV，"
                             "然后运行 python prepare_data.py --raw-dir /path/to/raw_csvs。")
        if "tags" in fields and not row["tags"]:
            row["tags"] = str(re.findall(r"[#@]\w+", row["text"]))
    rules_path = Path(metadata_dir) / "text_copy_fields.json"
    if rules_path.exists():
        rules = json.loads(rules_path.read_text(encoding="utf-8")).get(dataset, {})
        by_id = {row["sample_id"]: row for row in samples}
        for key, columns in rules.items():
            if key not in by_id or not columns or any(column not in {"caption", "theme", "type"} for column in columns):
                raise ValueError(f"原文恢复规则无效：{rules_path} / {key}")
            for column in columns:
                row = by_id[key]
                if row[column] and row[column] != row["text"]:
                    raise ValueError(f"原文恢复规则与已有增强字段冲突：{key} / {column}")
                row[column] = row["text"]
    return fields, samples


def build_plan(metadata_dir, splits_dir, data_root, datasets, raw_dir=None):
    plan = []
    for dataset in datasets:
        metadata_path = Path(metadata_dir) / f"{dataset}.csv"
        fields, samples = load_samples(metadata_dir, dataset, raw_dir)
        index = {}
        for row in samples:
            key = row["sample_id"]
            if key != sample_id(dataset, row):
                raise ValueError(f"sample_id 与样本内容不一致：{metadata_path} / {key}")
            if key in index:
                raise ValueError(f"完整样本表存在重复 sample_id：{key}")
            index[key] = row
        # 去掉索引列，输出字段及字段次序与训练 CSV 一致。
        output_fields = [field for field in fields if field != "sample_id"]
        destination = Path(data_root) / DATASETS[dataset] / "text/Qwen2-VL-7B-Instruct"
        plan.append((destination / "data.csv", output_fields, samples))
        for setting, suffix in [("S1", ""), ("S2", "2")]:
            seen = set()
            for part in ["train", "val", "test"]:
                path = Path(splits_dir) / dataset / setting / f"{part}.csv"
                manifest_fields, manifest = read_csv(path)
                if manifest_fields != ["order", "sample_id"] or not manifest:
                    raise ValueError(f"分割清单应包含 order,sample_id 两列且不能为空：{path}")
                selected = []
                for position, item in enumerate(manifest):
                    # 严格保留清单顺序，不按 ID 排序，确保恢复原来的训练与评测批次。
                    if item["order"] != str(position):
                        raise ValueError(f"order 必须从 0 连续递增：{path} / 第 {position} 行")
                    key = item["sample_id"]
                    if key not in index:
                        raise ValueError(f"分割清单中存在未知 sample_id：{path} / {key}")
                    if key in seen:
                        raise ValueError(f"同一设置存在重复或跨分割重叠的 sample_id：{key}")
                    seen.add(key)
                    selected.append(index[key])
                plan.append((destination / f"{part}{suffix}.csv", output_fields, selected))
            if seen != set(index):
                raise ValueError(f"{dataset}/{setting} 未覆盖全部样本，缺少 {len(set(index)-seen)} 条")
    return plan


def csv_payload(fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def write_plan(plan, overwrite=False):
    # 写入前检查全部目标，默认不会覆盖已有的不同数据；图片目录始终不作修改。
    outputs = [(path, csv_payload(fields, rows)) for path, fields, rows in plan]
    for path, payload in outputs:
        if path.exists() and path.read_bytes() != payload and not overwrite:
            raise FileExistsError(f"目标 CSV 已存在且内容不同：{path}；请换目录或显式传 --overwrite")
    for path, payload in outputs:
        if path.exists() and path.read_bytes() == payload:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--raw-dir", type=Path, help="用户自备原始字段 CSV 所在目录，包含 NED.csv / CrisisMMD.csv")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "splits")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/Event_data")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS), default=list(DATASETS))
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖内容不同的已有 CSV")
    args = parser.parse_args()
    missing = [str(args.metadata_dir / f"{dataset}.csv") for dataset in args.datasets
               if not (args.metadata_dir / f"{dataset}.csv").is_file()]
    if missing:
        parser.error("缺少完整样本表：" + ", ".join(missing) +
                     "。请先按 data/README.md 准备授权数据，再运行本命令。")
    try:
        plan = build_plan(args.metadata_dir.resolve(), args.splits_dir.resolve(), args.data_root.resolve(), args.datasets,
                          args.raw_dir.resolve() if args.raw_dir else None)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    write_plan(plan, overwrite=args.overwrite)
    for path, _, rows in plan:
        print(f"{path}: {len(rows)} rows")
    print("CSV 已准备完成，请将图片或图像特征放到对应数据集的 Images 目录。")


if __name__ == "__main__":
    main()
