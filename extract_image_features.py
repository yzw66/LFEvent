"""用冻结的 CLIP ViT-L/14 为整张图片提取 768 维全局向量。"""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "openai/clip-vit-large-patch14"
FOLDERS = {"NED": "NED", "CrisisMMD": "MMD"}


def image_path(data_root, dataset, row):
    name = Path(row["image_name"].split(",")[0])
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"image_name 应为数据集内的相对路径：{name}")
    base = Path(data_root) / FOLDERS[dataset] / "Images"
    if dataset == "NED":
        return base / f"E{int(row['label'])+1}" / name
    if name.parts and name.parts[0] == "data_image":
        name = Path(*name.parts[1:])
    return base / name


def collect_images(data_root, dataset):
    csv_path = Path(data_root) / FOLDERS[dataset] / "text/Qwen2-VL-7B-Instruct/data.csv"
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    seen, result = set(), []
    for row in rows:
        raw = image_path(data_root, dataset, row)
        target = raw.with_suffix(".pt")
        # 同一图片可被多个帖子引用；按输出路径去重，只提取一次。
        if target not in seen:
            seen.add(target)
            result.append((raw, target))
    return result


def encode_images(model, processor, paths, device):
    import torch
    from PIL import Image
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    pixels = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode():
        # 使用投影后的整图 image_embeds；不做额外归一化，不提取 patch token。
        vectors = model(pixel_values=pixels).image_embeds.float().cpu()
    if vectors.shape != (len(paths), 768) or not torch.isfinite(vectors).all():
        raise ValueError(f"预期有限的 [B,768] 图像特征，实际得到 {vectors.shape}")
    return vectors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/Event_data")
    parser.add_argument("--datasets", nargs="+", choices=list(FOLDERS), default=list(FOLDERS))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="每个数据集最多检查多少张图，0 表示全部")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="重新提取并覆盖已有 .pt；默认跳过已有文件")
    args = parser.parse_args()
    if args.batch_size < 1 or args.limit < 0:
        raise ValueError("batch-size 必须为正数，limit 必须非负")
    data_root = args.data_root.resolve()
    work, skipped = {}, {}
    for dataset in args.datasets:
        paths = collect_images(data_root, dataset)
        if args.limit:
            paths = paths[:args.limit]
        work[dataset] = [(raw, target) for raw, target in paths if args.overwrite or not target.exists()]
        skipped[dataset] = len(paths)-len(work[dataset])
        missing = [str(raw) for raw, _ in work[dataset] if not raw.is_file()]
        if missing:
            raise FileNotFoundError(f"{dataset} 缺少 {len(missing)} 张图片，例如：{missing[0]}")
    if not any(work.values()):
        print(f"所选图片均已有 .pt，未改写文件。跳过数量：{skipped}")
        return

    import torch
    from transformers import CLIPImageProcessorPil, CLIPVisionModelWithProjection
    torch.set_num_threads(4)
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CLIPVisionModelWithProjection.from_pretrained(args.model, local_files_only=args.local_files_only)
    if model.config.projection_dim != 768:
        raise ValueError("当前网络需要 768 维 CLIP 全局特征，请选择兼容的视觉编码器")
    model = model.eval().requires_grad_(False).to(device)
    processor = CLIPImageProcessorPil.from_pretrained(args.model, local_files_only=args.local_files_only)
    for dataset, paths in work.items():
        if not paths:
            continue
        written = []
        for start in range(0, len(paths), args.batch_size):
            batch = paths[start:start+args.batch_size]
            vectors = encode_images(model, processor, [raw for raw, _ in batch], device)
            for (_, target), vector in zip(batch, vectors):
                temporary = target.with_name(target.name+".tmp")
                # clone 避免把切片所引用的整批张量存储一起写入单张图片的文件。
                torch.save(vector.clone(), temporary)
                temporary.replace(target)
                written.append(str(target.relative_to(data_root)))
            print(f"{dataset}: {min(start+args.batch_size,len(paths))}/{len(paths)}", flush=True)
        report = {"model": args.model, "model_commit": getattr(model.config, "_commit_hash", None),
                  "dimension": 768, "dtype": "float32", "feature": "image_embeds",
                  "batch_size": args.batch_size, "device": str(device), "torch_version": str(torch.__version__),
                  "image_size": model.config.image_size, "processor": processor.to_dict(),
                  "newly_written_features": written, "skipped_existing_features": skipped[dataset]}
        # 仅声明本次新生成文件的提取配置；不推断被跳过的已有特征来自哪个模型。
        report_path = data_root / FOLDERS[dataset] / "feature_extraction.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
