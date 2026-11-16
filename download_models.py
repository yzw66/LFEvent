"""准备 MiniLM 文本编码器和 CLIP 图像编码器的 Hugging Face 缓存。"""
import argparse

MODELS = {"text": "sentence-transformers/all-MiniLM-L12-v2", "image": "openai/clip-vit-large-patch14"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=list(MODELS), default=list(MODELS))
    parser.add_argument("--offline", action="store_true", help="仅验证本机缓存，不访问网络")
    args = parser.parse_args()
    from transformers import AutoModel, AutoTokenizer, CLIPImageProcessorPil, CLIPVisionModelWithProjection
    for kind in args.models:
        name = MODELS[kind]
        kwargs = {"local_files_only": args.offline}
        if kind == "text":
            AutoTokenizer.from_pretrained(name, **kwargs)
            model = AutoModel.from_pretrained(name, **kwargs)
        else:
            CLIPImageProcessorPil.from_pretrained(name, **kwargs)
            model = CLIPVisionModelWithProjection.from_pretrained(name, **kwargs)
        del model
        print(f"Ready: {kind} / {name}", flush=True)


if __name__ == "__main__":
    main()
