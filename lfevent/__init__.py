"""LFEvent implementation."""
import os

# 默认使用本机模型缓存并关闭 tokenizer 并行；若外部已设置环境变量，保留外部设置。
for _name, _value in {"TOKENIZERS_PARALLELISM": "false", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}.items():
    os.environ.setdefault(_name, _value)
