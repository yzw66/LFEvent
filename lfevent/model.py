"""LFEvent text fusion, pooled CMAP and contrastive objectives.

ControlledLFEvent below retains the active encoder policy and reference heads.
See implementation_choices() and the training protocol for implementation details.
"""

# 阅读顺序：先看文件末尾实际训练的 ControlledLFEvent，再看文本融合、CMAP 和损失。
# 形状约定：B=批次样本数，D=文本维度（384），V=图像维度（768），P=图像 token 数。

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class LFEventConfig:
    # 模型结构与损失参数；学习率、梯度裁剪等训练参数在 engine.py 中设置。
    encoder: str = "sentence-transformers/all-MiniLM-L12-v2"
    text_dim: int = 384
    image_dim: int = 768
    attention_heads: int = 4
    fusion_hidden: int = 384
    temperature: float = .07
    modal_weight: float = .3

    def __post_init__(self):
        if min(self.text_dim, self.image_dim, self.attention_heads, self.fusion_hidden) <= 0:
            raise ValueError("Dimensions and attention_heads must be positive")
        if self.text_dim % self.attention_heads:
            raise ValueError("text_dim must be divisible by attention_heads")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not math.isfinite(self.modal_weight) or self.modal_weight < 0:
            raise ValueError("modal_weight must be finite and nonnegative")


def implementation_choices(config):
    """Explicit choices, not additional claims about the authors' implementation."""
    # 记录基类的实现说明；实际训练时，engine.worker 会覆盖编码器、预测器等相关说明。
    return {
        "config": asdict(config),
        "source_order": ["original", "type", "theme", "caption"],
        "text_input": "Each source encoded independently; original is raw text, without appended time.",
        "encoder_policy": "Existing MiniLM checkpoint; frozen weights AND eval mode, allowing feature caching.",
        "gates": "Four independent scalar Linear gates; softmax over sources, following Eq.4. Prose says sigmoid.",
        "fusion": "4-source self-attention, weighted concatenation, Linear/ReLU/Linear/LayerNorm (Eqs.3-5).",
        "cmap": "One shared text-query/image-key-value attention, dk=text_dim, then output Linear (Eqs.7-8).",
        "attention_layout": "For supplied pooled [B,768] vectors, Eq.7 is a B-by-B matrix over batch images. "
                            "Features depend on batch context and evaluation batch size/order. "
                            "Also accepts [B,P,768] patch tokens for per-sample attention when P>=2. "
                            "The paper does not specify these axes; existing pooled data cannot recover patch tokens.",
        "projection_head": "Existing reference three-layer Linear/BatchNorm/ReLU projector retained; exact MLP unspecified.",
        "predictor": "Single Linear, following Section4.1.3's simple linear predictor.",
        "target_gradient": "Eq.9 differentiated through both branches. No detach or EMA: otherwise the new "
                           "augmentation-only fusion would have no training signal. Exact alternating schedule unspecified.",
        "modal_alignment": "Reuse CMAP's image K and original-text Q projections to align 768/384 to text_dim "
                           "for Eq.10. This makes L2 update the actual fusion network, rather than discarded auxiliary heads. "
                           "Projection sharing and choice of text branch unspecified in paper.",
        "missing_sources": "Empty fields encoded as empty strings; no invented annotations or source masking.",
        "inference": "Original text -> shared CMAP -> projector -> predictor; no augmented sources required.",
    }


class FrozenSentenceEncoder(nn.Module):
    # 基类默认编码器：冻结权重并固定 eval，适合缓存特征。
    # 当前 ControlledLFEvent 使用 initialization.py 的编码器，训练模式由其 train() 控制。
    def __init__(self, model_name):
        super().__init__()
        from transformers import AutoModel
        self.bert = AutoModel.from_pretrained(model_name, local_files_only=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # 这个基类编码器始终关闭 Dropout，保证缓存与在线编码一致。
        return super().train(False)

    @torch.no_grad()
    def forward(self, input_ids, attention_mask, **kwargs):
        hidden = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)[0]
        # hidden: [B,文本长度,D]；只对有效 token 求均值，再将句向量做 L2 归一化。
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return F.normalize(pooled, dim=-1)


class AugmentedTextFusion(nn.Module):
    """Attend across four sources *within each sample*, then apply Eq.4 gates."""
    def __init__(self, dim, heads, hidden):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=0., batch_first=True)
        self.gates = nn.ModuleList([nn.Linear(dim, 1) for _ in range(4)])
        self.fusion = nn.Sequential(nn.Linear(4 * dim, hidden), nn.ReLU(),
                                    nn.Linear(hidden, dim), nn.LayerNorm(dim))

    def forward(self, sources, return_weights=False):
        if sources.ndim != 3 or sources.shape[1] != 4:
            raise ValueError("Expected [batch, 4, text_dim] sources in original/type/theme/caption order")
        # sources: [B,4,D]，四路依次为 原文、type、theme、caption。
        # 注意力沿四路文本这一维计算，每个样本内部融合，样本之间不混合。
        attended, _ = self.attention(sources, sources, sources, need_weights=False)
        # 每一路原始句向量通过独立门控得到分数，softmax 后四路权重之和为 1。
        logits = torch.cat([gate(sources[:, i]) for i, gate in enumerate(self.gates)], dim=1)
        weights = logits.softmax(dim=1)
        # 加权后的四路特征展平为 [B,4D]，经两层线性变换和 LayerNorm 得到 [B,D]。
        fused = self.fusion((attended * weights.unsqueeze(-1)).flatten(1))
        return (fused, weights) if return_weights else fused


class CMAP(nn.Module):
    """Unidirectional text -> image attention, Eqs.7--8 (no reverse branch/alpha)."""
    def __init__(self, text_dim, image_dim):
        super().__init__()
        # 文本生成查询 Q；图像生成键 K 和值 V；输出 o 将融合结果映射回文本维度。
        self.q = nn.Linear(text_dim, text_dim, bias=False)
        self.k = nn.Linear(image_dim, text_dim, bias=False)
        self.v = nn.Linear(image_dim, text_dim, bias=False)
        self.o = nn.Linear(text_dim, text_dim, bias=False)
        self.scale = math.sqrt(text_dim)

    def forward(self, text, image):
        if text.ndim != 2 or image.ndim not in (2, 3) or len(text) != len(image):
            raise ValueError("Expected text [B,D] and image [B,V] or [B,P,V]")
        query, key, value = self.q(text), self.k(image), self.v(image)
        if image.ndim == 2:
            # 当前训练走此分支：图像输入为已存储的全局向量 [B,V]。
            # Q @ K.T 得到 [B,B]，每个文本对同一批次的图像值向量进行加权。
            # 因此输出依赖批次组成；评估沿用原 CSV 顺序和原分批方式。
            attention = (query @ key.T / self.scale).softmax(dim=-1)
            result = attention @ value
        else:
            # 兼容 [B,P,V] 输入：每个文本只关注自己的 P 个图像 token。
            # 当前全局特征训练流程不使用这个分支。
            if image.shape[1] < 2:
                raise ValueError("A single image token makes attention independent of text; use patch tokens P>=2")
            attention = (query.unsqueeze(1) @ key.transpose(1, 2) / self.scale).softmax(dim=-1)
            result = (attention @ value).squeeze(1)
        return self.o(result)


def diagonal_contrastive_loss(predicted, target, temperature):
    """Row-wise InfoNCE with all batch candidates, computed stably as Eq.9/10."""
    if predicted.ndim != 2 or predicted.shape != target.shape or len(predicted) < 2:
        raise ValueError("Contrastive loss requires matching [B,D] features and B>=2")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    # 归一化后计算 [B,B] 余弦相似度；第 i 行的正样本位于第 i 列，其余列为负样本。
    # temperature 越小，softmax 对相似度差异越敏感；交叉熵内部稳定地计算 log-softmax。
    logits = F.normalize(predicted, dim=-1) @ F.normalize(target, dim=-1).T / temperature
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def lfevent_loss(outputs, temperature=.07, modal_weight=.3):
    """Eq.11 = prediction/target Eq.9 + lambda * image/text Eq.10."""
    if not math.isfinite(modal_weight) or modal_weight < 0:
        raise ValueError("modal_weight must be finite and nonnegative")
    # L1：原始分支的预测向量与增强分支的投影向量对齐，两个分支都参与反向传播。
    contrastive = diagonal_contrastive_loss(outputs["prediction"], outputs["target"], temperature)
    # L2：图像与原始文本经共享 K/Q 投影后对齐；总损失为 L1 + 0.3 * L2。
    modal = diagonal_contrastive_loss(outputs["image_alignment"], outputs["text_alignment"], temperature)
    return {"loss": contrastive + modal_weight * modal, "contrastive": contrastive, "modal": modal}


class LFEvent(nn.Module):
    # 网络计算的基类；实际使用的编码器、投影器、预测器由 ControlledLFEvent 保留原版配置。
    def __init__(self, config=None, text_encoder=None):
        super().__init__()
        self.config = config or LFEventConfig()
        c = self.config
        self.text_encoder = text_encoder if text_encoder is not None else FrozenSentenceEncoder(c.encoder)
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.augmented_fusion = AugmentedTextFusion(c.text_dim, c.attention_heads, c.fusion_hidden)
        self.cmap = CMAP(c.text_dim, c.image_dim)
        # 三层 projector 将 CMAP 输出映射到对比空间，最后一层 BN 不带可学习缩放和平移。
        self.projector = nn.Sequential(
            nn.Linear(c.text_dim, c.text_dim, bias=False), nn.BatchNorm1d(c.text_dim), nn.ReLU(),
            nn.Linear(c.text_dim, c.text_dim, bias=False), nn.BatchNorm1d(c.text_dim), nn.ReLU(),
            nn.Linear(c.text_dim, c.text_dim, bias=False), nn.BatchNorm1d(c.text_dim, affine=False))
        # 基类的单层预测器会在 ControlledLFEvent 中替换为原版两层预测器。
        self.predictor = nn.Linear(c.text_dim, c.text_dim)

    def train(self, mode=True):
        # 基类策略；ControlledLFEvent 覆盖此方法，训练时保留冻结编码器的 Dropout。
        super().train(mode)
        self.text_encoder.eval()
        return self

    @torch.no_grad()
    def encode_sources(self, token_sources):
        """Four independent PLM calls, with no concatenation/truncation across sources."""
        if len(token_sources) != 4:
            raise ValueError("Expected four separately tokenized sources")
        return torch.stack([self.text_encoder(**source) for source in token_sources], dim=1)

    def forward_features(self, sources, images):
        if sources.ndim != 3 or sources.shape[1:] != (4, self.config.text_dim):
            raise ValueError("Expected [B,4,text_dim] encoded sources")
        # 增强分支融合四路文本；原始分支只取 sources[:,0]，两者共享同一个 CMAP。
        enhanced = self.augmented_fusion(sources)
        original_fused = self.cmap(sources[:, 0], images)
        enhanced_fused = self.cmap(enhanced, images)
        original_projected = self.projector(original_fused)
        # 两个分支共享 projector；增强分支 target 不 detach，使文本增强模块能获得梯度。
        target = self.projector(enhanced_fused)
        alignment_images = images if images.ndim == 2 else images.mean(dim=1)
        # prediction/target 用于 L1；image_alignment/text_alignment 用于 L2。
        # 当前全局图像输入直接用于 L2，不重新编码或替换原有图像特征。
        return {"prediction": self.predictor(original_projected), "target": target,
                "image_alignment": self.cmap.k(alignment_images),
                "text_alignment": self.cmap.q(sources[:, 0])}

    def forward(self, token_sources, images):
        return self.forward_features(self.encode_sources(token_sources), images)

    def loss(self, outputs):
        return lfevent_loss(outputs, self.config.temperature, self.config.modal_weight)

    def cluster_features(self, original_text_features, images):
        """Inference uses only original text/image features, per Section4.2."""
        # 聚类特征取 predictor 的输出；推理不需要 type/theme/caption 等增强字段参与计算。
        return self.predictor(self.projector(self.cmap(original_text_features, images)))

    def cluster_feature(self, original_tokens, images):
        return self.cluster_features(self.text_encoder(**original_tokens), images)


from .initialization import Net as ReferenceNet
from .utils import tensor_digest


class ControlledLFEvent(LFEvent):
    # train.py 和 evaluate.py 实际构建的模型类；冻结 MiniLM 权重，沿用原版投影器和预测器。
    def __init__(self, config=None):
        # 先按原顺序初始化参考网络，让保留模块的初始权重与原版完全一致。
        reference = ReferenceNet()
        initial_rng = torch.get_rng_state()
        super().__init__(config or LFEventConfig(), text_encoder=reference.text_decoder)
        self.projector = reference.fc
        self.predictor = reference.predictor
        # 保存各模块初始权重的哈希，便于检查后续整理或改动是否改变初始化。
        self.initial_controls = {
            "projector_sha256": tensor_digest(self.projector.state_dict()),
            "predictor_sha256": tensor_digest(self.predictor.state_dict()),
            "encoder_sha256": tensor_digest(self.text_encoder.state_dict()),
            "reference_projector_predictor_copied_exactly": True,
        }
        # 恢复 CPU 随机数状态，避免新增模块的初始化改变后续随机数序列。
        torch.set_rng_state(initial_rng)

    def train(self, mode=True):
        # 冻结权重与 eval 模式是两回事：这里训练时开启编码器 Dropout，但不更新其权重。
        # 调用 model.eval() 时 mode=False，Dropout 和 BatchNorm 均切换到评估模式。
        nn.Module.train(self, mode)
        return self
