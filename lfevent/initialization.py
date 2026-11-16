"""Reference initialization retained to preserve weights and RNG draw order.

Only the encoder, projector and predictor are used by ControlledLFEvent.
The reference attention is instantiated to preserve the original initialization.
"""
# 本文件保留参考网络的初始化顺序。ControlledLFEvent 只取其中的文本编码器、fc 和 predictor。
# 参考注意力也要先初始化，以保留随机数消耗顺序；当前实际融合使用 model.py 中的 CMAP。
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel

def mean_pooling(model_output, attention_mask):
    # 对最后一层 token 表示做掩码均值池化，忽略补齐位置，输出 [B,384] 句向量。
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)



class SimpleCrossAttention(nn.Module):
    # 参考网络的双向注意力；此处保留供初始化复现，当前训练不调用它进行特征融合。
    def __init__(self, img_hsz=768, txt_hsz=384):
        super().__init__()
        # 文本到图像的投影
        self.proj_q_txt2img = nn.Linear(txt_hsz, txt_hsz // 2)
        self.proj_k_txt2img = nn.Linear(img_hsz, txt_hsz // 2)
        self.proj_v_txt2img = copy.deepcopy(self.proj_k_txt2img)

        # 图像到文本的投影
        self.proj_q_img2txt = nn.Linear(img_hsz, img_hsz // 2)
        self.proj_k_img2txt = nn.Linear(txt_hsz, img_hsz // 2)
        self.proj_v_img2txt = copy.deepcopy(self.proj_k_img2txt)

        # 输出投影
        self.proj_o_txt2img = nn.Linear(txt_hsz // 2, txt_hsz)
        self.proj_o_img2txt = nn.Linear(img_hsz // 2, txt_hsz)

        # 维度缩放系数
        self.d_h_txt2img = txt_hsz // 2
        self.d_h_img2txt = img_hsz // 2

        # 融合权重参数 (可学习)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 初始值为 0.5，代码未对训练后的取值施加范围约束。

    def forward(self, img, txt):
        # 文本到图像的交叉注意力
        q_txt2img, k_txt2img, v_txt2img = self.proj_q_txt2img(txt), self.proj_k_txt2img(img), self.proj_v_txt2img(img)
        score_txt2img = torch.softmax(torch.matmul(q_txt2img, k_txt2img.transpose(-1, -2)) / math.sqrt(self.d_h_txt2img), dim=-1)
        o_txt2img = torch.matmul(score_txt2img, v_txt2img)
        out_txt2img = self.proj_o_txt2img(o_txt2img)

        # 图像到文本的交叉注意力
        q_img2txt, k_img2txt, v_img2txt = self.proj_q_img2txt(img), self.proj_k_img2txt(txt), self.proj_v_img2txt(txt)
        score_img2txt = torch.softmax(torch.matmul(q_img2txt, k_img2txt.transpose(-1, -2)) / math.sqrt(self.d_h_img2txt), dim=-1)
        o_img2txt = torch.matmul(score_img2txt, v_img2txt)
        out_img2txt = self.proj_o_img2txt(o_img2txt)

        # 融合输出：加权平均
        fused_output = self.alpha * out_txt2img + (1 - self.alpha) * out_img2txt

        # 输出双向融合的特征
        return fused_output



class Text_Decoder_SBERT(nn.Module):
    # 当前实际复用的 MiniLM 编码器：只冻结权重，不在此强制固定 train/eval 模式。
    def __init__(self):
        super(Text_Decoder_SBERT, self).__init__()

        # 加载 384 维 MiniLM；训练时保留 Dropout，推理时由 model.eval() 关闭。
        self.bert = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L12-v2").requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # 获取BERT模型的输出
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # 忽略 padding 后做均值池化。
        sentence_embeddings = mean_pooling(outputs, attention_mask)
        # 将每条句向量的 L2 范数归一化到 1。
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # pooled_output = self.fc(sentence_embeddings)
        return sentence_embeddings



class Net(nn.Module):
    # 初始化时按 编码器 -> 参考注意力 -> 三层 projector -> 两层 predictor 的顺序创建模块。
    # 当前 dim 和 pred_dim 的实际默认值均为 384，以函数签名和层定义为准。
    """
    Build a SimSiam model.
    """
    def __init__(self, dim=384, pred_dim=384):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(Net, self).__init__()
        # 先创建冻结文本编码器。
        self.text_decoder = Text_Decoder_SBERT()
        # 保留参考注意力的初始化，以保证后续 fc/predictor 的初始权重不偏移。
        self.crossattention = SimpleCrossAttention()
        # 三层投影器；其完整权重和 BN 状态会直接交给 ControlledLFEvent。
        prev_dim = dim
        self.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                nn.BatchNorm1d(prev_dim),
                                nn.ReLU(inplace=True), # first layer
                                nn.Linear(prev_dim, prev_dim, bias=False),
                                nn.BatchNorm1d(prev_dim),
                                nn.ReLU(inplace=True), # second layer
                                nn.Linear(prev_dim, prev_dim, bias=False),
                                nn.BatchNorm1d(dim, affine=False)) # output layer
        # 两层预测器：Linear -> BN -> ReLU -> Linear，当前实际训练沿用这一结构。
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                       nn.BatchNorm1d(pred_dim),
                                       nn.ReLU(inplace=True), # hidden layer
                                       nn.Linear(pred_dim, dim)) # output layer
    def forward(self, t1_input, t2_input, image_embed):
        # 以下 forward/cluster_feature 是参考网络保留接口。
        # 当前训练走 ControlledLFEvent 继承的 forward，不使用这里的 z2.detach()。
        """
        Input:
            t1_input: 原始文本输入，torch.Size([batch_size, 768])
            t1_input: 增强文本输入，torch.Size([batch_size, 768])
            image_embed: 图像特征，torch.Size([batch_size, 768])，来源于clip视觉编码器
        Output:
        """
        t1_embed = self.text_decoder(**t1_input)
        t2_embed = self.text_decoder(**t2_input)
        fused_feature1 = self.crossattention(image_embed, t1_embed)
        fused_feature2 = self.crossattention(image_embed, t2_embed)
        z1 = self.fc(fused_feature1)
        z2 = self.fc(fused_feature2)
        p1 = self.predictor(z1) # NxC

        return p1, z2.detach()


    def cluster_feature(self, t1_input, image_embed):
        t1_embed = self.text_decoder(**t1_input)
        fused_feature1 = self.crossattention(image_embed, t1_embed)
        z1 = self.fc(fused_feature1)
        p1 = self.predictor(z1)

        return p1

    def cluster_feature2(self, t2_input, image_embed):
        t1_embed = self.text_decoder(**t2_input)
        fused_feature1 = self.crossattention(image_embed, t1_embed)
        z1 = self.fc(fused_feature1)
        p1 = self.predictor(z1)

        return p1
