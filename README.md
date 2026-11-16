# LFEvent

论文 **[Label-Free Contrastive Learning for Open-World Multimodal Social Event Detection](https://doi.org/10.1145/3773966.3777919)** 的代码，用于开放世界多模态社交事件检测，支持 NED 和 CrisisMMD 的 S1/S2 训练与评测。

## 环境配置

在仓库根目录执行。训练使用 Python 3.10 和 CUDA GPU。

```bash
git clone https://github.com/yzw66/LFEvent.git
cd LFEvent
conda create -n lfevent python=3.10 -y
conda activate lfevent
pip install -r requirements.txt
```

准备文本编码器缓存：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python download_models.py --models text
```

文本编码器为 `sentence-transformers/all-MiniLM-L12-v2`。已有缓存时可用 `--offline` 检查，训练和评测从本机缓存加载。

## 数据准备

### 1. 完整样本表与分割清单

公开仓库提供 12 份固定分割 ID 清单：

```text
splits/
  NED/S1/{train,val,test}.csv
  NED/S2/{train,val,test}.csv
  CrisisMMD/S1/{train,val,test}.csv
  CrisisMMD/S2/{train,val,test}.csv
```

每份清单包含 `order,sample_id`，保留样本的训练和评测顺序。完整样本表包含原始帖子文本，须按数据提供方条款获取，不随本公开包分发。数据来源、字段及授权获取方式见 [data/README.md](data/README.md) 和 [docs/DATA_USAGE.md](docs/DATA_USAGE.md)。

将取得的完整样本表放为 `data/NED.csv` 和 `data/CrisisMMD.csv` 后执行：

```bash
python prepare_data.py
```

默认生成 `data/Event_data/` 下的训练 CSV。脚本校验 ID 唯一性、顺序、分割互斥和样本覆盖；相同内容可重复生成，覆盖不同内容的已有 CSV 需要显式传入 `--overwrite`。

| 数据集 | 设置 | Train | Val | Test | 已知/未知事件数 |
|---|---|---:|---:|---:|---:|
| NED | S1 | 9846 | 3535 | 3574 | 30/10 |
| NED | S2 | 7170 | 4877 | 4908 | 20/20 |
| CrisisMMD | S1 | 12715 | 2680 | 2687 | 5/2 |
| CrisisMMD | S2 | 9071 | 4503 | 4508 | 4/3 |

### 2. 图像与图像特征

训练读取每张图片对应的 **768 维 CLIP 全局特征**。已有 `.pt` 特征时，按以下路径放置：

```text
data/Event_data/
  NED/
    text/Qwen2-VL-7B-Instruct/
      data.csv
      train.csv   val.csv   test.csv
      train2.csv  val2.csv  test2.csv
    Images/
      E1/*.pt
      E2/*.pt
      ...
  MMD/
    text/Qwen2-VL-7B-Instruct/
      data.csv
      train.csv   val.csv   test.csv
      train2.csv  val2.csv  test2.csv
    Images/
      <事件>/<日期>/*.pt
```

`CrisisMMD` 的磁盘目录名为 `MMD`。S1 使用无数字后缀的 CSV，S2 使用后缀为 `2` 的 CSV。

- NED：`label=0,image_name=example.jpg` 对应 `NED/Images/E1/example.pt`。
- CrisisMMD：去掉 `image_name` 开头的 `data_image/`，保留事件和日期路径，将图片后缀改为 `.pt`。

若从原图提取特征，把原图放到相同位置，并保留 CSV 中记录的图片文件名，再执行：

```bash
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 python download_models.py --models image
python extract_image_features.py
```

提取器使用 `openai/clip-vit-large-patch14`、224 输入尺寸、PIL 图像预处理和 float32，保存视觉投影后的 `image_embeds`，不做额外归一化。可通过 `--datasets NED`、`--batch-size 16`、`--device cuda:0` 选择提取范围和设备；已有 `.pt` 默认复用，重新提取需显式传入 `--overwrite`。本次新生成文件的配置保存在数据目录的 `feature_extraction.json`。

不同视觉模型、预处理或数值精度得到的特征可能不同，对照实验应固定同一套特征。

## 训练

以下入口默认使用 `data/Event_data`，可统一通过 `--data-root` 指向其他目录。

```bash
# 两个数据集的全部 S1/S2 设置
python train.py --output runs

# NED 的 S1/S2
python train.py --datasets NED --output runs_ned

# CrisisMMD 的 S2
python train.py --datasets CrisisMMD --settings S2 --output runs_crisis_s2

# 四组并行训练
python train.py --jobs 4 --output runs_parallel

# 相同数据、源码和训练配置下断点续训
python train.py --output runs --resume
```

默认参数：`seed=230`、`epochs=30`、`batch_size=64`；SGD 的 `lr=0.01`、`momentum=0.9`、`weight_decay=1e-4`，梯度裁剪阈值为 `1`；对比温度 `0.07`，模态一致性损失权重 `0.3`。

每轮在验证集评估，按 `mean(ARI,NMI,AMI)` 选择最佳 epoch，平分取最早轮次；选择后再评估测试集。

```text
runs/<数据集>_<设置>/
  checkpoints/       # 每轮训练头权重
  last.pth.tar       # 完整续训状态
  selected.pth       # 验证集选出的模型
  frozen_encoder.pth # 配套冻结文本编码器
  history.json       # 损失与验证指标
  selected_test/     # 测试指标、预测和聚类中间结果
```

各组结果汇总到输出目录的 `results.csv` 和 `results.json`。新实验请指定新的 `--output`。

## 评测

```bash
python evaluate.py --checkpoint runs/NED_S1/selected.pth --dataset NED --setting S1
python evaluate.py --checkpoint runs/CrisisMMD_S2/selected.pth --dataset CrisisMMD --setting S2
python evaluate.py --checkpoint runs/NED_S1/selected.pth --dataset NED --setting S1 --split val --output eval_ned_s1_val
```

输出 ARI、NMI、AMI，并保存预测标签、特征、图、Ward 层次树和结构熵曲线。复制模型时，保留 checkpoint 引用的 `frozen_encoder.pth` 及其相对位置。

默认评测在原始预测特征上构建 Ward 层次树，使用 Pearson 1NN 与共同 hashtag/@mention 连边、正余弦相似度边权，在全部候选簇数中选择二维结构熵最小的切分。

**样本顺序和批次是评测配置的一部分。** 全局图像向量输入 CMAP 时，注意力矩阵为 `[B,B]`，特征依赖同批次样本。评测保留分割清单顺序和 `batch_size=64`；改变批次组成可能改变特征及聚类结果。这些指标应按该固定评测协议解释。

## 检查

无需下载模型或原始数据即可运行单元测试；GitHub Actions 使用 CPU 执行这些检查。

```bash
python -m unittest discover -s tests -v
```

准备好完整样本表和图像特征后，可用少量样本检查训练流程：

```bash
python train.py --datasets NED --settings S1 --epochs 1 --smoke --output runs_smoke
```

## 代码结构

- `prepare_data.py`：按固定 ID 清单恢复训练 CSV。
- `download_models.py`：准备文本和视觉编码器缓存。
- `extract_image_features.py`：提取整图 CLIP 全局特征。
- `train.py` / `evaluate.py`：训练和评测入口。
- `lfevent/model.py`：文本融合、CMAP 和损失。
- `lfevent/data.py`：数据读取。
- `lfevent/clustering.py`：图构建、Ward 和结构熵选簇。
- `lfevent/engine.py`：训练、验证、模型选择与权重加载。

## 引用

```bibtex
@inproceedings{yang2026lfevent,
  author = {Zhiwei Yang and Haimei Qin and Hao Peng and Xiaoyan Yu and Li Sun and Lei Jiang},
  title = {Label-Free Contrastive Learning for Open-World Multimodal Social Event Detection},
  booktitle = {Proceedings of the Nineteenth ACM International Conference on Web Search and Data Mining},
  year = {2026},
  doi = {10.1145/3773966.3777919}
}
```

数据及第三方模型使用说明见 [docs/DATA_USAGE.md](docs/DATA_USAGE.md)。第三方材料不因本仓库公开而改变原有许可。

## 许可证

本项目代码采用 [MIT License](LICENSE)。数据集和第三方模型遵循各自的使用条款。
