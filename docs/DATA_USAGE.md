# 数据来源和使用说明

本仓库的代码、分割清单、第三方原始数据和预训练模型是不同的材料，需分别遵循其适用条款；代码公开不表示重新授予第三方数据的使用或分发权限。

## NED

- 数据来源：[RetrainIt/NED](https://github.com/RetrainIt/NED)。
- 对应研究：Zehang Lin, Jiayuan Xie, Qing Li, *Multi-modal news event detection with external knowledge*。
- 上游 README 声明研究用途及商业使用限制；使用和分发前请阅读上游说明及其 LICENSE，不将本项目的代码许可自动套用于原始数据。
- 本项目固定样本表包含 16,955 条样本、40 个事件；上游介绍的总量为 17,366，获取数据时需核对本项目清单是否全部匹配。

## CrisisMMD

- 数据来源：[CrisisMMD 官方页面](https://crisisnlp.qcri.org/crisismmd)。
- [CrisisNLP 使用条款](https://crisisnlp.qcri.org/terms-of-use.html)要求将数据内容保密，并允许分享 tweet IDs。本公开包不包含原始帖子全文 CSV。
- 上游 v2.0 包含 16,058 条不同帖子及 18,082 张图像；本项目按图文样本计数，共 18,082 行、7 个灾害事件。
- 使用该数据集时，请遵循官方页面列出的引用要求。事件标签在本项目中用于划分和评测，并非其原始人道主义分类标签。

## 固定分割

`splits/<dataset>/<setting>/{train,val,test}.csv` 仅包含 `order,sample_id`。
`order` 从 0 连续递增。完整样本表中的 `sample_id` 是稳定哈希：

```python
identity = [dataset, row["text"], row["image_name"], row["time"]]
payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
sample_id = prefix + "_" + hashlib.sha256(payload).hexdigest()
```

其中数据集名为 `NED` 或 `CrisisMMD`，前缀分别为 `ned` 或 `crisis`。
不对文本去空格、大小写转换或重新标准化时间。重复图片、同帖多图均不能仅凭图片名或帖子 ID 合并为一条样本。

清单用于恢复固定的样本与行顺序。缺失样本、重复 ID 或未覆盖完整样本表时，准备脚本会报错，避免静默改变实验数据。
