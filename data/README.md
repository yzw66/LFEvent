# 增强文本与本地原始数据

本目录提供两份固定增强文本表，分别包含 NED 的 16,955 条样本和 CrisisMMD 的 18,082 条样本：

```text
data/NED.csv
data/CrisisMMD.csv
```

表头与训练所需完整样本表一致：

```csv
sample_id,text,image_name,time,label,caption,theme,type,tags
```

公开文件仅填写 `sample_id,caption,theme,type`，`text,image_name,time,label,tags` 全部留空。增强文本取自实验使用的预生成结果，不在训练过程中重新生成。

NED 中有 6 个增强字段直接等于原始帖子文本；这些值在公开表中也留空。`text_copy_fields.json` 仅记录相应的样本 ID 和字段名，准备数据时从用户本地原文恢复，不分发这些原文。

原始数据来源及适用条款见 [数据使用说明](../docs/DATA_USAGE.md)。增强文本依赖上游图文内容；公开增强字段不意味着取得额外的数据使用或再分发权限。本项目代码的 MIT 许可证不自动适用于数据及增强文本。

## 准备本地原始字段

自行取得原始数据后，将需要的字段转换为 UTF-8 CSV，保留以下列名：

```csv
text,image_name,time,label
```

- `text`：帖子原文，精确保留空格、大小写、换行和 URL。
- `image_name`：实验使用的图片名或相对路径。NED 为文件名；CrisisMMD 为 `data_image/<事件>/<日期>/<图片名>`，同帖多图保留为不同样本。
- `time`：实验样本身份中使用的时间字符串，不能自行改写日期格式。
- `label`：从 0 开始的事件编号；NED 图片目录 `E1` 对应 0，依此类推。CrisisMMD 使用 7 个灾害事件编号，不使用原始人道主义分类标签。

也可保留额外列；脚本只合并上述原始字段，`tags` 统一从原文中的 `#hashtag` 和 `@mention` 提取。如果提供 `sample_id`，脚本会核验它与原始字段一致。

将本地 CSV 保存为：

```text
data/local/NED.csv
data/local/CrisisMMD.csv
```

然后执行：

```bash
python prepare_data.py --raw-dir data/local

# 也可使用仓库之外的本地 CSV 目录，或只准备一个数据集
python prepare_data.py --raw-dir /path/to/raw_csvs --datasets NED
```

本地 CSV 的行顺序可以任意改变，也可包含清单之外的额外样本。脚本按以下规则计算内容 ID，只选取公开表中存在的样本：

```python
identity = [dataset, row["text"], row["image_name"], row["time"]]
payload = json.dumps(identity, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
sample_id = prefix + "_" + hashlib.sha256(payload).hexdigest()
```

其中 `dataset` 为 `NED` 或 `CrisisMMD`，`prefix` 分别为 `ned` 或 `crisis`。不对字段去空格、修改大小写或标准化。下载文件若使用了不同的预处理方式，必须先转换为一致的字符串格式；这个脚本不自动适配所有上游文件格式。ID 不匹配、样本缺失、重复 ID 或分割不完整时会在写入前报错，不按行号猜测匹配关系，也不静默丢弃清单中的样本。

输出保存在 `data/Event_data/`。公开增强表和本地输入均不会被改写；`data/local/`、`data/Event_data/` 已加入 `.gitignore`，请勿将补齐后的原始数据提交至仓库。图片及其 `.pt` 特征需另外准备。

## 已有完整样本表

若已经取得与清单对应、包含原文和增强文本的完整样本表，可以放在仓库外的目录并运行：

```bash
python prepare_data.py --metadata-dir /path/to/full_metadata
```
