# 样本表

将按数据提供方条款取得的完整样本表放到本目录：

```text
data/
  NED.csv
  CrisisMMD.csv
```

完整样本表与固定分割清单对应，字段为：

```csv
sample_id,text,image_name,time,label,caption,theme,type,tags
```

- `sample_id` 由 `prepare_data.sample_id()` 计算，依次使用数据集名、`text`、`image_name`、`time` 的原始字符串值。
- `label` 为从 0 开始的事件编号。
- `caption/theme/type` 是预生成的增强文本。
- `tags` 保留样本附带标记。

本公开仓库提供分割 ID，不直接托管原始帖子全文。请遵循[数据来源和使用说明](../docs/DATA_USAGE.md)获取可使用的数据；需要与本项目完全对应的预处理样本表时，请通过仓库 Issues 联系维护者，按数据提供方授权方式获取。

取得完整样本表后执行：

```bash
python prepare_data.py
```

默认生成 `data/Event_data/` 下的训练、验证和测试 CSV。此目录及本目录中的完整 CSV 均被 `.gitignore` 排除。图像和 `.pt` 特征需另外准备。
