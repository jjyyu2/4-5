# 保险代理人线索点击与转化预测项目

本项目提供一个完整的保险领域线索预测流程，覆盖数据分析（EDA）与建模训练两部分。脚本会对数据进行缺失值统计、数值字段描述统计，并分别训练“代理人点击用户卡片线索”与“点击后用户购买转化”两个目标的模型。

## 数据要求

- 输入为 CSV 文件。
- 必须包含两列目标字段（可通过参数自定义）：
  - `clicked`：代理人是否点击用户卡片线索（默认）
  - `purchased`：点击后是否购买（默认）
- 其余字段将作为特征（客户信息 + 代理人信息）。
- 可通过 `--id-columns` 指定不参与建模的标识列。

## 快速开始

```bash
python insurance_agent_lead_model.py \
  --data /path/to/insurance_leads.csv \
  --click-target clicked \
  --purchase-target purchased \
  --id-columns user_id agent_id
```

## 输出内容

脚本默认输出到 `reports/` 目录：

- `eda_summary.md`：EDA 概览
- `missingness.csv`：字段缺失率
- `numeric_stats.csv`：数值字段描述统计
- `metrics_<target>.json`：单任务模型指标
- `metrics_summary.json`：汇总指标

## 模型说明

- 默认模型：`LogisticRegression`（类不平衡时使用 `class_weight=balanced`）。
- 可选模型：`RandomForestClassifier`（`--model random_forest`）。
- 预处理：
  - 数值字段：缺失值填充 + 标准化
  - 类别字段：缺失值填充 + One-Hot 编码

## 评估指标

- ROC-AUC
- Accuracy
- Precision
- Recall
- F1
