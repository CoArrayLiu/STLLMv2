# 实验记录索引

本目录汇总截至 **2026-08-05** 仓库中可确认的实验记录和结果。训练日志、checkpoint、逐步指标及分析图片仍保留在 `../logs/`；这里保存便于阅读的实验卡片、统一结果表和来源索引，不重复存放大体积模型文件。

## 总览

| 编号 | 实验 | 状态 | 数据集 | 主要结论 | 记录 |
|---|---|---|---|---|---|
| E01 | ST-LLM+ 基线复现 | 已完成（单种子） | Taxi/Bike 四任务 | 本地 MAE 比论文值高约 1.8%–5.3% | [E01](records/E01_baseline.md) |
| E02 | QK / Graph / QK+Graph 注意力模式消融 | 已完成（汇总结果可用） | Taxi/Bike 四任务 | QK 普遍优于 ST-LLM+；图先验收益依赖数据集 | [E02](records/E02_attention_mode_ablation.md) |
| E03 | PEMS08 QK 适配与调优 | 已完成（文档记录可用） | PEMS08 | 测试 MAE 16.5136，较 persistence 降低 34.9% | [E03](records/E03_pems08_qk.md) |
| E04 | 单任务 QK 逐层注意力分析 | 已完成 | Taxi/Bike 四任务 | 深层注意力趋于均匀；整体未直接复现自适应图 | [E04](records/E04_attention_analysis.md) |
| E05 | 四任务联合 QK 训练与注意力分析 | 已完成（单种子） | Taxi/Bike 四任务 | 宏平均 MAE 相对变化 +0.42%，存在轻微负迁移 | [E05](records/E05_joint_training.md) |
| E06 | QK 减层消融 | 已完成 24/24（单种子） | Taxi/Bike 四任务 | 3–5 层可显著省资源，但平均 MAE 比 6 层高约 1.6%–2.3% | [E06](records/E06_depth_ablation.md) |
| E07 | QK 加层扩展 | **未完成：13/24 完成** | Taxi/Bike 四任务 | 当前已完成子集未显示稳定收益，不能作完整结论 | [E07](records/E07_depth_expansion.md) |

## 统一结果表

- [核心预测结果](results/core_metrics.csv)：基线、注意力模式、PEMS08 和联合训练的主要测试指标。
- [减层汇总](results/depth_ablation_summary.csv)：每个减层变体的完成数、相对 MAE、资源指标。
- [加层汇总](results/depth_expansion_partial_summary.csv)：仅汇总已完成的 13 个任务，并明确样本不完整。
- [来源清单](SOURCES.md)：整理结果与原始日志之间的对应关系。

## 统一口径与限制

- Taxi/Bike 表中的 MAE、RMSE、WAPE/WMAPE 均为 12 个预测步平均值，越低越好。
- 深度实验的“宏平均 MAE 变化率”先相对各数据集的 6 层单训 QK 基准计算变化率，再对四个任务等权平均；正值表示变差。
- 现有正式结果均只确认了 seed `6666`，没有均值、标准差或显著性检验，因此结论应视为趋势。
- `E02` 与 `E03` 的完整训练目录当前不在 `logs/` 中，指标来自版本化 Markdown 记录；其余实验可追溯到现存日志和配置。
- `logs.zip` 是日志归档，未在本次整理中解压或改动；当前目录依据已解压的 `logs/` 和仓库文档生成。
