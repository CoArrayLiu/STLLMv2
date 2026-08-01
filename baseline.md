# Baseline

以下结果均为 12 个预测步的平均值，指标越低越好。

## 本次复现最佳结果

| 数据集 | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| NYCTaxi Pick-up | 5.3259 | 9.3011 | 20.15% |
| NYCTaxi Drop-off | 5.2032 | 9.1257 | 19.86% |
| CHBike Pick-up | 2.0152 | 3.1371 | 40.59% |
| CHBike Drop-off | 1.9146 | 2.8702 | 38.71% |

## 论文报告结果

| 数据集 | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| NYCTaxi Pick-up | 5.18 | 8.98 | 19.60% |
| NYCTaxi Drop-off | 4.94 | 8.68 | 18.86% |
| CHBike Pick-up | 1.98 | 3.05 | 40.11% |
| CHBike Drop-off | 1.88 | 2.79 | 38.20% |

论文结果来源：[ST-LLM+: Graph Enhanced Spatio-Temporal Large Language Models for Traffic Prediction, Table III](https://doi.org/10.1109/TKDE.2025.3570705)。
