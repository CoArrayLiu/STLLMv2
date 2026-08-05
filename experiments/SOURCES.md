# 原始来源与产物索引

| 实验 | 主要来源 | 原始产物状态 |
|---|---|---|
| E01 基线复现 | `../baseline.md` | 汇总文档存在；本次运行的逐 epoch 日志未在当前 `logs/` 中发现 |
| E02 注意力模式消融 | `../result.md`、`../TRANSFORMER_ABLATIONS.md` | 汇总文档存在；对应完整训练目录未在当前 `logs/` 中发现 |
| E03 PEMS08 QK | `../PEMS08_QK.md` | 汇总文档存在；文档所述 `logs/pems08_qk_tuned/...` 当前未发现 |
| E04 单任务注意力分析 | `../logs/qk_attention_analysis_seed6666/SUMMARY.md` | 完整；含四数据集 summary、CSV、NPZ、PNG/PDF |
| E05 联合训练 | `../logs/qk_joint_separate_heads_seed6666/` | 完整；含 config、train/test CSV、summary、checkpoint |
| E05 联合注意力分析 | `../logs/qk_joint_attention_seed6666_n64/` | 完整；含分析文档、CSV、NPZ、PNG |
| E06 减层消融 | `../logs/qk_depth_ablation_seed6666/` | 完整 24/24；manifest 状态为 `complete` |
| E07 加层扩展 | `../logs/qk_depth_expansion_seed6666/` | 未完成；manifest 状态仍为 `running`，但当前无相关进程 |

## 关键代码入口

| 用途 | 文件 |
|---|---|
| ST-LLM+ 原始训练 | `../train_plus.py` |
| 三种注意力模式训练 | `../train_transformer_ablation.py` |
| 联合 QK 训练 | `../train_qk_joint.py` |
| 单任务注意力分析 | `../analyze_topk10_ablation.py` |
| 联合注意力分析 | `../analyze_joint_qk_attention.py` |
| 减层训练 / 套件 | `../train_qk_depth_ablation.py`、`../run_qk_depth_ablation_suite.py` |
| 加层训练 / 套件 | `../train_qk_depth_expansion.py`、`../run_qk_depth_expansion_suite.py` |
| PEMS08 准备与验证 | `../prepare_pems08.py`、`../test_pems08_adaptation.py` |

## 状态判定说明

状态基于 2026-08-05 的文件系统快照。加层套件计划 6 个变体 × 4 个数据集，共 24 项：13 项存在 `summary.json`，`add_last_2_matched/taxi_pick` 只有训练到 epoch 113 的 `train.csv`，其余 10 项没有运行目录。由于 manifest 没有在中断后收尾，其 `running` 字段不能解释为后台仍在运行。
