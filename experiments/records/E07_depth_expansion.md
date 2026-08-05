# E07：QK 加层扩展（未完成）

## 目的与设计

在 6 层 QK 基准上增加最后 1/2/3 层：

- `unmatched`：保持 d_model=768、FFN=3072，参数量随深度增加。
- `matched`：缩小宽度，使参数量约等于 6 层基准。

计划覆盖 6 个变体 × 4 个数据集，共 24 项；训练配置为 seed 6666、BF16、最多 300 epochs、最少 200 epochs、Ranger `lr=1e-3`、无学习率调度。

## 完成状态（2026-08-05 快照）

- 已完成并写入 `summary.json`：13/24。
- `add_last_2_matched/taxi_pick`：仅运行到 epoch 113，没有 summary/test，视为中断。
- 其余 10 项没有运行目录；`add_last_3_*` 全部尚未产出结果。
- manifest 状态仍为 `running`，但检查时未发现相关训练进程。

| 变体 | 已完成/计划 | 参数比 | 已完成子集 MAE 宏平均变化 | 平均吞吐（样本/s） | 平均峰值显存（GiB） |
|---|---:|---:|---:|---:|---:|
| add_last_1_unmatched | 4/4 | 1.164 | +0.893% | 725.0 | 53.46 |
| add_last_1_matched | 4/4 | 1.000 | +1.264% | 740.9 | 50.80 |
| add_last_2_unmatched | 4/4 | 1.328 | +1.103% | 632.4 | 60.73 |
| add_last_2_matched | 1/4 | 0.999 | +1.508%* | 657.8 | 57.73 |
| add_last_3_unmatched | 0/4 | — | — | — | — |
| add_last_3_matched | 0/4 | — | — | — | — |

`*` 仅来自 taxi_drop，不能与四任务宏平均直接比较。

## 暂定观察

三个已完整覆盖四任务的 7/8 层变体，其宏平均 MAE 均比 6 层基准高 0.9%–1.3%，同时吞吐下降、显存增加。现有数据未显示加深带来稳定收益，但由于套件未完成且只有一个随机种子，只能报告“当前子集未见收益”，不能对全部加层方案下最终结论。

完整已完成行：`../logs/qk_depth_expansion_seed6666/suite_results.csv`；计划与命令：`../logs/qk_depth_expansion_seed6666/suite_manifest.json`。
