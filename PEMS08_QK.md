# PEMS08 QK-Transformer 适配与调优

## 数据准备

原始数据 `data/st_data/pems08/pems08.npz` 的形状为 `[17856, 170, 1]`。
执行：

```bash
conda run --no-capture-output -n st-llm-plus \
  python -u prepare_pems08.py
```

预处理先按完整自然日切分原始时间轴，再在各分区内部独立构造 12 步输入和
12 步目标，避免任何输入或目标时间点跨分区复用：

| 分区 | 原始日期 | 天数 | 样本数 |
|---|---|---:|---:|
| train | 2016-07-01 至 2016-08-06 | 37 | 10633 |
| val | 2016-08-07 至 2016-08-18 | 12 | 3433 |
| test | 2016-08-19 至 2016-08-31 | 13 | 3721 |

生成的 `x` 为 `[samples, 12, 170, 3]`，三个通道依次为流量、归一化日内
位置和星期；`y` 为 `[samples, 12, 170, 1]`。`manifest.json` 记录源文件
哈希、边界、形状和缩放往返误差，`baselines.json` 记录 persistence 与训练
节点均值基线。

## QK 模式

QK 模式完全不加载 `pems08_adj.npy` 或 `cached_dist_matrix.npy`。如果同时
传入 `--graph_type` 或 `--graph_path`，入口会报错，避免误以为实验使用了图。

推荐的 A100 配置：

```bash
conda run --no-capture-output -n st-llm-plus \
  python -u train_transformer_ablation.py \
  --device cuda:0 \
  --data pems08 \
  --attention_mode qk \
  --precision bf16 \
  --batch_size 512 \
  --eval_batch_size 256 \
  --epochs 60 \
  --min_epochs 15 \
  --es_patience 12 \
  --lrate 1e-3 \
  --grad_clip 10 \
  --lr_scheduler plateau \
  --lr_patience 3 \
  --lr_factor 0.5 \
  --min_lrate 1e-5 \
  --save_dir logs/pems08_qk_tuned
```

PEMS08 默认使用 `loss_space=standardized`：反向 MAE 除以训练集标准差，但
所有日志、早停和测试指标仍使用原始流量单位。该变换仅给目标函数乘正常数，
不改变最优解，可避免 PEMS08 较大的数值尺度导致每步都触发梯度裁剪。

验证和测试 DataLoader 不进行末样本补齐，指标按 batch 的真实样本数加权；
训练 DataLoader 保留补齐，并在每个 epoch 重新洗牌。

## 调优结果

全尺寸 6 层模型的单轮 batch 测试如下。吞吐受同卡其他进程影响，峰值显存是
当前训练进程的 PyTorch allocated memory：

| batch | 学习率 | 吞吐（样本/秒） | 峰值显存 | 结论 |
|---:|---:|---:|---:|---|
| 384 | 3e-4 | 472 | 20.07 GiB | 更新步数多，但吞吐较低 |
| 512 | 3e-4 | 543 | 26.51 GiB | 吞吐、显存和填充量均衡 |
| 640 | 4e-4 | 591 | 32.95 GiB | 边际提速，更新步数少且余量较小 |

学习率短试验中，`1e-3` 的前 5 轮验证 MAE 单调下降到 25.14；`2e-3`
在第 4 轮反弹，因此选择 Ranger `lr=1e-3`。`grad_clip=10` 仅在早期少量
step 触发，稳定阶段裁剪率为 0%。Plateau 调度器在第 27 轮将学习率降为
`5e-4`，验证 MAE 随后继续改善。

最终 seed 6666、60 轮运行的最佳验证 checkpoint 位于第 58 轮：

| 结果 | MAE | RMSE | MAPE | WMAPE |
|---|---:|---:|---:|---:|
| persistence test baseline | 25.3756 | 37.9981 | 0.1557 | 0.1090 |
| QK-Transformer test | 16.5136 | 25.9672 | 0.1201 | 0.0709 |

QK 模型测试 MAE 相对 persistence 降低约 34.9%。训练产物位于
`logs/pems08_qk_tuned/pems08/qk/seed_6666/`。

## 烟测

```bash
conda run --no-capture-output -n st-llm-plus \
  python -m unittest -v test_pems08_adaptation.py
```

烟测覆盖数据集配置、无泄漏边界、窗口与原始值对齐、时间特征、缩放器、
QK 禁用图约束、`.npy` 图加载和 QK 前向形状。
