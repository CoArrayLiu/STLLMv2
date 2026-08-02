# Transformer 与自适应图消融实验

本实验使用独立文件，不修改现有的 `model_ST_LLM_plus.py` 和
`train_plus.py`：

- `model_ST_Transformer_adaptive.py`：三种空间注意力分配方式。
- `train_transformer_ablation.py`：独立训练、验证和测试入口。
- `prepare_pems08.py`：PEMS08 无泄漏时间切分与 12→12 窗口生成。
- `PEMS08_QK.md`：PEMS08 QK 适配、调优参数和实测结果。

## 三种配置

通过 `--attention_mode` 选择：

| 配置值 | 名称 | 注意力分配 |
|---|---|---|
| `qk` | QK-Transformer | `softmax(QK^T / sqrt(d))` |
| `graph` | Graph-Transformer | `A_adp @ V`，完全不包含 Q、K 投影 |
| `qk_graph` | QK+Graph-Transformer | `softmax(QK^T / sqrt(d) + alpha * log(A_adp + eps)) @ V` |

`qk_graph` 中每层有一个独立、非负、可学习的 `alpha`。初始值由
`--graph_alpha` 设置，训练过程写入 `train.csv`，最佳模型的最终值写入
`summary.json`。

模型会对自适应图做形状、有限值、非负性和行和检查，然后仅进行行归一化。
不会再次执行 softmax，也不会把稠密矩阵中的小权重变成 0。图以冻结 buffer
形式参与实验，不会在预测模型训练过程中被修改。

## 数据集和图的默认映射

| 数据集 | 节点数 | 自适应图 |
|---|---:|---|
| `bike_drop` | 250 | `adp/bd/adaptive_adj_mx.pkl` |
| `bike_pick` | 250 | `adp/bp/adaptive_adj_mx.pkl` |
| `taxi_drop` | 266 | `adp/td/adaptive_adj_mx.pkl` |
| `taxi_pick` | 266 | `adp/tp/adaptive_adj_mx.pkl` |
| `pems08` | 170 | QK 不使用图；物理/语义图仅供显式图实验 |

图模式必须显式传入 `--graph_type adaptive|physical|semantic`，可使用
`--graph_path` 覆盖配置路径；加载器按后缀区分 `.pkl/.pickle` 与 `.npy`。
`qk` 模式完全不加载或使用图，并拒绝图参数。

## 单次训练

```bash
conda run --no-capture-output -n st-llm-plus \
  python -u train_transformer_ablation.py \
  --device cuda:0 \
  --data bike_drop \
  --attention_mode qk_graph \
  --graph_type adaptive \
  --batch_size 384 \
  --eval_batch_size 64 \
  --precision bf16
```

默认骨干参数与 GPT-2 的主要尺寸对齐：

```text
d_model=768
num_heads=12
num_layers=6
ffn_dim=3072
dropout=0.1
```

显存不足时可以先用较小模型检查流程：

```bash
conda run --no-capture-output -n st-llm-plus \
  python -u train_transformer_ablation.py \
  --device cuda:0 \
  --data bike_drop \
  --attention_mode graph \
  --graph_type adaptive \
  --batch_size 8 \
  --d_model 256 \
  --num_heads 8 \
  --ffn_dim 1024
```

正式对比时，三种模式必须使用完全相同的模型尺寸、训练参数和随机种子。
`graph` 模式由于按实验定义删除了 Q、K 投影，其参数量会略少；实际参数量会写入
`summary.json`。

## A100 80GB 训练效率实测

以下是 Taxi 的 266 节点、完整 6 层模型、Ranger 优化器在空闲 A100 80GB 上的
训练核心短步结果，不包含验证、测试和日志时间：

| 模式 | 精度 | batch size | 吞吐（样本/秒） | 峰值显存 |
|---|---|---:|---:|---:|
| `qk` | BF16 | 384 | 855.3 | 35.55 GiB |
| `graph` | BF16 | 384 | 1407.7 | 21.92 GiB |
| `qk_graph` | BF16 | 384 | 843.1 | 35.64 GiB |

对于最重的 `qk_graph`，batch 64 从 FP32 改成 BF16 后，吞吐从 577.2 提升至
748.8 样本/秒，峰值显存从 9.04 降至 6.56 GiB。batch 384 和 512 的原始吞吐
接近，但训练集有 2606 个样本：batch 384 每轮为 7 step、只填充 82 个样本，
因此实际每轮比 batch 512 更快。

三个模式可以在一张 A100 上并行：batch 128 合计峰值约 32.49 GiB，batch 256
合计峰值约 62.82 GiB。不过计算资源竞争使并行的估算每轮时间约为 10.4--10.7
秒；使用 batch 384 顺序训练三个模式的估算总时间约为 8.24 秒。因此一张 A100
追求最高总效率时推荐 **BF16 + batch 384 + 三模式顺序运行**。只有必须让三个任务
同时推进时，才建议三进程使用 BF16 + batch 128。

BF16 只影响模型前向和反向；损失、MAE、RMSE 等指标仍以 FP32 计算。FP16 会自动
启用梯度缩放。TF32 默认开启，可使用 `--disable_tf32` 关闭。

验证和测试不再补齐末 batch，聚合指标按真实样本数加权。因此
`--eval_batch_size` 只影响评测吞吐与显存，不会因重复末样本改变 early stopping。

## 运行四个数据集的三种实验

建议一次只运行一个 GPU 任务。以下循环为顺序执行：

```bash
experiment_root="logs/transformer_ablation_seed6666"

for dataset in bike_drop bike_pick taxi_drop taxi_pick; do
  for mode in qk graph qk_graph; do
    graph_args=()
    if [[ "$mode" != "qk" ]]; then
      graph_args=(--graph_type adaptive)
    fi
    conda run --no-capture-output -n st-llm-plus \
      python -u train_transformer_ablation.py \
      --device cuda:0 \
      --data "$dataset" \
      --attention_mode "$mode" \
      "${graph_args[@]}" \
      --seed 6666 \
      --batch_size 384 \
      --eval_batch_size 64 \
      --precision bf16 \
      --save_dir "$experiment_root"
  done
done
```

为了报告均值和标准差，可分别运行例如 `2024、6666、2026` 三个种子，并为
每个种子指定不同的 `--save_dir`。

## 输出文件

默认输出目录为：

```text
logs/transformer_ablation_<timestamp>/<dataset>/<mode>/seed_<seed>/
```

其中包含：

- `config.json`：完整运行参数和解析后的图路径。
- `graph_stats.json`：图的形状、范围、行和、非零数和 Top-10 权重占比。
- `best_model.pth`：验证 MAE 最优模型。
- `train.csv`：逐 epoch 训练/验证指标、耗时和各层 alpha。
- `test.csv`：每个预测步及 12 步平均测试指标。
- `summary.json`：最佳 epoch、平均测试结果、参数量和最终 alpha。

若目标实验目录非空，程序会直接报错，防止覆盖已有实验结果。

