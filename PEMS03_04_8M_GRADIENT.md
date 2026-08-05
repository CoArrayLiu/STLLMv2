# PEMS03/PEMS04 8M 联合模型与梯度冲突分析

## 模型设计

模型沿用当前项目的 QK-Transformer：

- 6 个 Transformer 层
- `d_model=320`
- 8 个注意力头
- `ffn_dim=1380`
- `embedding_dim=128`
- PEMS03/PEMS04 共享输入编码、时间编码、Transformer 和最终归一化层
- 两个数据集分别使用节点嵌入和预测头

PyTorch 实例化统计：

- 总参数：7,999,920
- 共享参数：7,948,696
- 数据集专属参数：51,224
- 可训练参数：7,999,920

## 数据协议

使用仓库已有的 170 节点版本：

- `data/st_data/pems03/pems03_clip.npz`
- `data/st_data/pems04/pems04_clip.npz`

按完整自然日先做 60%/20%/20% 时间切分，再在各分区内部构造
12 步输入到 12 步输出的窗口，避免输入或目标时间点跨分区复用。

若 `data/pems03` 和 `data/pems04` 尚未生成：

```bash
conda run --no-capture-output -n ST-LLM \
  python prepare_pems08.py \
  --source data/st_data/pems03/pems03_clip.npz \
  --output_dir data/pems03 \
  --start_time 2018-09-01T00:00:00

conda run --no-capture-output -n ST-LLM \
  python prepare_pems08.py \
  --source data/st_data/pems04/pems04_clip.npz \
  --output_dir data/pems04 \
  --start_time 2018-01-01T00:00:00
```

生成的窗口数量：

| 数据集 | Train | Validation | Test |
|---|---:|---:|---:|
| PEMS03 | 15,529 | 5,161 | 5,449 |
| PEMS04 | 10,057 | 3,145 | 3,721 |

## 正式训练

省略 `--save_dir` 时，每次会自动创建新的时间戳目录，不会覆盖旧实验。

```bash
conda run --no-capture-output -n ST-LLM \
  python -u train_pems_joint_gradient_conflict.py \
  --device cuda:0 \
  --precision bf16 \
  --epochs 60 \
  --min_epochs 15 \
  --es_patience 12 \
  --batch_size 512 \
  --eval_batch_size 512 \
  --gradient_batch_size 64 \
  --gradient_probe_every 5 \
  --gradient_probe_batches 2
```

两个任务的损失均为原始 MAE 除以各自训练集标准差，然后等权平均。
每轮以较长的 PEMS03 为准；PEMS04 数据加载器耗尽后重新打乱并循环，
保证每个优化步包含两个任务且不会丢弃 PEMS03 样本。

## 测试

```bash
conda run --no-capture-output -n ST-LLM \
  python -m unittest -v test_pems_joint_gradient_conflict.py
```

快速真实数据冒烟命令：

```bash
conda run --no-capture-output -n ST-LLM \
  python -u train_pems_joint_gradient_conflict.py \
  --device cuda:0 \
  --precision bf16 \
  --epochs 1 \
  --min_epochs 0 \
  --es_patience 1 \
  --batch_size 64 \
  --eval_batch_size 512 \
  --gradient_batch_size 16 \
  --gradient_probe_batches 1 \
  --max_steps_per_epoch 2
```

## 梯度冲突定义与输出

只比较两个任务在共享参数上的梯度。数据集专属节点嵌入和预测头没有共享语义，
因此不纳入冲突排名。

每个模块记录：

- `cosine`：两个任务梯度的余弦相似度，越小冲突越强，负值表示方向相反。
- `negative_cosine_rate`：探针批次中余弦为负的比例。
- `sign_conflict_rate`：两个梯度逐元素符号相反的比例。
- `conflict_mass`：负梯度乘积绝对量占全部梯度乘积绝对量的比例。
- 两个任务各自的梯度范数。

输出目录包含：

- `best_model.pth`：宏平均标准化验证 MAE 最优的 checkpoint。
- `train.csv`：逐轮训练和验证指标。
- `gradient_conflicts.csv`：训练期间的逐批次、逐模块冲突值。
- `gradient_conflicts_best.csv`：最佳 checkpoint 的原始探针结果。
- `gradient_conflict_summary.csv`：最佳 checkpoint 的聚合排名，默认按平均余弦从小到大排列。
- `test_pems03.csv`、`test_pems04.csv`：逐预测步和平均测试指标。
- `summary.json`：完整摘要；`strongest_conflict_by_mean_cosine` 即冲突最大的共享模块。

