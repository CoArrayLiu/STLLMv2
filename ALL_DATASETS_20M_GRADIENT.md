# 10 数据集 20M 联合训练与训练梯度冲突分析

入口为 `train_all_datasets_20m_gradient_conflict.py`。它沿用项目现有的 QK Transformer：共享主干，数据集专属节点嵌入和预测头；没有加入路由、MoE 或其他新模块。

## 数据集

联合训练包含项目当前的全部 10 个数据集：

- Bike Drop、Bike Pick：各 250 节点，48 个日内时间槽
- Taxi Drop、Taxi Pick：各 266 节点，48 个日内时间槽
- PEMS03、PEMS04、PEMS08：各使用当前项目配置的 170 节点，288 个日内时间槽
- UrbanEV：275 节点，24 个日内时间槽
- Shenzhen：247 节点，288 个日内时间槽
- LargeST-SD：673 节点，288 个日内时间槽

UrbanEV、Shenzhen 和 SD 已由原始文件生成 `data/<dataset>/{train,val,test}.npz`。切分按完整天进行 60%/20%/20%，再分别构造输入/输出窗口，避免跨切分时间戳泄漏。

共享时间嵌入表有 288 个槽。24 槽和 48 槽数据分别映射到 288 槽的一小时和半小时间隔位置，不增加新的模型结构。

## 模型规模

- 总参数：19,996,800
- 共享主干参数：19,387,840
- 数据集专属参数：608,960
- `d_model=512`，8 头，6 层，`ffn_dim=2064`，节点嵌入维度 200

参数量由程序启动时强制校验，偏离目标超过 1,000 会直接报错。

## 推荐启动命令

```bash
cd /root/STLLMv2

conda run --no-capture-output -n ST-LLM \
  python -u train_all_datasets_20m_gradient_conflict.py \
  --device cuda:0 \
  --precision bf16 \
  --epochs 100 \
  --min_epochs 15 \
  --es_patience 12 \
  --batch_size 192 \
  --eval_batch_size 256 \
  --gradient_probe_batches 8
```

在 A100 80 GiB 上，真实的 10 数据集、20M 模型、45 个任务对梯度探测中，`batch_size=192` 的实测峰值约 54.1 GiB；256 约 71.8 GiB，余量较小，因此推荐 192。

脚本会按训练样本量为各任务自动设置实际 batch size，使各数据集每个 epoch 都大致遍历一次，避免小数据集因 SD 较大而被严重过采样。`--batch_size 192` 是最大数据集 SD 的 batch size，不代表所有任务都固定使用 192。

## 梯度记录含义

入口强制 `gradient_probe_every=1`，所以每个 epoch 都会记录。`--gradient_probe_batches 8` 表示在每轮训练循环内均匀选择 8 个真实 optimizer step：

1. 使用该 step 各数据集的实际训练 batch 和同一次 forward 图；
2. 分别取得 10 个标准化 MAE 对共享参数产生的梯度；
3. 对全部 45 个数据集对、40 个模块组计算 cosine、符号冲突、冲突质量和 tug-of-war strength；
4. 随后仍使用同一批损失执行正常反向传播和参数更新。

因此这些是训练时梯度，不是验证集梯度。每轮 8 次探测会产生 `8 × 45 × 40 = 14,400` 条原始冲突记录。若希望每个训练 step 都探测，可把 `--gradient_probe_batches` 设为不小于每轮 step 数，但训练会明显变慢。

主要输出位于自动创建的 `logs/all10_20m_gradient_conflict_<时间>/`：

- `training_gradient_conflicts.csv`：每个探测 step、任务对、模块的原始冲突数据
- `training_gradient_task_norms.csv`：每个探测 step、任务、模块的梯度范数
- `gradient_conflict_epoch_summary.csv`：逐 epoch、逐模块汇总
- `gradient_conflict_pair_epoch_summary.csv`：逐 epoch、逐任务对、逐模块汇总
- `gradient_conflict_summary.csv`：跨全部已训练 epoch 的模块排序
- `gradient_conflict_pair_summary.csv`：跨 epoch 的任务对/模块排序
- `gradient_task_norm_summary.csv`：任务梯度范数汇总
- `summary.json`：参数量、批次配置、最佳轮次和总体最强冲突

频率解释：`pairwise_conflict_rate` 是全部任务对比较中 cosine 小于 0 的比例；`any_pair_conflict_rate` 是探测点中该模块至少有一对发生冲突的比例；`mean_tug_of_war_strength` 同时考虑方向相反程度和两侧梯度幅度平衡，比只看负 cosine 更适合判断实际拉扯强度。

## 测试

```bash
cd /root/STLLMv2
conda run --no-capture-output -n ST-LLM \
  python -m unittest \
  test_all_datasets_20m_gradient_conflict.py \
  test_pems03_04_08_gradient_conflict.py \
  test_pems_joint_gradient_conflict.py
```

