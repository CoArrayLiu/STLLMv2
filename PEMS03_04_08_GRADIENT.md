# PEMS03/04/08 三数据集联合训练与训练梯度冲突分析

## 模型与探测语义

入口：`train_pems03_04_08_gradient_conflict.py`

- 共享 QK Transformer 骨干：7,948,696 参数
- 三套数据集专属节点嵌入与预测头：76,836 参数
- 总参数量：8,025,532（约 8.03M）
- 联合损失为三个数据集标准化 MAE 的等权平均。

梯度探测发生在真实训练循环内部。对一个被选中的 optimizer step，程序使用该 step 的完整训练 batch、同一次前向传播和同一组 dropout 掩码，分别提取 PEMS03、PEMS04、PEMS08 对共享参数的未缩放损失梯度，然后继续执行原本的联合反向传播、梯度裁剪和参数更新。探测不会替换或修改实际联合梯度。

## 推荐启动命令

下面配置适合 80 GiB A100。第 1 个 epoch 一定探测，之后每 10 个 epoch 探测一次；每个探测 epoch 从训练过程中均匀选择 8 个真实 optimizer step：

```bash
cd /root/STLLMv2

conda run --no-capture-output -n ST-LLM \
  python -u train_pems03_04_08_gradient_conflict.py \
  --device cuda:0 \
  --precision bf16 \
  --epochs 60 \
  --min_epochs 15 \
  --es_patience 12 \
  --batch_size 2048 \
  --eval_batch_size 8192 \
  --gradient_probe_every 10 \
  --gradient_probe_batches 8
```

`gradient_probe_batches` 现在表示每个探测 epoch 选择多少个训练 step，而不是验证集 batch 数。`gradient_batch_size` 仅为旧命令兼容而保留，真实梯度探测始终使用 optimizer 实际看到的完整训练 batch。

如果显存不足，把 `batch_size` 依次降到 1536、1024 或 512。探测 step 会多做一次任务独立梯度求导，因此探测 epoch 会比普通 epoch 慢；非探测 epoch 没有这部分开销。

## 拉扯指标

每个共享模块分别分析 P03–P04、P03–P08、P04–P08：

- `cosine`：梯度余弦；小于 0 表示两个任务总体方向冲突。
- `negative_cosine`：本次比较是否冲突。
- `sign_conflict_rate`：两个任务梯度符号相反的活跃参数比例。
- `conflict_mass`：负点积绝对质量占全部点积绝对质量的比例。
- `tug_of_war_strength`：定义为 `1 - ||ga + gb|| / (||ga|| + ||gb||)`。接近 0 表示方向一致，接近 1 表示大小相近且方向相反、更新被严重抵消。
- `task_a_grad_norm`、`task_b_grad_norm`：两个数据集在该共享模块上的梯度大小，用来判断谁的影响更强。

## 输出文件

- `training_gradient_conflicts.csv`：每个真实训练探测 step、模块和任务对的原始统计，包含 epoch、epoch_step 和 global_step。
- `training_gradient_task_norms.csv`：每个探测 step 中，各数据集在各共享模块上的独立梯度范数。
- `gradient_conflict_summary.csv`：整个训练过程按模块汇总。
  - `any_pair_conflict_rate`：一次探测中任意任务对冲突的频率。
  - `pairwise_conflict_rate`：三个任务对全部比较中的冲突频率。
  - `mean_tug_of_war_strength` / `max_tug_of_war_strength`：平均和最大拉扯程度。
- `gradient_conflict_pair_summary.csv`：按模块和具体任务对汇总。
- `gradient_conflict_epoch_summary.csv`：按 epoch 和模块汇总，用于观察冲突随训练变化。
- `gradient_conflict_pair_epoch_summary.csv`：按 epoch、模块和任务对汇总。
- `gradient_task_norm_summary.csv`：按模块和数据集汇总梯度范数。
- `summary.json`：记录冲突频率最高、平均余弦最低和拉扯强度最高的模块。

分析范围是多个数据集共同更新的参数：输入值投影、时间嵌入、融合投影、每层 Q/K/V/输出投影、FFN、归一化和最终归一化。数据集专属节点嵌入与预测头没有共同参数，因此不定义跨任务梯度冲突。

## 频率分母示例

使用 8 个探测 step 时：

- 某个 epoch 内，一个具体任务对的频率分母为 8，分辨率为 12.5%。
- 某个 epoch 内，模块综合 `pairwise_conflict_rate` 的分母为 `8 × 3 = 24`。
- 训练 60 epoch、每 10 epoch 探测并包含 epoch 1 时，正常情况下共探测 7 个 epoch，即每个任务对最多 56 次观测；早停时会相应减少。

