# SmolVLA 在 LIBERO 中改变每次预测后的动作执行长度 K

**状态：1200/1200 episode 已完成。**这是对 GitHub Pages 单次演示的统计扩展，仍是三个固定任务上的探索结果，不是 LIBERO 四套件正式基准。

## 对照设置

固定公开 `smolvla_libero_paper` checkpoint（SHA256 `71d9563c8295284acba8fc2d5c19de000d6fe9ba58a406832af7ef3d221ed52f`）、LeRobot commit `8ae2354592ec4480c38134fb75350b8e7992e0e9`、`chunk_size=50`、flow sampling `num_steps=10`、相对控制、256×256 双相机输入；只改变 `n_action_steps`，即 K=1/5/10/20。注意这次改变的是每次预测后从队列执行多少步，并没有改变 checkpoint 的 50 步预测块长度。

选择三个任务：Spatial task 0（把黑碗放到盘子上）、Object task 0（把字母汤罐放进篮子）、Goal task 0（打开柜子中间抽屉）。每个任务与 K 跑 100 个完整 episode，总计 1200 个。trial 0–99 对应 seed 1000–1099；每个任务的 50 个初态各跑两次。相同任务下，不同 K 使用相同 trial、seed 和初态，形成配对条件。每 episode 开始时重置随机种子、环境和 LeRobot 动作队列。使用 rollout 返回的 LIBERO `success` 判定。未录视频。

## 结果

延迟为 100 个 episode 的算术平均值。`推理累计时间`只累加 SmolVLA 生成动作块的 `_get_action_chunk`；调用前后执行 CUDA synchronize。`完整 rollout 时间`还包括图像/状态预处理、环境仿真、Python 循环等，不包含录像编码。两种时间均是本机同步仿真耗时。

| 任务 | K | 成功率 | 平均模型调用/episode | 平均推理累计时间 | 平均完整 rollout 时间 |
|---|---:|---:|---:|---:|---:|
| Spatial task 0 | 1 | 68/100 (68%) | 141.72 | 48.53 s | 52.57 s |
| Spatial task 0 | 5 | 70/100 (70%) | 28.37 | 9.80 s | 13.80 s |
| Spatial task 0 | 10 | 72/100 (72%) | 14.29 | 4.99 s | 9.04 s |
| Spatial task 0 | 20 | 74/100 (74%) | 6.80 | 2.37 s | 6.20 s |
| Object task 0 | 1 | 78/100 (78%) | 189.25 | 64.58 s | 68.64 s |
| Object task 0 | 5 | 87/100 (87%) | 33.64 | 11.55 s | 15.15 s |
| Object task 0 | 10 | 90/100 (90%) | 16.43 | 5.66 s | 9.20 s |
| Object task 0 | 20 | 92/100 (92%) | 8.20 | 2.83 s | 6.29 s |
| Goal task 0 | 1 | 72/100 (72%) | 189.56 | 64.98 s | 69.01 s |
| Goal task 0 | 5 | 68/100 (68%) | 39.67 | 13.60 s | 17.71 s |
| Goal task 0 | 10 | 65/100 (65%) | 20.30 | 7.01 s | 11.16 s |
| Goal task 0 | 20 | 63/100 (63%) | 10.38 | 3.57 s | 7.77 s |

## 如何解读

- K 从 1 增至 20 后，三个任务的平均推理累计时间都降低约 18–23 倍，平均实际模型调用数也显著减少。平均单次动作块调用耗时仍约 0.34 秒，所以主要区别来自每个 episode 调用多少次模型。
- 成功率趋势依任务而异：Spatial 从 68% 升至 74%，Object 从 78% 升至 92%，Goal 从 72% 降至 63%。较长的动作执行间隔在前两个任务的这批初态上表现较好，在打开抽屉任务上则较差。
- 因而不能说 K 越大普遍越好。K 是计算量与重新观察/规划频率之间的控制取舍；任务动作是否需要频繁修正也会影响成功率。
- 每格 100 次提供了比单条展示视频更有用的估计，但任务仅各选一个 task，且不同 K 的成功率差异仍有抽样不确定性。本结果不能代表整个 LIBERO suite，也不推断真机控制周期或真实机器人效果。

## 原始记录与环境

被 Git 忽略的本机目录：`runs/action-horizon-success-rate/20261121-100ep-timed/`。

- `episodes.jsonl`：1200 条逐 episode 记录，包括任务、trial、seed、初态、K、成功、环境步数、实际模型调用次数、逐次推理耗时数组、推理累计时间、调用耗时均值/p50/p95/首调用耗时和完整 rollout 时间。
- `summary.json`：按任务与 K 汇总；`complete=true`，各格 `completed=100`。
- `protocol.json`、`hardware.json`：固定参数、模型 hash 与运行环境。
- GPU：NVIDIA GeForce RTX 3090；PyTorch 2.11.0+cu128，CUDA runtime 12.8，Transformers 5.5.4，Python 3.12.13。

逐 episode 结果使用 `src/vla_inference_engine/benchmark/action_horizon_success_rate.py` 收集；在仓库根目录通过 `bash scripts/action_horizon_success_rate.sh` 可复跑。采集入口、种子/初态配对方式和计时边界见[实验运行手册](../../runbooks/action-horizon-success-rate.zh-CN.md)。该实验与正式双模型基线分开保存。
