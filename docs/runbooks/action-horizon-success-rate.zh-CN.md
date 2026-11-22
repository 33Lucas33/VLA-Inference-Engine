# K 值成功率与推理时间实验

从仓库根目录运行 `bash scripts/action_horizon_success_rate.sh`。脚本沿用展示页的公开 SmolVLA checkpoint、Spatial/Object/Goal 各 task 0、`chunk_size=50`、10 步 flow sampling、256×256 双相机和 `relative` 控制；只改变 `n_action_steps`，即 K=1/5/10/20。每个任务与 K 跑 100 个完整 episode，共 1200 个。与先前的 12 条展示视频分开保存，不覆盖正式双模型基线。

2026-11-22 当前这组 1200 episode 已完成。运行期间的 `summary.json` 可读取进度；如果运行中断，用同一脚本和输出目录重启会跳过已完成的 `(task, trial, K)`。现成结果汇总见[实验报告](../experiments/action_horizon_success_rate_20261121/REPORT.zh-CN.md)。

同一任务中，第 `i` 次 trial 对四个 K 使用相同的环境 seed `1000+i` 和初态编号 `i % 该任务初态总数`；实际总数逐条写在 `init_state_count`。若总数为 50，则每个初态出现两次，第二次使用不同 seed。四组 K 在同一批 100 个条件下配对。每个 episode 开始前重设随机数生成器、动作队列和环境；使用固定 LeRobot `rollout` 的 `success` 判定。该实验不录制视频，以免编码开销影响计时。

原始结果逐条追加至被 Git 忽略的 `runs/action-horizon-success-rate/20261121-100ep-timed/episodes.jsonl`；中断后用同一脚本与输出目录重启，会跳过已经完成的 `(task, trial, K)`。`protocol.json` 固定条件与模型 SHA256，`hardware.json` 保存 GPU、CUDA、PyTorch、Transformers、Python 和固定 LeRobot commit，`summary.json` 随每个 episode 更新。只有 `summary.json` 的 `complete=true` 且每格 `completed=100` 时，才把成功率写成最终结果。

每条原始记录包含成功、环境步数、**实际**动作块推理调用次数、每次调用的耗时数组、推理总耗时、调用耗时均值/p50/p95/首调用耗时，以及整条 rollout 墙钟时间。计时边界为 SmolVLA 的 `_get_action_chunk`，在每次调用前后同步 CUDA；包括模型内图像准备、动作块采样和输出裁切，不包括外部 observation/action processor、环境 step 与视频编码。`rollout_wall_s` 包含环境、processor、推理和 Python 循环。它们是本机同步仿真的时间，不等于真机控制周期。
