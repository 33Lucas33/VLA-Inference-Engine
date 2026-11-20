# SmolVLA 在 LIBERO 中的动作执行长度展示实验

这个探索回答：同一个 checkpoint 每次预测 50 步动作后，连续执行其中多少步，才重新使用新观测推理？固定 `chunk_size=50`，改变 LeRobot `n_action_steps` 为 1、5、10、20。页面给浏览者选择的数值应标为“每次预测后执行步数 K”。

在仓库根目录运行：

```bash
bash scripts/action_horizon_showcase.sh
```

入口固定公开 SmolVLA checkpoint、`libero_spatial/object/goal` 各 task 0、seed 1000、初态 0、`relative` 控制和 10 步 flow sampling。每个任务和 K 运行一个完整 episode，共 12 个案例。`scripts/action_horizon_showcase.sh` 使用本地环境与模型缓存，在被 Git 忽略的 `runs/action-horizon-showcase/<时间>/` 创建实验目录。

每个案例保存双相机并排 MP4（外部视角在左、腕部视角在右；两路图像旋转 180° 供观看），按每个仿真动作一帧、20 fps 回放。`manifest.json` 记录模型 SHA256、任务、参数、是否成功、环境执行步数、实际 `_get_action_chunk` 调用次数、视频路径和大小、包含录像开销的同步运行墙钟时间。模型采样调用次数通过包装固定 LeRobot 方法直接计数；环境步数通过包装 `env.step` 直接计数。

三种频率要区分：仿真每步都会产生新观测；队列未空时 `select_action` 会取已生成的动作，不调用模型处理新观测；队列耗尽才重新从当前观测生成 50 步动作块。因此 K 越大，通常推理调用次数越少，但使用旧观测连续行动的时段也越长。

本实验每格只有一个 episode，只能展示具体轨迹和机制，不能当作成功率统计结论。录像开销、冷启动和同步仿真都会影响墙钟时间；它不是机器人物理控制性能指标。正式基线位于 `results/`，本探索单独留在 `runs/`。最终结果与所有案例的哈希见对应的实验报告。
