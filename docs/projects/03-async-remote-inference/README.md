# 03 · 异步与远程推理

**状态：Mock 原型已存在，真实策略和远程链路待实现。** 本项目研究策略推理耗时大于仿真控制周期时，动作队列如何保持执行连续、动作新鲜和时序正确，并逐步扩展到 RTC 与远程推理。

## 当前原型

CPU Mock 展示观测、同步后端、动作队列、控制循环和 trace 的数据流：

```bash
vla-engine mock --ticks 100 --seed 0 --out runs/mock-demo
```

主要代码位于 `src/vla_inference_engine/contracts/`、`inference/`、`control/`、`adapters/` 和 `telemetry/`。Mock 不调用真实 SmolVLA、MuJoCo 或机械臂；LIBERO 实时演示台目前复用 LeRobot rollout，尚未接入这套自有队列。

## 后续交付

分阶段验证真实策略适配、动作块校验与对齐、超时和延迟注入、队列更新策略，再比较同机 RTC 和远程 loopback。每阶段记录队列覆盖率、动作年龄、控制间隙、端到端时延、失败行为和任务成功率。

## 建议验收

无效或过期动作不会被派发；reset/generation 更换后旧动作不会进入新 episode；暂停、停止和故障恢复有确定行为；同步、RTC 和远程实验共用相同任务与指标；网络超时不会让控制器无限等待。第一阶段保持 CPU 可测，真实 GPU 和远程服务作为单独集成层。
