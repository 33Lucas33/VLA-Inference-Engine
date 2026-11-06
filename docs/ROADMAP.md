# P0–P7 roadmap for this repository

This project is organized from the user's detailed document
[`ORIGINAL_ROADMAP.md`](ORIGINAL_ROADMAP.md) (2026-10-27). That full document
defines the acceptance criteria; this file records the implementation order
and local status.

| Phase | Goal | Local status |
|---|---|---|
| P0 | CPU-only mock control loop, contracts, clock, queue, trace | Implemented; run tests and demo on each new machine |
| P1 | LIBERO environment adapter and single-task smoke | 两模型真实 LIBERO smoke 与正式 rollout 已完成；复用官方 LeRobot 环境 |
| P2 | ACT adapter and synchronous learned-policy baseline | Planned |
| P3 | Async Action Queue, alignment, delay/fault injection | Planned beyond P0 queue |
| P4 | LeRobot RTC and remote loopback/LAN | Planned |
| P5 | SmolVLA adapter, prompt/camera mapping and traces | 论文与本地权重的 processor/tokenizer 契约预检通过；尚未接入自有控制循环 |
| P6 | Config-driven benchmark and GPU/system report | 固定协议双模型 2400 episode 已完成并通过审计；部署性能矩阵仍待下一项目 |
| P7 | Robot/HIL adapter | Optional, planned |

当前首要交付以 [中文项目思考文档](PROJECT_GUIDE.zh-CN.md) 的 SmolVLA + LIBERO 测评为准。
首个测评子项目已于 2026-11-05 阶段结项，见 [结项报告](experiments/smolvla_libero_20261105/REPORT.zh-CN.md)。论文分数差距诊断与新机器重建验证仍待完成，后续推进部署与加速项目。

面向用户的四项交付导航见[子项目总览](projects/README.md)；P0–P7 是工程实施阶段，不与四项交付逐项对应。
