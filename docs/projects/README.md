# VLA + Infra 子项目导航

仓库把一个完整系统拆成四个可以分别阅读、运行和验收的交付项目。共用代码保存在 `src/vla_inference_engine/`，共享的协议和基础工具放在 `configs/`、`scripts/`、`tools/`。每个子项目文档说明自己的范围、入口、验收方法和当前状态。

| 序号 | 子项目 | 当前状态 | 项目说明 |
|---|---|---|---|
| 01 | SmolVLA + LIBERO 测评 | 本机阶段完成 | [项目说明](01-smolvla-libero/README.md) |
| 02 | 部署性能测评 | 规划中 | [项目说明](02-deployment-benchmark/README.md) |
| 03 | 异步与远程推理 | Mock 原型 | [项目说明](03-async-remote-inference/README.md) |
| 04 | 推理加速 | 规划中 | [项目说明](04-inference-acceleration/README.md) |

四个项目共享代码和模型资产约定，但各自记录运行依赖、配置、测试及结果。技术实施顺序仍见[路线图](../ROADMAP.md)；P0–P7 是工程步骤，不等同于四个交付项目。
