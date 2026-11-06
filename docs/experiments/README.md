# Phase status and reuse ledger

| Phase | Current status | Reuse in this repository |
|---|---|---|
| P0 | Implemented mock loop; test and demo evidence required after install | New contracts, clock, queue, trace and CLI |
| P1 | Both model smoke runs and formal rollouts completed using official LeRobot environments | Official LeRobot environment, suite metadata and fixed smoke flags |
| P2 | Planned | No ACT checkpoint or adapter imported |
| P3 | Planned beyond P0 queue | P0 generation/deadline/append rules are the starting point |
| P4 | Planned | No RTC/remote implementation copied from vendor |
| P5 | Paper and local 100k processor preflights passed; own control-loop integration pending | SmolVLA schema, tokenizer, processor and checkpoint hash checks |
| P6 | 2400 formal episodes complete and validated; public checkpoint 70.08%, local checkpoint 75.75%; deployment performance work pending | Parent evaluation protocol, storage/GPU gates and strict raw evidence checks |
| P7 | Optional/planned | No robot driver or hardware claims |

The evaluation-specific scripts and validators are now in this repository under
`scripts/` and `tools/`. The parent training and Dashboard workflows remain
outside this first project. The canonical progress record is
[`PROJECT_GUIDE.zh-CN.md`](../PROJECT_GUIDE.zh-CN.md).

## 已归档实验

- [2026-11-05：SmolVLA + LIBERO 阶段结项报告](smolvla_libero_20261105/REPORT.zh-CN.md)：训练来源、实际超参数、结构差异、正式结果与未闭环事项。
- 同目录保存实际 `train_config.json` 和包含来源哈希的 `experiment_record.json`；不包含权重或原始视频。
