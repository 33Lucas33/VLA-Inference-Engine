# 01 · SmolVLA + LIBERO 测评

**状态：本机阶段完成。** 本项目把公开 SmolVLA checkpoint 与本地 100k checkpoint 放到相同 LIBERO 协议下测评，保留逐 episode 原始证据和可复算汇总，并提供可视化交互演示。

## 交付内容

- 四个 LIBERO suite、共 40 个任务；每模型 1200 个 episode，两模型合计 2400 个 episode。
- 两套正式 summary 与 `formal_validation.json` 均通过校验；本机结果和统计口径见[实验报告](../../experiments/smolvla_libero_20261105/REPORT.zh-CN.md)。
- 本地交互演示台可以选择 suite/task/checkpoint，实时看双相机与动作并暂停、单步、重置。
- checkpoint 来源、训练参数、协议和结果哈希保存在 `docs/experiments/`；权重、数据和完整测评视频不入库。

## 运行入口

在仓库根目录执行。资产准备、环境安装和正式测评命令见[测评手册](../../runbooks/libero.md)。

```bash
vla-engine eval-preflight --model artifacts/models/smolvla_libero_paper
vla-engine eval-smoke --model artifacts/models/smolvla_libero_paper \
  --suite libero_spatial --task-id 0 --episodes 1 --seed 1000 \
  --out runs/smoke-paper-spatial0-seed1000
```

交互演示入口：

```bash
python -m pip install -e '.[live]'
vla-engine live --host 127.0.0.1 --port 8765
```

核心实现位于 `src/vla_inference_engine/benchmark/` 与 `src/vla_inference_engine/live/`；固定模型推理、processor 和 LIBERO rollout 复用记录在[项目主文档](../../PROJECT_GUIDE.zh-CN.md)。

## 验收与边界

验收证据：`results/smolvla_libero_paper/formal_validation.json`、`results/smolvla_libero_100k/formal_validation.json`、`docs/experiments/smolvla_libero_20261105/` 和 `runs/live/`。目前已记录 47 项全量本机测试通过、Ruff 通过，以及交互 task 0 完整成功演示。

这代表本机固定环境验收完成，不代表已在全新机器上重建，也不代表公开模型分数与论文完全复现。新机器重建和 Long 套件分数差距归因仍待后续处理。当前项目不要求安装尚未实现的部署性能、远程推理或加速模块。
