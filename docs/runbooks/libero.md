# LIBERO 测评运行手册

从本仓库根目录运行，使用 Python 3.12 和固定 LeRobot 0.6.2 环境。新机器可执行 `bash scripts/bootstrap_env.sh`（Linux/cu128 精确 lock、固定 LeRobot commit）和 `bash scripts/download_assets.sh`（论文/base 权重、数据、LIBERO assets、tokenizer）。这条从空仓库下载的路径尚未在新机器实测。

大型资产也可以存放在仓库外，按 `tools/link_assets.py` 的映射链接进本仓库；脚本会拒绝覆盖已有普通目录或不同链接。`--source-root` 目录须提供固定论文权重、base、`lerobot/libero` 数据、LIBERO assets、LeRobot 源码、HF 缓存和 LIBERO 配置。请将示例路径替换为你自己的资产目录。

```bash
conda activate libero
python tools/link_assets.py --source-root /path/to/asset-collection \
  --local-model /path/to/checkpoints/100000/pretrained_model
python -m pip install -e . --no-deps
vla-engine eval-preflight --model artifacts/models/smolvla_libero_paper
```

预检产物在 `outputs/provenance/` 和 `outputs/preflight/`。检查 `valid=true`，特别是论文/本地 checkpoint 的 processor、LeRobot 绑定和命令 parser。预检成功不代表推理已成功。

```bash
vla-engine eval-smoke --model artifacts/models/smolvla_libero_paper \
  --suite libero_spatial --task-id 0 --episodes 1 --seed 1000 \
  --out runs/smoke-paper-spatial0-seed1000
```

smoke 先通过 GPU/存储门禁，然后运行官方 `lerobot-eval`。输出保留命令、模型哈希、环境 provenance 引用、LeRobot `eval_info.json`、视频与 `smoke_validation.json`；其 `formal_result=false`。若 GPU 忙，`scripts/gpu_gate.sh` 返回 3；先确认其他任务结束，再重新执行即可。每次 smoke 请使用新的 `--out` 路径。

```bash
vla-engine eval-formal --model artifacts/models/smolvla_libero_paper \
  --label smolvla_libero_paper
LOCAL_MODEL_PATH=artifacts/models/smolvla_libero_100k bash scripts/eval_two_models.sh
```

正式 runner 固定三 seed、每任务 10 episode，每个 checkpoint 共 1200 episode。每 seed 开始前默认要求 20 GiB 空闲；`EVAL_MIN_FREE_GIB` 可根据真实容量调整，但记录会保留阈值。GPU 忙时 `wait_for_gpu.sh` 每小时重试，日志见 `outputs/gpu_gate/checks.jsonl`。中断时不覆盖失败证据，已完整通过 seed 和 provenance 校验的 run 可跳过。

正式结果位于 `results/raw/<label>/seed_<seed>/`，汇总位于 `results/<label>/metrics_summary.json`，现场复验位于 `results/<label>/formal_validation.json`。只有 `formal_validation.json` 为 `valid=true`，才能引用成功率。双模型对照写入 `results/comparison/`，差距诊断索引写入 `results/diagnostics/`。完整方法和当前进度见 [项目思考文档](../PROJECT_GUIDE.zh-CN.md)。
