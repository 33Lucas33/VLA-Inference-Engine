# VLA + Infra 项目思考与实施记录

> 中文主文档，持续维护。当前优先项目：SmolVLA 在 LIBERO 上的可复现测评。首个测评子项目及交互演示已完成本机阶段验收；论文分数差距诊断、新机器重建仍待完成。更新于 2026-11-06。

## 1. 总体目标和本轮决定

整个仓库将由几个各自完整的小项目组成：先交付 SmolVLA + LIBERO 测评，再在同一环境、模型和成功率口径上增加部署测评、动作队列和加速实验。当前的测评项目必须能单独解释其输入、执行协议、原始证据和结果，不依赖后续的推理 Infra 模块。

**原训练与测评项目的目标不只是在 LIBERO 上测评。**其 README 包括固定版本、数据与模型、100k 更新训练、四套件三 seed 测评、双模型比较、差距诊断、Dashboard 和最终报告。该项目已有训练 checkpoint、固定资产、成熟的测评与校验代码；迁移初期，其 `results/final_report_audit.json` 指出两套正式结果缺失。当时复用的是可执行能力和协议。2026-11-05，两套各 1200 episode 的正式测评已在本仓库完成并通过审计；新结论以本仓库结果为准，不把旧审计状态当作当前状态。

本轮决定沿用本仓库现有的根目录分层，不新建第二套 `projects/smolvla_libero_eval/` 包：

```text
VLA-Inference-Engine/
  src/vla_inference_engine/benchmark/  # 固定协议、模型契约、命令、运行环境与 smoke
  scripts/                             # 预检、正式运行、双模型编排、GPU/存储门禁
  tools/                               # 可单独审计的资产/协议/结果验证与统计
  configs/evaluation/                  # 公开基线及来源；实际协议在 benchmark/protocol.py
  tests/                               # 无 GPU 的核心/测评契约回归
  docs/                                # 项目思考、使用手册、协议
  artifacts/, vendor/, .cache/         # 仅本地链接或外部资产，不提交权重/数据
  outputs/, results/, runs/            # 本机证据，不提交原始视频
  docs/design/config-examples/         # 未接入加载器的旧 YAML 设计草案
```

理由：首个项目共享已有的 Python 包与 CLI；后续部署项目可在 `src/vla_inference_engine/` 下继续扩展。独立性由清楚的入口、协议、测试、外部资产绑定和结果验证保证。大文件可以在仓库外，但运行命令和校验代码属于本仓库。

## 2. 测评对象与固定协议

LIBERO 提供现成的 MuJoCo/robosuite 任务与 Panda 机械臂，不需为现有 40 个任务另写 XML。任务由 BDDL、初态和场景资产定义。LeRobot `lerobot-eval` 加载 SmolVLA、processor、仿真环境并执行 rollout；本仓库负责资产校验、固定命令、资源门禁、保存证据、重算成功率与对照。

```text
LIBERO BDDL/初态 → 机器人与双相机观测 → LeRobot 环境 processor
→ checkpoint 前处理器 → SmolVLA → checkpoint 后处理器
→ 7D relative 动作 → 仿真 step → 每 episode 成功与视频
→ 本项目的 seed 校验 → summary 重算 → 正式验证/双模型对照
```

| 参数 | 固定值 | 证据/实现 |
|---|---|---|
| 套件 | Spatial、Object、Goal、Long，各 10 任务 | `benchmark/libero.py`、固定 LeRobot 源码 |
| 机械臂/动作 | LIBERO Panda，末端相对平移 3 + 旋转 3 + 夹爪 1，即 7D | `tools/validate_processor_contract.py` |
| 模型输入 | 两路 256×256 RGB、8D state、语言指令 | checkpoint config、processor、数据 schema |
| 控制 | `relative`、固定初态、hard reset、单任务同步执行、`n_action_steps=1` | `scripts/eval_all_suites.sh`、命令预检 |
| 推理 | flow steps 10、chunk size 50、`compile_model=false` | checkpoint config 与 LeRobot parser |
| 正式矩阵 | 4 套件 × 10 任务 × 每任务 10 episode × seed 1000/2000/3000 = 每 checkpoint 1200 episode | `validate_eval_command.py`、`validate_eval_info.py` |
| 每套件 horizon | Spatial 280、Object 280、Goal 300、Long 520 | 固定 LeRobot `TASK_SUITE_MAX_STEPS` |
| smoke | 单任务 1–3 episode，单独保存且 `formal_result=false` | `benchmark/smoke.py` |

训练数据 `lerobot/libero` 是 40 任务、1693 episode、273465 frame、10 fps 的离线演示数据。评测观测由模拟器在线产生，**不从训练 Parquet 回放成功率**。原项目已训练的 100k checkpoint 是另一个受测模型；两模型 VLM 初始化和实际层数/专家宽度不同（公开权重 32 层，本地 16 层）；本地从预训练 VLM 加新建动作专家训练，未从公开 LIBERO 权重微调。对照是同场景部署表现比较，不能解释成同结构同初始化的训练消融。公开论文基线见 `configs/evaluation/baselines.json`，只用于参考。

## 3. 复用清单与职责

| 层 | 当前文件 | 来源和作用 | 状态 |
|---|---|---|---|
| CLI/测评接口 | `src/vla_inference_engine/cli.py`、`benchmark/libero.py`、`benchmark/smoke.py` | 保留原有 Mock/命令生成；新增预检、smoke、正式评测入口和单任务结果校验 | 已实现；论文权重与本地 100k 权重的单任务 smoke 均已通过 |
| 环境与资产 | `scripts/common.sh`、`resolve_env.sh`、`bootstrap_env.sh`、`download_assets.sh`、`fetch_lerobot.sh`、`tools/link_assets.py`、`ensure_libero_config.py`、`validate_libero_assets.py`、`manage_lerobot_binding.py` | 移入环境解析、Linux/cu128 lock 和 LIBERO 校验；新增本仓库本地资源链接及固定 LeRobot 获取 | 本机链接已验证；下载脚本已补齐固定 Hub tree 清单，新机器下载/重建尚未实测 |
| 模型与数据契约 | `pin_tokenizer_assets.py`、`validate_tokenizer_assets.py`、`validate_processor_contract.py`、`validate_dataset_coverage.py` | 核对双相机、8D/7D、归一化、tokenizer 与 40 任务数据 | 论文与本地 100k 权重预检均已通过 |
| 正式协议 | `scripts/preflight_eval.sh`、`eval_all_suites.sh`、`gpu_gate.sh`、`wait_for_gpu.sh`、`tools/validate_eval_command.py`、`validate_pairing_protocol.py`、`record_storage_gate.py` | 固定命令、LeRobot 原生解析器校验、初态配对、每 seed 前资源门禁、失败产物保留 | 两模型正式矩阵均已完成，共 2400 episode；六个 seed 校验通过 |
| 结果审计 | `validate_eval_info.py`、`validate_formal_seed.py`、`summarize_results.py`、`validate_formal_summary.py`、`generate_provenance.py` | 任务矩阵、视频、命令哈希、环境证据与 raw 重算；只有完整 seed 可跳过 | 两套正式 summary 与 raw 一致，formal_validation 均通过；2400 个非空视频 |
| 双模型/诊断 | `scripts/eval_two_models.sh`、`compare_models.py`、`prepare_gap_diagnostics.py`、`diagnose_gap.py`、`validate_diagnostic_closure.py`、`build_diagnostic_closure.py` | 串联论文/本地模型测评、配对比较与可审计差距诊断 | 双模型配对结果已生成；公开权重 Long 触发差距诊断，根因/seed 4000 复测待完成 |
| 范围外 | 训练脚本、Dashboard、最终综合报告生成器 | 原项目的训练和可视化工作流，当前测评项目不以它们为运行依赖 | 未迁入 |

“已迁入”表示文件在本仓库并去除了已发现的父目录读写路径；“已验证”要看具体证据。原项目的 `run_full_evaluation_managed.sh` 混合训练、Dashboard 与最终报告，故重组为本项目的 `eval_all_suites.sh` 和 `eval_two_models.sh`。

## 4. 运行路径与可审计产物

从本仓库根运行。先准备 Python 3.12 的 LeRobot 0.6.2 环境、固定资产和固定 commit 的源码。新机器用 `scripts/bootstrap_env.sh`、`scripts/download_assets.sh`；前者按本仓库 `environment.yml` 与 `requirements-linux-cu128.lock.txt` 建环境，后者下载固定模型、数据、LIBERO assets 和 tokenizer。已有外部资产可用 `tools/link_assets.py` 链接；`--source-root` 是资产存储根，不要求必须是父项目；`--local-model` 可另指向训练 checkpoint。链接不会覆盖已有普通目录或不同目标的链接。新机器路径尚未端到端复测。

```bash
conda activate libero
python tools/link_assets.py --source-root /path/to/asset-collection \
  --local-model /path/to/100k-checkpoint/pretrained_model
python -m pip install -e . --no-deps
vla-engine eval-preflight --model artifacts/models/smolvla_libero_paper
vla-engine eval-smoke --model artifacts/models/smolvla_libero_paper \
  --suite libero_spatial --task-id 0 --episodes 1 --seed 1000 \
  --out runs/smoke-paper-spatial0-seed1000
vla-engine eval-formal --model artifacts/models/smolvla_libero_paper \
  --label smolvla_libero_paper
LOCAL_MODEL_PATH=/path/to/100k-checkpoint/pretrained_model \
  bash scripts/eval_two_models.sh
```

`eval-preflight` 不创建 LIBERO 环境或分配 CUDA；它写 `outputs/provenance/assets.json` 与 `outputs/preflight/*.json`。`eval-smoke` 先过 GPU 门禁和预检，产出 `runs/.../{command.sh,manifest.json,eval.log,eval_info.json,smoke_validation.json,videos/}`。成功或失败仅证明链路能否运行，不计入正式成功率。`eval-formal` 固定三 seed、每任务 10 episode；每 seed 放在 `results/raw/<label>/seed_<id>/`，失败 run 移至 `results/failed_runs/`，完整通过验证的 seed 才能续跑跳过。随后 `results/<label>/metrics_summary.json`、任务表、`formal_validation.json` 从 raw 生成并复验。双模型脚本再写 `results/comparison/`、`results/diagnostics/`。

正式结果须有精确 1200 episode、逐 episode 非空且唯一的视频、原始 `eval_info.json`、实际执行的 `command.sh`、命令解析和模型哈希、软件/LeRobot 版本、每 seed 日志、raw 重新聚合且通过的 `formal_validation.json`。不能仅凭 `metrics_summary.json` 宣称完成。

## 5. 当前事实、已做验证与差距

本机外部资产包含论文权重、SmolVLA base、`lerobot/libero` 数据、LIBERO assets、Hugging Face 缓存、LeRobot 源码；公开仓库只保留获取方式和固定 revision，不包含这些大文件。固定 LeRobot commit 为 `8ae2354592ec4480c38134fb75350b8e7992e0e9`；数据 revision 为 `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4`；论文模型 revision 为 `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`。本地 100k checkpoint 来自原训练项目的 `outputs/train/smolvla_libero_100k_b64/checkpoints/100000/pretrained_model/`，未复制进 Git。

2026-10-29 的本仓库验证：

- `bash scripts/preflight_eval.sh` 返回 0；`outputs/provenance/assets.json` 和资产、tokenizer、processor、配对检查均为 `valid=true`。LIBERO 资产校验 586 个文件；数据校验 1693 episode、273465 frame、40 任务。正式 seed 1000 命令由 LeRobot parser 解析，期望 400 episode。
- 本地模型 `EVAL_LABEL=smolvla_libero_100k` 的相同预检也返回 0，processor 和固定正式命令均为 `valid=true`；本地 100k 权重已链接到本仓库忽略的 `artifacts/models/smolvla_libero_100k`。
- 本仓库完整 Python 测试于 2026-11-03 通过 43 项（75.23 秒），包含固定资产契约与结果审计；`ruff check src tools tests` 和所有 Shell 脚本语法检查通过。无资产 CI 仍运行 `tests/unit tests/contract tests/integration`。
- 2026-11-03 两套权重的真实 smoke 均已运行：`runs/smoke-paper-spatial0-seed1000-20261103/` 与 `runs/smoke-local100k-spatial0-seed1000-20261103/` 的 `smoke_validation.json` 均为 `valid=true`，各 1 个 episode 成功且视频非空；LeRobot `eval_s` 分别为 61.29 秒、58.60 秒。两者均标为 `formal_result=false`，不能推导正式成功率。
- 2026-11-03 正式启动前曾因根盘低于 20 GiB 门槛暂停；回收站由外部操作清空后，用户要求继续。GPU 忙时等待，后改为每 60 秒复查。历史 runner PID 为 1336441，主日志 `outputs/logs/eval_two_models_20261103_60s.log`。
- 2026-11-05 约 17:15，双模型正式测评、汇总、校验、配对比较和诊断准备均已完成，runner 已退出。每模型 1200 episode、1200 个非空视频；六个 seed 及两份 `formal_validation.json` 均通过。
- 2026-11-05 结项归档再次核对两模型权重哈希、summary 哈希与正式验证报告的绑定、六份 raw 文件哈希及 2400 个非空视频。原始产物保持在 `results/`，小型 Git 记录见下文。

| 套件 | 公开 checkpoint 本机结果 | 本地 100k 结果 | 本地减公开 |
|---|---:|---:|---:|
| Spatial | 75.33% | 77.67% | +2.33 个百分点 |
| Object | 88.33% | 86.00% | −2.33 个百分点 |
| Goal | 79.00% | 81.33% | +2.33 个百分点 |
| Long | 37.67% | 58.00% | +20.33 个百分点 |
| 总体 | 70.08%（841/1200） | 75.75%（909/1200） | +5.67 个百分点 |

**结项决定：首个“SmolVLA + LIBERO 双模型测评与结果审计”项目完成本机验收，可以阶段结项。**完整中文报告见 [阶段结项报告](experiments/smolvla_libero_20261105/REPORT.zh-CN.md)，包括训练初始化链路、完整超参数表、模型结构差异、成功率及结论边界；实际保存的配置见 [train_config.json](experiments/smolvla_libero_20261105/train_config.json)，可审计的小型结果与来源哈希见 [experiment_record.json](experiments/smolvla_libero_20261105/experiment_record.json)。这些记录放在 `docs/`，避免随着被 Git 忽略的 `results/` 一起遗漏。

本地训练要点：预训练 `SmolVLM2-500M-Video-Instruct`，新建动作专家，冻结 VLM；100000 更新、batch 64、梯度累积 1、AdamW LR 1e-4、warmup 1000、cosine decay 到 2.5e-6、bf16、训练 policy compile=max-autotune、seed 1000。最终配置 `pretrained_path=null`、`resume=false`。训练脚本与运行证据仍属于原项目，本仓库已接入该 checkpoint 的正式测评；未将训练入口迁入本仓库。

未闭环事项：公开 checkpoint 的 Long 相对论文 71% 基线存在 46.95% 相对缺口，已生成失败视频索引，根因审查与独立 seed 4000 复测尚未完成；新机器从零重建仍待验证。当前结项不宣称已严格复现论文分数，不宣称独立打包了完整训练工作流。下一项目可使用现有成功率参考开展部署/推理性能实验；须同时报告成功率、延迟和吞吐，保留原始基线。

## 6. 源码阅读路线

1. 读本文、`README.md` 和 `configs/evaluation/baselines.json`：理解目标与协议。
2. 读 `benchmark/protocol.py`、`benchmark/libero.py`、`benchmark/runtime.py`、`benchmark/smoke.py`、`cli.py`：从固定协议和 CLI 参数追到实际 `lerobot-eval` 命令、统一 Conda 运行环境和 smoke 证据。
3. 读 `scripts/preflight_eval.sh`、`tools/validate_eval_command.py`、`validate_processor_contract.py`：看模型/场景契约如何在用 GPU 前被拒绝或确认。
4. 读固定 LeRobot 源码 `vendor/lerobot/src/lerobot/scripts/lerobot_eval.py` 的 `eval_main`、`rollout`，再看 `envs/libero.py` 与 `processor/env_processor.py`：走一次观测、推理、动作、成功判定。
5. 读 `scripts/eval_all_suites.sh`、`tools/build_eval_command.py`、`validate_eval_info.py`、`validate_formal_seed.py`：理解命令复用、seed 级续跑和原始证据要求。
6. 读 `summarize_results.py`、`validate_formal_summary.py`、`compare_models.py`：从 raw 推导任务/套件成功率、置信区间与配对比较。
7. 第一项目验收后，读 `contracts/`、`control/loop.py`、`control/queue.py` 和 Mock 测试，进入部署与加速小项目。

后续每做一个模块，继续在本文补充入口、输入输出、不变量、真实结果路径和未解决问题。

## 7. 架构核对与运行决定（2026-11-03）

- 以 `src/vla_inference_engine/benchmark/protocol.py` 保存评测参数，`libero.py` 构造 smoke/formal 命令；Shell 通过 `tools/build_eval_command.py` 取得同一 argv。这样预检与正式执行不会各自维护一份参数列表。LeRobot parser 校验仍独立核对协议，避免构造和验收互相掩盖错误。
- smoke 和正式测评统一由 `scripts/common.sh` 解析 Conda 环境。同步推理后端接入 Mock 控制循环；未被读取的 YAML 移至 `docs/design/config-examples/`，并在 `docs/architecture.md` 标明不是运行配置。
- 下载脚本现记录固定 Hugging Face 仓库的远端 tree 元数据，以满足本项目资产预检要求；`link_assets.py` 也支持仅绑定本地 checkpoint。固定版本的大型资产继续在 Git 外部。
- 正式续跑以本次指定 checkpoint 路径和原模型文件哈希为准；GPU 等待脚本只对“GPU 忙”重试，其他门禁错误立即退出。
- 正式结果目录位于项目根盘。2026-11-03 解决存储门禁后启动，2026-11-05 已完成；完整矩阵与校验证据见第 5 节。此节保留架构和运行决策，当前进度以第 5 节及阶段报告为准。

## 8. 交互式 LIBERO 模型演示台（2026-11-05 实现）

本模块作为首个测评项目的交互展示入口，**不改变正式测评协议或已归档的成功率**。`vla-engine live --host 127.0.0.1 --port 8765` 经 `scripts/common.sh` 使用固定 Conda/EGL 环境，在本机启动 Starlette + Uvicorn 服务。浏览器从 LIBERO 元信息选择四个 suite 的 task、两个已注册 checkpoint、seed 和初态编号；只允许一个活动会话。`src/vla_inference_engine/live/session.py` 为每次加载新建独立 Python 工作进程，拥有唯一的模型、CUDA 和 MuJoCo 上下文；切换任务会先关闭旧进程。`events.py` 在步边界管理接管、暂停、继续、单步和结束；重置生成新的 generation，旧画面被丢弃。

`worker.py` 调用固定 LeRobot 的 `make_env`、`make_policy`、processor 工厂和 `rollout`。观察包装记录 `policy.select_action` 输出、checkpoint 后处理动作及 `env_postprocessor` 最终动作，不修改原动作；`render_callback` 在 reset 后及每个环境步后读取两路相机、发布快照并等待控制命令。前端 `static/app.js` 通过 HTTP 发命令，WebSocket 接收最新快照；浏览器刷新可以恢复当前活动会话。服务和工作进程是本仓库的最小独立实时链路，没有复制 LeRobot 主控制循环，也没有让 `src/` 导入 `tools/`。当前机器已有 Starlette/Uvicorn，因此使用它们实现，无须另外安装 FastAPI。

每次运行写入 `runs/live/<session_id>/generation_<n>/{config.json,worker.log,trace.jsonl,frames/,image.mp4,image2.mp4,summary.json}`，标记 `mode=interactive` 和 `formal_result=false`。两路 MP4 按每步一帧、10 fps 回放；`summary.json` 记录 checkpoint SHA256、终态及视频信息。一对相机帧与同一步的 7D 动作共用 step ID；画面是仿真观测显示图，动作并非直接测得的实际末端位移。页面在本次连接期间绘制环境动作 XYZ 历史；刷新后曲线不会重建旧记录，完整逐步数据在 trace 中。安装、操作、故障定位和代码阅读顺序见 [交互演示台运行手册](runbooks/live-viewer.zh-CN.md)，原设计与验收记录见 [实现规划](design/LIBERO_LIVE_VIEWER_PLAN.zh-CN.md)。

**已验证**：公开模型 Spatial 任务真实初态、单步与运行中暂停（停在第 7 步后步数不再增长）；本地 100k 模型 Spatial 任务真实加载与单步；同 seed/初态重置后的初始 JPEG SHA256 与重置前一致；浏览器中公开模型 Object 任务双相机、任务原文与第 1 步三层动作均可见。四个 suite 的 task 0 均以两套 checkpoint 完成真实单步。浏览器连续接管的 Spatial 任务在第 72 步成功，`runs/live/5a5568eddc2a4c829b6a408e1acab204/generation_0/summary.json` 为 `success=true`。之后发现并修复断线暂停恰好落在步边界的竞态；复测在第 13 步确认 PAUSED、步数保持不变，恢复后仍在第 72 步成功，两路 MP4 均可解码，见 `runs/live/94a1836fd02445ba82ad03782903d08e/generation_0/`。三次相同模型/seed/初态运行的前两步环境动作逐元素完全一致，覆盖连续运行、断线恢复和手动单步三种控制方式。浏览器的 XYZ 曲线已经用连续两步验证。当前全量 Python 测试 47 项与 Ruff 均通过，构建的 wheel 已确认包含页面静态资源。交互运行不等于完整正式 episode 的统计复现；页面目前不显示模型内部完整动作块、自然语言解释或机器人遥操作。

**新的阅读顺序**：先读 `live/events.py` 的状态门，再读 `catalog.py`、`server.py` 和 `session.py` 的选择、命令与工作进程管理；然后逐行读 `worker.py::execute` 和 `render`，对照固定 LeRobot `lerobot_eval.py::rollout` 与 `envs/libero.py` 的 reset、观测和成功判定；最后读 `static/app.js` 的同 generation 快照和页面状态。该顺序接在第 6 节第 4 步之后最容易理解。

## 9. 阶段结项与 GitHub 组织建议（2026-11-06）

**当前阶段可以按“SmolVLA + LIBERO 双模型测评、结果审计与交互演示”结项，建议作为首个 `v0.1.0` 发布范围。**本次复查两套 `results/<model>/formal_validation.json` 均为 `valid=true`；交互验收 `runs/live/94a1836fd02445ba82ad03782903d08e/generation_0/summary.json` 记录第 72 步成功、两路视频且无编码错误。47 项本机测试和 Ruff 的验收记录见第 8 节。本次仅核对已有证据，没有重新执行正式矩阵。

阶段完成的边界是本机固定环境下的测评和展示。论文分数差距归因、全新机器重建、训练入口迁入、严格部署性能测量、原始 rollout 与交互观察包装的独立动作对齐仍作为后续事项公开记录；不以当前阶段完成推导整个 VLA + Infra 系统完成。

本仓库现已按 **一个 GitHub 仓库、一个共享 Python 包、四个可独立说明和验收的子项目** 整理导航。四项划分是 GitHub 展示和交付边界；运行代码仍由共享包维护：

| 子项目建议 | 要解决的问题与独立交付 | 当前状态 |
|---|---|---|
| 01 SmolVLA + LIBERO 测评 | 固定协议、双模型成功率、可审计结果、交互展示 | 本机阶段完成 |
| 02 部署性能测评 | 分段耗时、端到端 p50/p95/p99、吞吐、显存与成功率联合报告 | 待实现，复用 01 的任务和模型基线 |
| 03 异步与远程推理 | 同步、动作队列、RTC、远程执行及延迟/故障实验 | 有 Mock 原型，真实链路待实现 |
| 04 推理加速 | 编译、精度或执行优化；比较速度、资源和任务成功率 | 待实现，以 02 的测量方法验收 |

原 P0–P7 保留为技术实施步骤；上述 01–04 是面向用户与 GitHub 展示的交付划分，二者不一一对应。交互演示归入 01，也可作为后续项目共用的观察入口；Mock 是共用开发工具。四项共享资产与协议，各项应有明确的运行依赖，完成 01 不要求安装或实现 03/04。

当前项目导航与主代码的布局如下：

```text
VLA-Inference-Engine/                 # 仓库本身就是大项目
  README.md                           # 总目标、四项导航、当前成果和状态
  pyproject.toml                      # 共享包与按需依赖
  src/vla_inference_engine/            # 按职责维护唯一代码实现
    benchmark/                        # 测评协议与执行
    live/                             # 交互演示
    inference/, control/, adapters/   # 后续推理与控制实现
    contracts/, telemetry/            # 共享契约与记录
  configs/                            # 各实验配置
  scripts/, tools/                    # 启动编排与审计工具
  tests/                              # 核心与各子项目的验收测试
  docs/projects/
    01-smolvla-libero/README.md
    02-deployment-benchmark/README.md
    03-async-remote-inference/README.md
    04-inference-acceleration/README.md
  docs/experiments/                    # 可提交的小型报告与来源记录
  docs/PROJECT_GUIDE.zh-CN.md          # 持续更新的总设计与决策
```

四个子项目的导航和说明已创建于 [`docs/projects/README.md`](projects/README.md) 与其下各项目 README。每份说明列出目标、入口、核心代码、验收或待实现状态。代码由共享包维护；不复制四份 LeRobot 接入、LIBERO 资产、模型加载或相同控制循环。后续某项确实有独立用户、依赖、版本与维护周期时，再考虑拆成单独仓库或包。现阶段不新增多个 Git 仓库、submodule 或长期独立分支。

阶段版本使用 tag 和 GitHub Release 固定源码与验收报告；首个测评交付建议标记为 `v0.1.0`，GitHub Release 尚待远端仓库配置后创建。模型/数据/完整视频继续存于外部资产位置，Git 保留获取方法、revision、哈希和小型结果；参见 [GitHub 大文件说明](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

本次已整理四项导航、每个项目的说明及首项目入口；文档和代码可形成首个 Git 提交。当前仓库仍未配置远端，GitHub 上传需在确定目标仓库并配置安全认证后完成。当前 CI 只运行无资产核心子集；本机完整测试/GPU 验收与远端 CI 分别描述。

## 更新记录

| 日期 | 决策与变化 | 证据 |
|---|---|---|
| 2026-10-28 | 建立总文档，确立先完成 SmolVLA + LIBERO 测评的顺序 | 首次项目盘点 |
| 2026-10-29 | 核实原项目范围和正式结果缺失；将测评能力按本仓库现有 `src/scripts/tools/configs/tests` 分层迁入；建立预检、smoke、正式评测和双模型入口；补充固定环境和资产获取 | 当日论文与本地权重预检 JSON、37 项完整测试、`ruff check src`；GPU 占用导致 rollout 待运行 |
| 2026-11-03 | 完成架构去重与命令/环境统一；执行全量测试和两套权重的真实 smoke；两次核实正式存储门禁 | `tests/` 43 项通过、`ruff` 与 Bash 语法通过；`runs/smoke-paper-spatial0-seed1000-20261103/`、`runs/smoke-local100k-spatial0-seed1000-20261103/`；`outputs/storage/evaluation_checks.jsonl`，正式门禁末次为 17.557 GiB |
| 2026-11-03 | 继续审计根盘占用，发现回收站和另一个项目的 Git 临时 pack；回收站由外部操作清空，正式测评按用户要求继续暂停 | `df -B1 /` 从约 14.84 GiB 恢复到约 92.85 GiB；`du` 与 `git count-objects -vH` 的只读检查 |
| 2026-11-03 | 用户恢复正式测评；21:55 启动双模型后台 runner。首次 GPU 忙触发一小时等待，改用每 60 秒复查后重启 | 原日志 `outputs/logs/eval_two_models_20261103.log` 保留；现行日志 `outputs/logs/eval_two_models_20261103_60s.log`、PID 文件 `outputs/logs/eval_two_models_20261103.pid`；现行 PID 1336441 |
| 2026-11-03 | 论文权重 seed 1000 通过存储和 GPU 门禁，进入 40 任务正式 rollout | `outputs/storage/evaluation_checks.jsonl` 22:03 的 89.05 GiB 记录；`outputs/logs/eval/smolvla_libero_paper/seed_1000.log` 出现首任务的动作步进 |
| 2026-11-05 | 完成双模型 2400 episode 正式测评与阶段结项；核实本地训练未从公开 LIBERO 权重微调，记录完整配置及实际模型结构差异；保留论文分数差距与新机器复现为未闭环事项 | 两套 formal_validation、六份 seed raw、comparison；原训练 train_config/runtime_manifest/log；权重 header 与 SHA256；docs/experiments/smolvla_libero_20261105/ |
| 2026-11-05 | 将根 README 改写为公开仓库审核稿：明确项目职责、实测结果、模型来源、训练记录、安装与测评入口、结果验收和后续范围；命令使用通用路径 | 对照 CLI、bootstrap/download/link 脚本、正式 runner 与已归档实验记录；本轮仅文档修改，待用户审核 |
| 2026-11-05 | 规划 LIBERO 实时交互演示台：任务选择、模型接管、双视图、动作遥测与暂停/单步；按用户要求本轮仅规划 | 固定 LeRobot rollout/render_callback、processor 与 LiberoEnv reset 源码；原 Dashboard；docs/design/LIBERO_LIVE_VIEWER_PLAN.zh-CN.md |
| 2026-11-05 | 分步实现交互演示台：本地 Web 页面、独立 GPU/仿真工作进程、真实逐步动作与双相机、步边界控制、同初态重置及交互记录；补充运行手册和阅读路线 | `src/vla_inference_engine/live/`、`tests/live/`；本机公开/本地模型真实单步、重置首帧哈希与浏览器 Object 页面验证 |
| 2026-11-05 | 完成连续接管成功终态；修复断线暂停竞态与 Ctrl+C 回溯；增加动作曲线、双路 MP4、checkpoint SHA256 和断线自动暂停 | Spatial task 0 第 72 步成功；断线第 13 步暂停并可恢复到成功；两路视频解码与哈希检查；浏览器曲线；47 项测试及 Ruff，wheel 静态文件检查 |
| 2026-11-06 | 确认测评与交互演示可按本机验收范围阶段结项；记录一个仓库、共享代码与四个子项目展示入口的建议，区分后续项目与 P0–P7 技术步骤 | 两份 formal_validation、交互成功 summary、原路线图与 Git 状态；第 9 节为建议，未迁移代码或创建发布 |
| 2026-11-06 | 创建四个子项目导航和说明，修订 README 与 GitHub 展示结构；清理文档中开发机绝对路径，准备首个可审阅提交 | `README.md`、`docs/projects/`、主文档及 runbook；实际文件清单与 Git 暂存区审查 |
