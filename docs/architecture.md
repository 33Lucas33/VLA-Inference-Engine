# 当前架构与依赖边界

当前可执行入口包括 LIBERO 测评编排、实时交互演示台和 CPU Mock 原型。训练工作流仍在父项目；当前仓库以已有 checkpoint 为测评输入。论文权重与本地 100k 权重的单任务真实 rollout 均已通过，双模型正式三 seed 矩阵已完成，共 2400 episode，六个 seed 与两套汇总均通过正式校验。

## 1. SmolVLA + LIBERO 测评

```text
vla-engine CLI
  ├─ eval-preflight → scripts/preflight_eval.sh
  ├─ eval-smoke → benchmark/smoke.py
  └─ eval-formal → scripts/eval_all_suites.sh
         ↓
scripts/common.sh                   同一 Conda 解释器、资源路径和 EGL 设置
benchmark/protocol.py + libero.py    固定协议、统一命令构造
         ↓
LeRobot lerobot-eval                 模型/processor、任务环境、同步 rollout
         ↓
LIBERO → robosuite → MuJoCo          场景、机器人控制与物理仿真
         ↓
eval_info.json + 视频 + 命令/日志/环境证据
         ↓
tools/validate_* → summarize_results → compare_models / diagnostics
```

smoke 用 `benchmark/runtime.py` 从 `common.sh` 读取环境；正式脚本直接 source 同一文件。Shell 通过 `tools/build_eval_command.py` 获取 Python 构造的 argv，使用 NUL 分隔保留路径中的空格和特殊字符，生成失败会立即退出。预检与正式执行共用命令构造器。

`src/` 不导入 `tools`；需要审计工具时通过明确的子进程调用。`tools` 可以导入 `benchmark` 的协议和统计函数。LeRobot/torch 在实际评测或资产检查阶段加载；基础 CLI、Mock、命令构造不加载它们。

## 2. 模块所有权

| 所在层 | 负责内容 | 运行边界 |
|---|---|---|
| `benchmark/protocol.py` | suite/horizon、输入输出形状、正式 seed 和公共推理参数 | 纯 Python；无 GPU/网络 |
| `benchmark/libero.py` | checkpoint 元信息、smoke/formal 命令构造 | 不加载模型权重 |
| `benchmark/runtime.py` | 找到源码 checkout、取得统一评测环境 | 需要 editable 安装和完整仓库 |
| `benchmark/smoke.py` | 单任务执行和结果校验 | 使用 LeRobot；结果不算正式指标 |
| `scripts/` | 安装/下载/资源门禁、跨 seed 和跨模型编排 | Bash；不再维护独立的 LeRobot 参数列表 |
| `tools/` | 资产、processor、结果、溯源、比较和诊断 | 一个命令对应一个审计或报告职责 |
| `configs/evaluation/` | 公开对照基线与来源 | 不决定运行 suite；必须与协议一致 |
| checkpoint | 网络配置、tokenizer、归一化状态与权重 | 外部资产；由检查工具约束兼容性 |
| `vendor/lerobot` | VLA 实现、训练能力、评测循环、环境桥接 | 固定版本的外部源码；当前库调用其评测入口 |

工具中的独立检查有意保留：模型元信息检查、processor 张量契约、命令原生解析、原始结果矩阵、运行证据、summary 重算分别验证不同问题。跨工具共享固定协议和 Wilson 计算，避免维护两套相同实现。

## 3. Mock 与后续 Infra

```text
MockEnvironment → ObservationFrame → ControlLoop → SyncInferenceBackend → MockPolicy
       ↑                                  ↓                              ↓
       └──────────── apply / hold ← ActionQueue ← ActionChunk 校验 ───────┘
                                          ↓
                                    TraceRecorder
```

`contracts/` 定义观测、动作与适配器接口；`adapters/mock.py` 提供演示实现；`inference/sync.py` 记录同步调用的起止时间；`control/` 管理 generation、队列、dispatch 与显式 hold；`telemetry/` 写 trace。

当前 CLI 用 FakeClock 验证逻辑时间和队列行为，没有实现真实硬件周期调度。ACT、真实 SmolVLA 的自有 adapter、RTC、远程推理和机器人驱动仍属后续工作。LIBERO 测评不依赖这些待实现模块。

## 4. 配置、安装与产物

- 原先无人读取的 YAML 已保存在 `docs/design/config-examples/`，仅作为设计草案。
- 本阶段的评测 CLI 面向源码仓库，安装方式为 `pip install -e .`。普通 wheel 不包含 `scripts/tools/configs`，不能作为独立测评分发物；评测入口会给出明确错误。
- 下载器下载固定 revision 后，通过 `record_hub_tree.py` 记录 Hub 的文件大小和 LFS 哈希，供本地预检使用。完整下载和清洁机器重建仍需实测。
- `outputs/` 保存环境预检和日志；`runs/` 保存 smoke/Mock；`results/` 保存正式 raw、汇总和比较；目录不合并，因为续跑和产物验收的生命周期不同。
- 正式续跑检查当前请求的模型路径，并重新核对原命令与模型文件哈希。模型变化时旧 seed 不能直接跳过。
- 基础测试走 CI；固定 LeRobot/模型/数据契约测试需要本机资源。单任务 GPU smoke 已验证两套权重的控制链路；完整正式矩阵和原始证据已验收，结果见中文项目主文档。

## 5. 交互演示台

`vla-engine live` 用 `scripts/common.sh` 启动本地 Starlette 服务；浏览器经 HTTP 选任务和发命令、经 WebSocket 接收双相机与动作快照。`live/session.py` 只保持一个活动会话，并为每次加载/重置创建单独工作进程。进程内的 `live/worker.py` 复用固定 LeRobot rollout、LIBERO 环境及 processor，在步边界暂停或单步，写入 `runs/live/`。它与 `benchmark/` 共享 suite horizon 和模型检查，但交互结果不进入正式测评的 `results/`。代码与运行入口见 [运行手册](runbooks/live-viewer.zh-CN.md)。
