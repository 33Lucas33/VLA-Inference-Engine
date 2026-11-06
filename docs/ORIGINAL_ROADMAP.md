# VLA-Inference-Engine：分阶段小项目路线图

> **阅读方式**：按项目编号完成。每项都包含目标、范围、实现步骤、交付物和验收条件，可以独立运行和展示；完成后再接入下一项。
> **总目标**：把 VLA 推理部署、动作队列、远程执行和性能评测拆成一组可复现的小项目，最终组合成一个完整的 VLA 推理与评测系统。
> **开发环境建议**：Linux 优先；Python、PyTorch、LeRobot、MuJoCo/LIBERO。无真机也能完成主要项目。
> **资料核对日期**：2026-10-27。

---

## 项目路线总览

| 阶段 | 小项目 | 主要技术点 | 预计投入 | 单项完成后的展示物 |
|---|---|---|---:|---|
| 0 | P0：Mock 推理闭环 | Python 项目结构、接口、单调时钟、事件日志 | 3–5 天 | 不需要 GPU/机器人即可运行的端到端 demo |
| 1 | P1：LIBERO 仿真适配器 | 仿真环境、reset、固定 seed、episode 结果 | 4–7 天 | 可复现的 LIBERO 单任务 rollout |
| 2 | P2：ACT 策略适配器 | LeRobot dataset/checkpoint、图像与状态、同步推理 | 1 周 | ACT 在 LIBERO 中的基线结果 |
| 3 | P3：Action Queue 实验室 | 异步调度、chunk 对齐、时钟、超时与延迟注入 | 1–2 周 | 可视化比较同步推理和异步队列的实验 |
| 4 | P4：LeRobot RTC/远程推理接入 | 本地 RTC、policy server、Zenoh、LAN 推理 | 1–2 周 | 同一策略在 sync、RTC、remote 下的运行记录 |
| 5 | P5：SmolVLA 多模态接入 | 语言指令、多相机映射、VLA checkpoint | 1–2 周 | 第二种策略接入统一推理接口的演示 |
| 6 | P6：Benchmark 与 GPU 分析 | p50/p95/p99、显存、利用率、报告和复现 | 1–2 周 | 可公开的性能报告和实验产物 |
| 7（可选） | P7：有限真机/HIL 验证 | driver、hold、限速、现场测试流程 | 视硬件时间 | 真机证据或有故障注入的 HIL 证据 |

预计主线约 10–12 周，每周 10–15 小时。若时间或算力紧张，完成 P0–P6 的仿真与远程推理部分已经能形成完整作品集；P7 属于扩展验收。

## 最终系统长什么样

完成后，系统由小项目逐步拼成以下结构：

```text
                 ┌──────────── 模型与数据 ──────────────┐
                 │ LeRobotDataset / ACT / SmolVLA       │
                 └────────────────┬─────────────────────┘
                                  ▼
传感器或 LIBERO ─► Observation Adapter ─► Inference Adapter
                                             │
                             ┌───────────────┼─────────────────┐
                             ▼               ▼                 ▼
                           Sync             RTC          Remote Server
                             └───────────────┼─────────────────┘
                                             ▼
                                  Action Queue + Alignment
                                             ▼
                                  Safety Gate / Hold
                                             ▼
                                 LIBERO 或 Robot Driver
                                             │
                                  Trace / Metrics / Report
```

### 项目组合后的主数据流

```text
采集 observation
  → 校验 schema、时间戳和新鲜度
  → 发送同步或异步推理请求
  → 收到 action chunk
  → 检查 generation/sequence/deadline
  → 对齐到尚未执行的控制时刻
  → 检查动作范围与安全策略
  → 执行一个 tick 或 hold
  → 写入 trace 和 benchmark 指标
```

### 项目边界

- 模型由 LeRobot policy/checkpoint 提供；首版不从零训练大型 VLA。
- 仿真和真机分别报告；LIBERO 成功率只代表仿真 benchmark 结果。
- 首版目标是单机器人、单任务或小规模任务集合；暂不实现多租户、云端调度或产品化 UI。
- LeRobot 的 remote 路径优先调用其公开 API；自研代码负责统一接口、实验编排、trace 和对比分析。
- 硬件安全以 robot driver、限位、急停和现场监督为基础；模型与 Action Queue 不能替代这些机制。

---

# 阶段 0：先做一个不用机器人也能跑的闭环

## P0：Mock 推理闭环与统一数据接口

### 项目介绍

用假环境和假策略跑完整数据流：环境产生观测，策略返回动作序列，控制循环逐 tick 消费动作，并将全过程写入日志。这个小项目不依赖 GPU、LeRobot 或机器人，先把系统骨架和数据契约固定下来。

### 目标

- 一条命令运行 100–500 个控制 tick。
- 控制循环、策略和环境通过接口解耦。
- 用可替换的 fake clock 测试 deadline 和动作时间对齐。
- 日志可以还原每个 tick 收到了什么、执行了什么、为何 hold。

### 范围

实现 `MockEnvironment`、`MockPolicy`、`ControlLoop`、`ActionQueue` 初版、`TraceRecorder`。动作可以是简单的二维/七维向量；本项目不调用真实模型。

### 实现步骤

1. 创建 Python 包、配置入口、pytest 和 lint/type check 配置。
2. 定义 `ObservationFrame`、`ActionChunk`、`ActionSchema`。
3. 定义 `EnvironmentAdapter`、`PolicyAdapter` 两个 protocol。
4. 实现 `FakeClock` 与单调时钟接口，生产代码禁止用墙钟判断超时。
5. Mock policy 每次返回固定长度的 action chunk；mock environment 能 reset、capture、apply 和 hold。
6. ControlLoop 每个 tick 先读取已完成的推理结果，再选动作或 hold。
7. 将事件写成 JSON Lines；生成一个最简 `summary.json`。

### 最小接口

```python
@dataclass(frozen=True)
class ObservationFrame:
    generation_id: int
    tick_id: int
    capture_mono_ns: int
    features: Mapping[str, np.ndarray]
    task: str | None

@dataclass(frozen=True)
class ActionChunk:
    generation_id: int
    sequence_id: int
    source_capture_mono_ns: int
    received_mono_ns: int
    dt_ns: int
    actions: np.ndarray  # [horizon, action_dim], float32
    action_schema_id: str

class PolicyAdapter(Protocol):
    def predict_chunk(self, observation: ObservationFrame) -> ActionChunk: ...

class EnvironmentAdapter(Protocol):
    def reset(self, seed: int) -> ObservationFrame: ...
    def capture(self) -> ObservationFrame: ...
    def apply(self, action: np.ndarray) -> None: ...
    def hold(self) -> None: ...
```

### 交付物

- 可安装的 Python package 和一条 mock quickstart 命令。
- 一段可读的 JSONL trace。
- 单元测试：reset、正常推理、队列为空、超时结果、非法动作。
- 一张简单时序图或 README 中的事件样例。

### 验收条件

- 新环境安装依赖后，不需要 GPU/相机即可运行 mock demo。
- trace 能区分 observation、inference request/result、action dispatch、hold。
- queue 空时明确调用 `hold()`，不会默认为发送零向量。
- reset 会递增 `generation_id`，旧 generation 的迟到结果会被丢弃。

### 本项目学到什么

Python 工程结构、接口设计、单调时钟、异步系统测试、结构化日志和最小 CI。

---

# 阶段 1：把仿真环境接入闭环

## P1：LIBERO Environment Adapter

### 项目介绍

将 P0 的 mock 环境替换成 LIBERO。只接环境，不急着训练模型；先能可靠 reset、取 observation、发送 action、拿到 terminated/success 信息并保存 episode 记录。

### 目标

- 使用固定 task、固定 seed 完成一段 episode。
- 将 LIBERO 图像、机器人状态、reward/done 转换为项目内部契约。
- 明确控制模式和 action 坐标，避免后续把 relative 与 absolute action 混用。

### 实现步骤

1. 按 LeRobot 文档安装 LIBERO 额外依赖，使用 Linux 环境。
2. headless 环境设置 MuJoCo 渲染后端，例如 `MUJOCO_GL=egl`；先通过官方环境 smoke test。
3. 实现 `LiberoEnvironmentAdapter.reset(seed)`、`capture()`、`apply(action)`、`close()`。
4. 配置 task suite、task id、camera 尺寸、控制模式、最大 episode 步数。
5. 固定一个 spatial task 和 1–3 个 seed，先接 `MockPolicy` 或 replay action。
6. 每个 episode 保存配置、seed、success、步数、视频路径和 trace 文件名。

### 交付物

- `configs/env/libero.yaml`。
- `src/.../adapters/libero.py`。
- `scripts/run_libero_smoke` 或统一 CLI 的 `run --env libero`。
- 一个 episode 的视频、trace、summary 和 manifest。

### 验收条件

- 单任务 reset/step/close 可重复运行，不遗留模拟器资源。
- observation shape、dtype、颜色空间和 action shape 在启动时打印并验证。
- 改变 seed 会写入 manifest；同 seed 的运行可以对照。
- 日志明确记录 LIBERO task、relative/absolute control mode、reset 类型。

### 不接入真机的处理

P1 到 P6 都可以用仿真完成。若 LIBERO 安装环境不稳定，先运行官方最小 task example，固定 Linux、MuJoCo 和 LeRobot 版本，再继续写 adapter。

---

# 阶段 2：接入第一个真实策略

## P2：ACT Policy Adapter 与同步推理基线

### 项目介绍

以 ACT 作为首个 learned policy，建立真实模型从 checkpoint 到 LIBERO rollout 的路径。同步推理用于形成简单、可比较的参考基线。

### 目标

- 模型适配层读取 checkpoint 的输入输出 feature 规格。
- 同步调用模型生成 action chunk，经过统一处理后逐步执行。
- 记录预处理、模型 forward、后处理、环境 step 的耗时。

### 实现步骤

1. 固定 LeRobot 版本、checkpoint revision 和数据来源；将 revision/hash 写入 manifest。
2. 接入官方 ACT policy 和 processor，不重写 ACT 模型内部实现。
3. 在 adapter 里校验图像/state feature 名称、维度、dtype、device。
4. 将预测的 torch tensor 转换到项目内部规范；保留模型坐标与执行坐标的说明。
5. 建立 `sync` inference backend：每次预测返回 action chunk，控制循环按频率消费。
6. 先在单任务 1–3 episodes 验证流程，再逐步扩大到多个 task/seed。

### 交付物

- `ActPolicyAdapter`。
- `configs/policy/act.yaml` 与 `configs/inference/sync.yaml`。
- 一份 ACT + LIBERO 的最小结果表。
- checkpoint、环境、seed、运行配置的 manifest。

### 验收条件

- ACT checkpoint 可以加载；模型预处理与输出后处理完整。
- 同一配置可重复启动，不依赖当前 shell 的隐式参数。
- 成功率、episode 时长、推理耗时均可从结果中读取。
- 失败时记录异常阶段，例如 model load、schema validation、forward、action validation。

### 接入原则

ACT 作为第一种策略，主要用来验证图像+状态输入、动作分块和推理适配器。训练规模与模型效果不是此阶段的验收重点。ACT 官方指南：[ACT for LeRobot](https://huggingface.co/docs/lerobot/act)。

---

# 阶段 3：独立验证异步动作执行

## P3：Action Queue、时间对齐与故障注入

### 项目介绍

用可控的 mock inference 延迟独立研究“模型返回 chunk 时，执行器已经走到哪里”。在接入复杂远程服务前，先验证队列、chunk 合并、超时和迟到结果的规则。

### 目标

- 推理在后台执行时，控制 tick 不等待模型返回。
- 新 chunk 只影响尚未执行的动作区间。
- 延迟、抖动、丢包、乱序、reset 与停止都能通过测试复现。
- 指标能展示动作连续性、buffer 覆盖时间和 observation age。

### Action Queue 数据结构

建议以 `target_tick` 为 key 的有限 ring buffer。ControlLoop 是 queue 唯一写入者；推理线程通过线程安全 mailbox 投递结果，控制 tick 开始时集中处理结果，减少并发锁和状态竞争。

### 必须成立的不变量

1. 已经 dispatch 的 action slot 不可修改。
2. 同一 generation、同一目标 tick 只有一个可执行动作。
3. reset/stop 会递增 generation，旧 generation 动作不能进入 queue。
4. 已过 deadline 或早于当前可执行时刻的 action 不得 dispatch。
5. shape、dtype、schema 错误以及 NaN/Inf 整个 chunk 拒绝。
6. MVP 的 `max_inflight=1`，不允许无界请求排队。
7. buffer 为空时调用明确的 hold 策略；不能把零向量当 hold。

### 时间对齐规则

- deadline、request age、action target time 均用执行端的单调时钟。
- 记录 `capture_mono_ns`、`request_mono_ns`、`result_mono_ns`、`dispatch_mono_ns`。
- 远端时钟不需要参与本地 dispatch 决策。远端返回 request identity；客户端根据本地收到结果的时间安排未来动作。
- 每个 chunk 带 `generation_id`、`sequence_id`、`source_capture_mono_ns` 和 action schema id。
- 新 chunk 回来后保留已执行动作；根据 `effective_from=max(result_arrival+guard, next_unexecuted_target)` 对齐后续动作。

### Chunk 合并选项

| 方法 | 作用 | 需要观察的代价 |
|---|---|---|
| `append` | 在旧 chunk 末尾追加 | 动作连续，但可能使用较旧观测。 |
| `aligned_replace` | 替换尚未执行的未来部分 | 对新观测响应快；需准确掌握执行位置。 |
| `blend` | 对连续关节动作做平滑过渡 | 可能改变预测动作；需要限位、速度和任务成功率验证。 |

### 实现步骤

1. 实现 `ActionQueue.admit/select/revoke/clear/coverage`。
2. 实现 `generation_id` 和 `sequence_id` 校验。
3. 实现 `DelayedMockPolicy`：可配置延迟、抖动、丢弃、乱序和结果异常。
4. 比较 sync 与 async 控制循环的 tick overrun、queue starvation、observation age。
5. 保存每个 chunk 的接收、准入、替换、丢弃原因与目标动作区间。
6. 绘制 queue coverage 随时间变化的曲线。

### 建议初始参数

| 参数 | 仿真起点 | 备注 |
|---|---:|---|
| 控制频率 | 20 或 30 Hz | 按环境和策略配置。 |
| `max_inflight` | 1 | 先让请求顺序简单可控。 |
| `max_buffer_seconds` | 1.5 秒 | 初始实验参数，之后依据 p95/p99 调整。 |
| `refill_seconds` | 0.6 秒 | 由实测时延和抖动余量调节。 |
| `action_timeout_s` | 依模型测量 | 不将示例值固化成默认真机策略。 |

### 交付物与验收

- Queue 单元测试 + 随机到达顺序的性质测试。
- 一套延迟故障注入配置。
- sync/async 对比 trace 和结果图。
- 测试证明迟到 chunk、旧 generation 和非法动作被拒绝；已 dispatch slot 不会被覆盖。

---

# 阶段 4：使用 LeRobot 官方推理机制

## P4：本地 RTC 与远程推理

### 项目介绍

把 P3 的时序问题连接到 LeRobot 的实际推理后端。先测同机本地 RTC，再测同机双进程/loopback remote，最后测试受信任 LAN 中的远程 GPU。

### 三种执行方式

| 模式 | 执行方式 | 用途 |
|---|---|---|
| `sync` | 当前控制流程等待模型推理 | ACT 的正确性和延迟基线。 |
| `rtc` | 同机后台推理，动作继续从已有 chunk 执行 | 同机低算力执行端与较慢推理。 |
| `remote` | 机器人/仿真 client 请求另一进程的 policy server | 执行端 GPU 不足或算力在实验室服务器。 |

### 实施顺序

1. 按当前固定版本的官方教程检查参数和 API；`main` 文档对应源码安装路径。
2. 本地运行 ACT 的 `sync` 与 RTC，记录端到端时延、queue coverage 和动作新鲜度。
3. 运行官方 policy server/client loopback 示例；先不引入网络变量。
4. 实现薄的 `RemoteEngineAdapter`：生命周期、schema 映射、trace identity 与结果上报。
5. 将 client 和 server 放在同一受信任 LAN 两台机器上；注入网络延迟，测完整 turnaround。
6. 测 server 退出、网络断开、timeout、旧响应和 client stop 的恢复行为。
7. 只有在同机/LAN 路径稳定后，才考虑 router 或云 GPU 等拓扑。

### 接口与配置约束

- 远程路径采用官方 `lerobot.inference`、`lerobot.remote_inference` 公共入口和 policy server/client。
- 远程传输使用官方的 admission/capability/schema 语义；动作单位和排列需要两端一致。
- 观测 state 使用 schema 声明的 `float32` 顺序；RGB 输入按声明的 `uint8` 尺寸和特征名传递。
- default raw RGB 可作为 baseline；JPEG 只能作为独立变量测量，需一起报告编码时间与任务结果。
- token、证书私钥和访问密钥从环境变量/secret store 读取，不提交到仓库。
- 首轮网络部署只运行在受信任 LAN；公共网络使用时必须完成认证与加密配置、网络和运动预算验证。

### 交付物

- `configs/inference/rtc.yaml` 和 `configs/inference/remote.yaml`。
- policy server 示例配置与本地一键启动脚本。
- loopback 与 LAN 的网络拓扑说明。
- 断联/超时测试结果和安全状态变化 trace。

### 验收条件

- 同一 ACT checkpoint 可由 sync、RTC、remote 路径调用。
- client stop 会撤销未执行的未来动作并请求 hold/停止。
- 网络异常时不会执行超时返回的陈旧 chunk。
- 可报告 remote round-trip p50/p95/p99、buffer starvation、observation age。

### 官方接口资料

当前 LeRobot 远程推理使用 `lerobot[remote]`、`lerobot-policy-server` 和 `lerobot-rollout --inference.type=remote` 等新路径；旧 `lerobot[async]` 与旧 import 路径已经被新接口替代。RTC 是同机本地异步选项。开始开发前要固定版本并照该版本的教程执行：[Remote Inference](https://huggingface.co/docs/lerobot/main/remote_inference)、[Policy Deployment](https://huggingface.co/docs/lerobot/main/inference)。

---

# 阶段 5：接入语言条件 VLA

## P5：SmolVLA Adapter

### 项目介绍

在 ACT 适配器之后接入 SmolVLA，验证统一执行系统能处理带语言条件和相机映射的策略。项目重点是 adapter、数据特征和模型部署，不要求从零预训练 SmolVLA。

### 实现步骤

1. 按固定 LeRobot 环境安装 `.[smolvla]` 额外依赖。
2. 选用公开 checkpoint 或已有微调 checkpoint，锁定 Hub revision。
3. 对照 checkpoint 的 camera/state/language feature 配置建立 `SmolVLAAdapter`。
4. 将 task prompt 作为明确配置字段传入策略；不要在控制循环内做隐式 prompt 拼接。
5. 训练、评估和 rollout 使用相同 `rename_map`；在 manifest 保存映射。
6. 先做模型加载和单次 prediction smoke，再跑单任务 rollout/benchmark。
7. 记录图像预处理、tokenization、模型 forward、动作后处理的独立耗时。

### 数据要求

- 同一任务要覆盖不同物体位置、背景和初始状态，不要只重复完全相同的演示。
- 若微调自有任务，先整理 demo 质量与 camera/state/action 对齐；再做训练预算规划。
- 数据集和模型的相机名称要与 checkpoint schema 对齐；camera rename mismatch 需要显式报错。

### 交付物

- `SmolVLAAdapter` 和 policy config。
- 经过验证的 `rename_map`。
- 至少一份 checkpoint 加载和推理记录。
- ACT vs SmolVLA 可比实验说明；若环境或动作空间不可比，明确列出差异。

### 验收条件

- checkpoint 能正常加载，model feature schema 可打印和校验。
- prompt、camera feature、state/action shape 一致地从配置传入。
- 同一推理模式下 trace 中能关联 prompt/task、模型耗时和 chunk identity。
- 完成一个最小评测运行；将训练数据/权重来源写入 manifest。

参考：[SmolVLA 官方指南](https://huggingface.co/docs/lerobot/main/smolvla)。该指南说明 SmolVLA 的专用依赖、feature 对齐、rename map 和微调流程；具体训练时间、batch 和显存需要在实际 GPU 上重新测量。

---

# 阶段 6：把实验做成可复现的性能报告

## P6：Benchmark Runner、GPU Profiling 与报告

### 项目介绍

将 P1–P5 的运行方式收敛成配置驱动的 benchmark runner。每个实验可复现输入、环境、模型和推理设置，并生成结果表、时延分布和队列分析。

### LIBERO 评测层级

| 级别 | 规模 | 目的 |
|---|---|---|
| Smoke | 1 task × 1–3 episodes | 检查依赖、schema、episode 生命周期。 |
| 快速回归 | 2 tasks × 3 seeds | 每次改动后验证执行正确性。 |
| 正式报告 | Spatial/Object/Goal/Long suites | 形成有代表性的多任务仿真结果。 |
| 性能压力 | 固定 task × 至少 30 episodes/多种网络延迟 | 分析尾延迟、queue starvation 和稳定性。 |

LeRobot LIBERO 文档当前介绍五个套件、共 130 个任务；标准多套件评测包括 Spatial、Object、Goal 和 Long（CLI 名称 `libero_10`），每任务常用 10 episodes。正式复现使用 hard reset；soft reset 更快，但结果可能与 hard reset 有差别。参考：[LeRobot LIBERO](https://huggingface.co/docs/lerobot/libero)、[LIBERO 官方仓库](https://github.com/Lifelong-Robot-Learning/LIBERO)。

### 必报指标

| 指标 | 计算方式/用途 |
|---|---|
| `success_rate` | 成功 episode / 总 episode；附 episode 数和置信区间。 |
| `inference_ms_p50/p95/p99` | 推理请求到结果到达；拆开预处理、forward、后处理。 |
| `remote_rtt_ms_p50/p95/p99` | client 发出请求到 client 收到结果。 |
| `tick_overrun_rate` | 执行 tick 超过目标周期的比例。 |
| `queue_coverage_ms` | 当前时间之后可执行的动作覆盖时长。 |
| `starvation_rate` | 需要执行但无可用动作的 tick 占比。 |
| `observation_age_ms` | action dispatch 时间减去 observation capture 时间。 |
| `late_chunk_drop_rate` | 超时、过期、旧 generation 等被拒 chunk 比例。 |
| `gpu_mem_peak` | 模型/推理阶段峰值显存。 |
| `gpu_utilization` | 指定采样周期内 GPU 利用率，并写明采样方法。 |
| `hold_count/fault_count` | 按 episode、错误原因分组统计。 |

### GPU 与系统性能测量步骤

1. 先做 20–50 个 warm-up 请求，单列启动耗时，避免首次 CUDA 初始化扭曲结果。
2. 使用单调时钟记录 capture、request、result、dispatch；最终以端到端指标为主。
3. 在隔离 profiling run 中用 PyTorch Profiler 或 Nsight Systems 深挖；profiling 开销不计入正式结果。
4. GPU kernel 区间可用 CUDA Event 测量；必要时在测量边界同步 CUDA。不要在实时控制热路径中随意同步。
5. 传输实验记录 RGB 原始字节、编码耗时、网络耗时和成功率变化。
6. 每个结果绑定 git commit、LeRobot/LIBERO 版本、checkpoint revision、resolved config、seed 和机器/GPU 信息。

### 必做对比实验

| 实验 | 对比变量 | 固定项 |
|---|---|---|
| B0 | Mock policy 延迟 0/固定/随机 | queue 规则与环境 seed |
| B1 | ACT：sync vs RTC | checkpoint、task、seed、动作模式 |
| B2 | ACT：RTC vs remote loopback/LAN | checkpoint、图像编码、task、seed |
| B3 | 多个 `refill_seconds` | checkpoint、网络配置、task |
| B4 | append vs aligned replace vs blend | 同一组延迟注入和 chunk |
| B5 | raw RGB vs JPEG | 同一 remote 环境和图像分辨率 |
| B6 | ACT vs SmolVLA | 同类任务；报告输入/训练数据差异 |

### 交付物

- `benchmark` CLI：按 YAML 配置启动任务矩阵。
- 每个 run 的 manifest、resolved config、trace、episode summary。
- Markdown 报告和 CSV/JSON 结果表。
- 至少两张图：推理/round-trip 时延分布；queue coverage 与 observation age 随时间变化。

### 验收条件

- 干净环境能够按 README 复现 smoke benchmark。
- 任何结果表格行能追溯到 run id、配置和 trace。
- 结果报告区分模型效果与系统性能，标明 episode 数、seed、reset 方式和限制。
- 同配置重复运行 3 次并汇报波动，不只挑选最好的一次。

---

# 阶段 7（可选）：有限真机或 HIL 项目

## P7：Robot/HIL Adapter 与现场集成验收

### 项目介绍

如果能预约到机械臂，使用少量固定时间做部署链路验收；如果没有真机，用 HIL mock driver、数据回放和网络故障注入完成相同的软件边界测试。

### 真机实施顺序

1. 先确认 driver、关节命名/顺序、弧度或角度、夹爪量纲、最大速度和 position hold 语义。
2. 在无动作模式读取相机和状态，验证同步、shape、单位和采集频率。
3. 使用人工示教和低速空载动作确认方向、限位和停止路径。
4. 用固定 checkpoint、固定配置，在有人现场监督的条件下运行少量 episode。
5. 先故障注入验证 timeout、network disconnect 和 stop；再执行正常任务。
6. 保存完整 run sheet、配置、视频、trace 和失败说明。

### 真机前的准入条件

- 现场急停可用，测试者清楚它的位置和停止范围。
- driver 的 hold 与 stop 行为已单独测试。
- action 单位、排列、hard limits 和速度限制都已在配置明确声明。
- 长时间无结果、断网、模型异常时，测试流程有明确回退行为。
- 每次 rollout 由现场人员主动 arm；停止可以撤销 queue 中尚未执行的动作。

### 无真机替代路线

- 在 LIBERO 上跑所有策略与 benchmark。
- 通过 `DelayedMockPolicy` 注入 latency、jitter、packet loss、乱序和模型异常。
- 对公开 LeRobotDataset 做 observation replay，评测预处理和推理时延。
- 实现 `MockRobotDriver`，模拟 ack、hold、限位触发和通信断连。
- 报告中清楚标注仿真、回放或 mock 结果，不将其描述为真机验证。

### 真机时间有限时的安排

- 真机前完成安装、仿真 rollout、配置冻结、相机测试、checkpoint 下载和现场 run sheet。
- 预约窗口只做校准、空载、低速和预定短任务；现场不临时升级依赖或训练模型。
- 出现 schema、限位、网络、queue starvation 等安全相关异常时，结束 rollout 并返回仿真排查。

---

# 全项目共用的工程约定

## 一、推荐仓库结构

```text
VLA-Inference-Engine/
├── README.md
├── pyproject.toml
├── uv.lock                      # 若选 uv；也可选唯一的其他锁文件
├── configs/
│   ├── env/{mock,libero,robot}.yaml
│   ├── policy/{mock,act,smolvla}.yaml
│   ├── inference/{sync,rtc,remote}.yaml
│   ├── safety/default.yaml
│   └── benchmark/libero_smoke.yaml
├── src/vla_inference_engine/
│   ├── contracts/               # ObservationFrame / ActionChunk / schema
│   ├── adapters/                # mock / LIBERO / LeRobot / policy
│   ├── inference/               # sync / RTC / remote 封装
│   ├── control/                 # loop / clock / ActionQueue / safety
│   ├── telemetry/               # JSONL trace / metrics
│   ├── benchmark/               # runner / aggregation / report
│   └── cli.py
├── tests/{unit,property,contract,integration}/
├── docs/{architecture,runbooks,experiments}/
├── artifacts/                   # 小型图表与报告
└── runs/                         # 原始大文件，加入 .gitignore
```

## 二、统一 Observation / Action 契约

| 项目 | 约定 |
|---|---|
| 图片边界格式 | HWC、RGB、`uint8`；模型内部转换由 policy processor 处理。 |
| 状态向量 | `float32`，顺序由 checkpoint feature config 声明。 |
| Action chunk | `[horizon, action_dim]`；带 `dt_ns`、generation、sequence 和 schema id。 |
| 机器人语义 | 关节顺序、弧度/角度、绝对/相对、夹爪范围显式配置。 |
| 时间 | 控制路径使用本地单调时间；墙钟只用于人类可读时间。 |
| 结果身份 | request/session/generation/sequence 必须可关联；不按“最后返回”直接执行。 |
| 错误处理 | 错误 feature、NaN/Inf、超时、过期和旧 generation 都返回稳定错误枚举。 |

推荐事件类型：`episode_reset`、`observation_captured`、`inference_requested`、`inference_completed`、`chunk_admitted`、`chunk_rejected`、`action_dispatched`、`action_held`、`safety_transition`、`episode_done`。

## 三、异常处理约定

| 异常 | 处理动作 | 记录项 |
|---|---|---|
| inference deadline 超时 | 丢弃迟到结果；buffer 用尽则 hold/stop | request id、时延、queue coverage |
| 网络断开 | revoke session/generation，停止接收未来结果 | link 状态、最后 ack、故障状态 |
| queue 空 | 调用明确声明的 driver hold | hold 起止时间、发生频率 |
| NaN/Inf、shape/schema 错误 | 整个 chunk 拒绝；真机进入 fault | 原因、schema id、模型版本 |
| action 越限/速度过大 | 不 dispatch，触发降级或 stop | 原动作、限制值、决策 |
| 图像/state 不同步 | 不构造新推理请求，按既定策略消耗有限 buffer | image/state 时间差 |
| trace writer 阻塞 | 热路径继续运行，必要时丢诊断事件并计数 | dropped telemetry count |

## 四、配置与复现规范

每次运行生成 `runs/<run_id>/manifest.json`，包含：

```json
{
  "run_id": "日期时间_实验名_commit短hash",
  "git_commit": "...",
  "dirty_worktree": false,
  "python_version": "...",
  "platform": "...",
  "cuda_version": "...",
  "gpu_name": "...",
  "lerobot_version_or_sha": "...",
  "libero_version_or_sha": "...",
  "checkpoint_revision_or_hash": "...",
  "dataset_revision": "...",
  "resolved_config_sha256": "...",
  "seeds": [0, 1, 2],
  "reset_mode": "hard",
  "started_at_utc": "..."
}
```

- 保存完整 resolved config；不要只保存命令行覆盖字段。
- 权重、原始视频、token、证书私钥和可能带个人信息的数据不进入 Git。
- 依赖升级单独开分支，记录旧/新版本并重跑合同测试和 smoke benchmark。
- Markdown 报告、图表和 CSV 从 run artifacts 生成，避免手工抄数字。

## 五、Git 与 GitHub 工作流

### 分支命名

| 类型 | 分支例子 |
|---|---|
| 功能 | `feat/action-queue` |
| 修复 | `fix/stale-result-drop` |
| 文档 | `docs/libero-setup` |
| 实验 | `exp/refill-sweep-20261027` |
| 依赖升级 | `chore/lerobot-version-update` |

### 提交与 PR

- 使用简洁、可读的 commit，例如 `feat(queue): add generation-aware replacement`。
- PR 描述：解决什么、改了哪些模块、如何运行/验证、结果链接、性能影响和回滚方法。
- `main` 保护：通过 CI 后合并；每个小项目完成时打一个 tag 或发布一个里程碑 release。
- 先用小 PR：一次只交付一个可验收增量，例如 `P0 contracts`、`P1 LIBERO smoke`。
- 建议在 GitHub 建里程碑：P0–P1、P2、P3、P4、P5、P6、P7。

## 六、测试层级

| 测试层 | 测什么 | 运行频率 |
|---|---|---|
| Unit | clock、schema、ActionQueue、metrics | 每次改动 |
| Property | 随机 chunk 顺序/reset 后的不变量 | 每个 queue PR |
| Contract | PolicyAdapter、LeRobot schema、remote loopback | 依赖升级和合并前 |
| Integration | LIBERO 单 task smoke episode | PR/每日 |
| Benchmark regression | 固定 task 下 p95、starvation 和成功率偏移 | 发布前 |
| Robot acceptance | driver、hold、限位、stop、现场急停 | 有真机窗口时人工监督 |

---

# 12 周执行表

| 周 | 项目 | 本周可交付结果 |
|---:|---|---|
| 1 | P0 | repo、contracts、FakeClock、MockPolicy、JSONL trace、CI |
| 2 | P0/P3 | ActionQueue 初版、generation/sequence、关键不变量测试 |
| 3 | P1 | LIBERO 环境 smoke、固定 seed、视频和 manifest |
| 4 | P2 | ACT checkpoint 加载、同步推理、单任务 rollout |
| 5 | P2/P3 | observation/action 校验、延迟注入、queue 覆盖分析 |
| 6 | P3 | append/replace/blend 对比和故障测试 |
| 7 | P4 | 本地 RTC、trace 与 sync 对比 |
| 8 | P4 | remote loopback、LAN 推理、断联和 deadline 测试 |
| 9 | P5 | SmolVLA load/predict smoke、rename map 配置 |
| 10 | P6 | 多 seed benchmark runner 与自动报告 |
| 11 | P6/P7 | 性能报告；有真机则跑预演，否则补 HIL 故障测试 |
| 12 | 汇总 | 整理 README、架构图、结果图、demo 和发布 tag |

每周执行节奏：确定一个可以被测试验证的目标；实现一个最小切片；补测试和 trace；固定配置运行；记录结果、依赖和下一周风险。每周最多安排 2–3 个主要交付项。

---

# 首周任务单

## Day 1：创建 P0 仓库

- [ ] 建立 GitHub 仓库 `VLA-Inference-Engine`。
- [ ] 创建 README、许可证、`.gitignore`、`pyproject.toml`、CI 工作流。
- [ ] 明确 Python 版本和唯一依赖锁定方案。
- [ ] 写 `ObservationFrame`、`ActionChunk`、`ActionSchema`。

## Day 2：时钟与队列

- [ ] 写 `MonotonicClock` protocol 和 `FakeClock`。
- [ ] 写单线程 `ActionQueue` 的 `admit/select/revoke/coverage`。
- [ ] 测试正常消费、队列为空、reset 清队列、旧 generation 丢弃。

## Day 3：Mock 端到端运行

- [ ] 写 `MockEnvironment`、`MockPolicy`、`ControlLoop`。
- [ ] 运行 100 ticks，输出 JSONL trace。
- [ ] 生成 `summary.json`，包含延迟分位数、hold 次数和 queue minimum。

## Day 4：把 demo 做成别人能跑的项目

- [ ] README 加 mock quickstart 和一段预期输出。
- [ ] CI 执行 lint、类型检查和 unit tests。
- [ ] 用一个 PR 合并 P0 最小闭环，并打 `p0-mock-loop` tag。

## Day 5：验证 LIBERO 安装路径

- [ ] 创建隔离的 LIBERO 环境，锁定 LeRobot 和 LIBERO revision。
- [ ] 按官方教程运行单个环境创建和 step 示例。
- [ ] 记录 Linux、MuJoCo、EGL/显卡驱动和安装方式。
- [ ] 不把安装排错扩展成 P1 adapter；先保证 smoke example 可运行。

**首周结束标准**：mock 闭环能独立运行，关键数据结构和 queue 行为有测试，trace 可解释，LIBERO 的安装风险已知。

---

# 算力与预算安排

| 可用资源 | 适合完成的项目 | 策略 |
|---|---|---|
| CPU | P0、queue、mock、trace、CI、部分 LIBERO/数据处理 | 不在 CPU 上训练大型 VLA；先完成无需 GPU 的部分。 |
| 8–12 GB VRAM | ACT 推理/小规模训练，部分 LIBERO 仿真 | 优先 ACT；训练 batch 和分辨率按显存测量。 |
| 16–24 GB VRAM | ACT、SmolVLA 推理和常规开发 | SmolVLA 先以 batch 1 做加载与 inference smoke。 |
| 云端 40–80 GB VRAM | SmolVLA 微调、多 seed 正式实验 | 配置先通过本地 smoke；云 GPU 设置预算、超时和自动关机。 |

遵循先低成本验证后放大规模：mock → 单任务仿真 → 固定 checkpoint 推理 → 小规模 benchmark → 多 seed 正式运行 → 再考虑微调。ACT、SmolVLA 的官方训练时间仅作估算；本机实测应写入实验报告。

---

# 学习资料按项目配套

| 学习内容 | 对应项目 | 推荐资料 |
|---|---|---|
| Python、typing、pytest、Git、CI | P0 | [pytest 文档](https://docs.pytest.org/)、[GitHub Skills](https://skills.github.com/) |
| LeRobot 数据集和 policy 结构 | P1/P2 | [LeRobot 官方文档](https://huggingface.co/docs/lerobot/index) |
| ACT 模型和模仿学习 | P2 | [ACT 指南](https://huggingface.co/docs/lerobot/act) 与 ACT 原论文 |
| LIBERO task、控制模式、评测 | P1/P6 | [LeRobot LIBERO](https://huggingface.co/docs/lerobot/libero)、[LIBERO 仓库](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| 实时调度、deadline、抖动和 backpressure | P3/P4 | 本路线图中 Action Queue 与时钟部分；结合实际 trace 学习 |
| RTC、remote、server/client | P4 | [LeRobot Inference](https://huggingface.co/docs/lerobot/main/inference)、[Remote Inference](https://huggingface.co/docs/lerobot/main/remote_inference) |
| SmolVLA、相机 feature、数据微调 | P5 | [SmolVLA 指南](https://huggingface.co/docs/lerobot/main/smolvla) |
| GPU profiling | P6 | [PyTorch Profiler 教程](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) |
| 真机和 driver 安全 | P7 | 对应 robot 官方手册、LeRobot robot guide 和现场安全流程 |

资料阅读完成的标准：在对应项目中留下一个运行结果、测试、接口说明或实验图，避免只收藏链接。

---

# 求职作品集输出

完成 P0–P6 后，在仓库首页放这六项：

1. **项目目标**：统一运行 ACT/SmolVLA 的同步、本地 RTC 和远程推理。
2. **架构图**：环境、policy adapter、queue、safety gate、trace 的关系。
3. **快速运行**：mock quickstart 和 LIBERO smoke 命令。
4. **结果表**：每种模式的成功率、p95/p99 时延、queue starvation、observation age。
5. **可复现证据**：固定配置、seed、checkpoint revision、run manifest 和 trace 示例。
6. **已知限制**：仿真与真机结果分开；写明运行硬件和还没验证的拓扑。

简历项目描述模板：

> 基于 LeRobot 与 LIBERO 构建 VLA 推理执行和评测项目，支持 ACT/SmolVLA 的同步、本地 RTC 与远程推理；实现带 generation/sequence 校验的 Action Queue、单调时钟对齐、timeout/hold/fault 处理，并通过多 seed benchmark 对比端到端 p95/p99、动作新鲜度、队列饥饿率与任务成功率。项目结果：`[填写真实测量值和链接]`。

只有在相应实验真实完成后填写模型、机器人和性能数字。若只在 LIBERO 上运行，明确写“仿真”。

---

# 风险与降级决策表

| 问题 | 先做什么 | 降级完成方式 |
|---|---|---|
| LeRobot 版本或 API 变化 | 固定 release/tag/SHA，按版本检查 CLI help | 先只支持一个版本；升级单独开分支和兼容测试。 |
| LIBERO/MuJoCo 渲染安装失败 | 验证官方最小 example、Linux/EGL 配置 | 先完成 P0、P2 数据/模型 smoke，再回到仿真安装问题。 |
| 显存不足 | 缩小 batch、先 ACT、先推理后训练 | SmolVLA 使用云 GPU；不让训练阻塞系统工程主线。 |
| 没有真机 | 加强 LIBERO 和 HIL mock | 对外明确为仿真/回放结果。 |
| 真机只有短时可用 | 提前冻结软件、模型、配置和 run sheet | 只做校准、空载、低速、短 rollout；现场不排查依赖。 |
| 远程网络抖动大 | 先测 loopback，再测 LAN，注入 jitter/loss | 以 RTC 或本地 sync 作为退路；报告 remote 限制。 |
| 任务成功率波动 | 增加 seeds/episodes，核对初始状态与 reset | 同时报告区间和 episode 数，避免选择性报告。 |
| 范围变大 | 检查是否属于 P0–P7 交付 | ROS、UI、多机器人等进 backlog，不插入主线。 |

---

# 最终发布验收表

## P0–P6 核心版本

- [ ] 仓库能从干净环境安装并运行 mock demo。
- [ ] LIBERO 单任务 smoke 能输出视频、trace、summary 和 manifest。
- [ ] ACT checkpoint 可通过统一 adapter 做同步推理。
- [ ] Action Queue 对迟到结果、reset、非法动作有自动化测试。
- [ ] 至少有 RTC 和 remote 中的一种异步路径跑过；若都完成，分别报告。
- [ ] SmolVLA 至少通过 checkpoint load/predict 和 schema 校验。
- [ ] benchmark 报告包含 seed、episode 数、版本、reset 类型和延迟分位数。
- [ ] README 能让陌生开发者按步骤复现 mock 和 LIBERO smoke。

## P7 真机扩展

- [ ] driver 坐标/单位/hold/限位已核对。
- [ ] 急停与 stop 路径已在现场测试前确认。
- [ ] 网络断开与 inference timeout 已经在仿真/HIL 里测试。
- [ ] 真机结果单独标注，不与 LIBERO 成功率合并。

---

## 推荐执行顺序

```text
P0 Mock 推理闭环
  → P1 LIBERO 环境
  → P2 ACT 同步基线
  → P3 Action Queue 与故障注入
  → P4 RTC / Remote
  → P5 SmolVLA
  → P6 Benchmark / GPU 报告
  → P7 真机或 HIL（可选）
```

每个 P 都作为独立的小项目完成一次“代码 → 测试 → 运行结果 → README 更新”。这样即使中途停止，已经完成的部分仍能单独运行和展示；全部完成后，它们组成 VLA-Inference-Engine。
