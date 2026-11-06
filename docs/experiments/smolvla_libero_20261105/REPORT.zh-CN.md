# SmolVLA + LIBERO 首个子项目阶段结项报告

记录日期：2026-11-05。正式矩阵在本项目完成，训练发生于原项目。本文记录已经发生的实验；完整配置快照见 [train_config.json](train_config.json)，指标、模型结构核对、来源路径与 SHA256 见 [experiment_record.json](experiment_record.json)。原始视频、训练权重和数据保留在 Git 外部。

## 1. 两个 checkpoint 的来源

项目标签 `smolvla_libero_paper` 指从 Hugging Face 下载的公开 checkpoint `HuggingFaceVLA/smolvla_libero`，固定 revision 为 `6721902bc4d61e50a3bfdb11dfb4cb626f05d102`。资产清单及权重 SHA256 与下载清单一致；本地绑定路径为 `artifacts/models/smolvla_libero_paper`。

“论文权重”是项目内的简称。该快照模型卡没有公布可直接绑定的 LIBERO 分数与完整训练记录，因此不能仅凭名称认定它就是论文 Table 2 的同一个实验产物。论文表格的公开分数与这份 checkpoint 在本机重测的分数需要分别展示。

本地 `smolvla_libero_100k` 来自原项目 `outputs/train/smolvla_libero_100k_b64/checkpoints/100000/pretrained_model`，由本机训练生成。本次测评使用最终 100000 步 checkpoint，未按测评结果选择其他步数。

**本地训练没有加载公开 LIBERO checkpoint，也没有加载 `lerobot/smolvla_base` 的整套 VLA 权重。**保存的配置中 `policy.pretrained_path=null`、`resume=false`；运行时命令使用 `--policy.type=smolvla` 和 `--policy.load_vlm_weights=true`，日志明确记录加载 `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`。本机存在 smolvla_base 下载文件不代表本次训练使用了它。

实际链路为：

```text
预训练 SmolVLM2-500M-Video-Instruct（保留前 16 层文本层并冻结 VLM）
  + 新建动作专家、state/action 投影和动作时间 MLP
  → 使用 lerobot/libero 离线演示训练 100000 次优化器更新
  → 保存本地 SmolVLA checkpoint 及前/后处理器
  → 在实时 LIBERO 仿真中执行三 seed 正式测评
```

源码依据：固定 LeRobot 的 `policies/factory.py::make_policy` 在没有 `pretrained_path` 时新建 policy；`smolvla/smolvlm_with_expert.py` 对 VLM 调用 `from_pretrained`，对动作专家调用 `AutoModel.from_config`；`smolvla/modeling_smolvla.py` 新建 state/action 投影与时间 MLP。`train_expert_only=true` 冻结整套 VLM，`train_state_proj=true` 允许 state 投影训练。它属于基于预训练 VLM 的机器人策略训练，完整模型并非全部随机初始化。

## 2. 实际训练超参数

以最终 checkpoint 保存的 `train_config.json` 为主，并与运行中的进程 argv、训练日志和运行时审计交叉核对，不用今天脚本的默认值代替历史实际配置。

| 项目 | 本次实际值 |
|---|---|
| 训练运行名 | `smolvla_libero_100k_b64` |
| 训练 seed | `1000`；仅一次训练 seed，评测的三个 seed 不等于三次训练 |
| 数据 | `lerobot/libero`，revision `a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4` |
| 数据规模 | Panda，40 任务、1693 episode、273465 frame、10 fps；使用全部 episode |
| 优化器更新 | `100000` |
| batch | 单卡物理 batch `64`，梯度累积 `1`，全局 batch `64` |
| 优化器 | AdamW，LR `1e-4`，betas `(0.9, 0.95)`，eps `1e-8` |
| 权重衰减 / 梯度裁剪 | `1e-10` / norm `10` |
| 学习率调度 | cosine decay with warmup；warmup `1000`，decay steps `100000`，最终 LR `2.5e-6` |
| 初始化 VLM | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，加载预训练权重 |
| 可训练模块 | 动作专家、state/action 投影、动作时间 MLP；VLM 冻结 |
| 参数量（训练日志） | 总计 `450046176`；可训练 `99880992`，约 `22.19%` |
| 文本层 / 专家层 | `16 / 16`；expert width multiplier `0.75`，cross attention |
| 观测 / 动作 | 两路 `3×256×256` RGB、8D state、文本；7D action；`n_obs_steps=1` |
| 模型内部图像处理 | `resize_imgs_with_padding=[512,512]`；不要与环境原始图像大小混淆 |
| 归一化 | VISUAL=IDENTITY；STATE/ACTION=MEAN_STD；state/action 内部最大维度均为 `32` |
| 动作块 / 执行步数 / flow 步数 | `chunk_size=50` / `n_action_steps=1` / `num_steps=10` |
| 文本长度上限 | `48` |
| 图像增强 | 关闭（`dataset.image_transforms.enable=false`） |
| 混合精度 | accelerator `bf16`；policy 的 `use_amp=false` 不等于没有 accelerator 混合精度 |
| 编译 | policy `compile_model=true`、`compile_mode=max-autotune`；accelerator compile 关闭 |
| 数据加载 | `num_workers=4`、prefetch `4`、persistent workers、`spawn`、`torchcodec` |
| 确定性设置 | `cudnn_deterministic=true` |
| EMA / PEFT | 关闭 / 未使用 |
| 训练期间仿真评测 | 关闭：`env_eval_freq=0`、`eval_steps=0`；正式测评在训练结束后单独执行 |
| 保存 / 日志 | 每 `50000` 步 checkpoint；每 `10` 步指标；W&B offline |
| 硬件 | 单张 RTX 3090，24 GiB |

`chunk_size=50` 表示每次预测动作序列长度；本轮评测 `n_action_steps=1` 表示执行一个动作后重新观测推理。训练启用了编译，而本轮正式评测显式关闭了 `compile_model`。

训练日志从 2026-08-15 到 2026-08-16，优化器循环约 `21.509 h`；loss 从首个记录点 `2.488` 到最后 `0.092`，最小记录值 `0.069`。日志覆盖 100000 步，每 10 步一行，共 10000 行；汇总报告没有记录非有限值、致命异常或真实 CUDA OOM。50k/100k checkpoint 的历史验证及文件清单核验均通过。这些是训练过程证据，任务能力以仿真成功率衡量。

训练环境审计记录 LeRobot commit `8ae2354592ec4480c38134fb75350b8e7992e0e9`，当时 editable import 指向另一磁盘的同 commit、同 tree 干净副本；正式测评前已校正为本项目绑定的 vendor。训练配置没有显式固定初始化 VLM 的 Hub revision，不能把当前记录解释为已经证明可从网络逐位复现训练；后续重训应固定该 revision 并记录初始化资产哈希。

## 3. 本轮正式结果

每模型 4 套件 × 10 任务 × 10 episode × seed 1000/2000/3000，共 1200 episode；两模型合计 2400。使用相同固定初态、相同任务和相同控制参数，单环境同步执行。Spatial/Object/Goal/Long horizon 分别为 280/280/300/520。每模型六项基础输入保持一致：双相机、语言、8D state、7D relative action、flow steps 10、每次执行动作 1 步。

| 套件 | 公开 checkpoint 本机测评 | 本地 100k checkpoint | 本地减公开 checkpoint |
|---|---:|---:|---:|
| Spatial | 226/300，75.33% | 233/300，77.67% | +2.33 个百分点 |
| Object | 265/300，88.33% | 258/300，86.00% | −2.33 个百分点 |
| Goal | 237/300，79.00% | 244/300，81.33% | +2.33 个百分点 |
| Long | 113/300，37.67% | 174/300，58.00% | +20.33 个百分点 |
| 合计 / 四套件平均 | 841/1200，70.08% | 909/1200，75.75% | +5.67 个百分点 |

两个 `formal_validation.json`、全部 6 个 seed 均通过；有 2400 个非空 episode 视频。结项核对重新检查模型哈希、summary 与验证报告的绑定、6 份 raw 文件哈希和视频数量。完整产物在 `results/raw/`、`results/<model>/` 和 `results/comparison/`；主日志是 `outputs/logs/eval_two_models_20261103_60s.log`，汇总/对比于 2026-11-05 约 17:15 完成。

配对统计以 seed/suite/task/episode index 配对。Spatial/Object/Goal 的 seed×task 聚类 bootstrap 差值区间均跨 0；Long 的差值为 +20.33 个百分点，95% 区间 `[+11.00,+29.00]`，task→初态敏感性区间 `[+3.67,+34.67]`。结论是本次 Long 优势较明确，前三套件差异不足以支持稳定优势；这一结论不包含跨训练 seed 的不确定性。

## 4. 为什么本地结果更高：事实与尚未解决的问题

本地总体更高主要来自 Long；Object 反而略低。没有证据表明本地优势来自“在公开 LIBERO 权重上继续微调”，因为实际初始化链路没有加载该 checkpoint。

两者存在已核实的配置和权重结构差异：

| 项目 | 公开 checkpoint | 本地 checkpoint |
|---|---|---|
| VLM 名称 | `SmolVLM2-500M-Instruct` | `SmolVLM2-500M-Video-Instruct` |
| VLM 文本层数（权重键核对） | 32 | 16 |
| 动作专家层数（权重键核对） | 32 | 16 |
| expert width multiplier | 0.5 | 0.75 |
| 专家首层 q_proj 权重 shape | `[960,480]` | `[960,720]` |
| safetensors 张量元素总数 | 604934176 | 450046176 |
| config 的 scheduler decay steps | 30000 | 100000 |

张量元素数由 safetensors header 的 shape 乘积求和，用于资产结构核对；公开模型没有运行时可训练参数审计，不能把它与本地“可训练参数量”混用。公开 config 的 `num_vlm_layers=0` 在当前代码中表示不截断，实际权重有 32 层，含义不是零层。公开 config 的 scheduler 值也不足以证明其实际只训练了 30000 步。

这些差异使本实验成为“两个具体 checkpoint 在同一 LIBERO 协议下的部署表现比较”。尚不能据此证明 VLM、层数、宽度或训练步数中的哪一项造成优势；这需要控制变量的训练消融。也不能把结果表述为“同结构同初始化的训练方法超越论文”。

公开 checkpoint 的 Long 成功率 37.67%，与项目固定论文基线 71% 相比，相对缺口为 `(71−37.67)/71≈46.95%`，触发项目预设的 40% 诊断门槛。它是相对缺口，绝对差为 33.33 个百分点。`results/diagnostics/index.json` 已索引 187 个失败视频并生成诊断计划；失败原因审查、根因确认及 seed 4000 诊断复测尚未完成。视频/任务失败不等于测试程序崩溃。seed 4000 若运行，应单独报告，不替换既有正式 seed 来抬高分数。

本机与论文的环境实现、horizon、processor、模型结构和训练条件仍需做完整对应核查。现有形式校验证明本轮按项目协议执行且证据完整，并不能证明已经复现论文所有实验细节或解释全部分数差异。

## 5. 首个项目的完成范围与结项口径

**可以对“SmolVLA 在 LIBERO 上的双模型测评与结果审计”作阶段结项。**交付包含：固定资产和环境记录、CLI/脚本入口、processor/数据/协议预检、两模型真实 smoke、完整 2400 episode 测评、视频和 raw 证据、seed 级续跑、正式重算验证、任务/套件统计与配对比较、差距诊断索引，以及本文的训练来源与超参数记录。

2026-11-03 已完成仓库 43 项测试、`ruff check src tools tests` 与 Shell 语法检查；该测试结论来自当日记录。本次为文档与证据快照归档，未修改执行代码，也未重跑 GPU 测评。

整个工作区已经实际连通“本地训练 → checkpoint/processor → VLA-Inference-Engine 正式测评 → 结果审计”。训练脚本和训练运行仍在原项目，本仓库的首个独立交付范围是测评，不能将它描述为已独立封装全部训练流程。

保留的后续事项：

1. 公开 checkpoint 的 Long 分数差距诊断尚未闭环，因此“论文分数严格复现”仍未验收。
2. 新机器不依赖本机原目录的 bootstrap、下载和资产绑定仍待端到端验证；当前复现证据限于本机。
3. 后续自有控制循环、远程推理、动作队列、RTC、延迟与吞吐、编译加速和真机接入属于下一项目，本次结果提供成功率参考。

源码和这些小型记录可作为 Git 首个项目的结项材料。权重、数据、视频及大型运行结果保持外部存储；本次未执行 Git commit、push 或发布。对外可描述为：“基于固定 LeRobot/LIBERO 环境完成 SmolVLA 公开权重与自训权重的可审计双模型评测，共 2400 episodes；自训权重总体 75.75%，公开权重本机结果 70.08%，并记录初始化和结构差异及论文分数差距。”
