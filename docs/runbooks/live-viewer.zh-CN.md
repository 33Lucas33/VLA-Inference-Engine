# LIBERO 实时交互演示台运行手册

## 启动

在仓库根目录准备好固定的 `libero` Conda 环境、LIBERO assets、LeRobot 源码，以及至少一个完整的 `artifacts/models/smolvla_libero_paper` 或 `artifacts/models/smolvla_libero_100k` checkpoint。安装本仓库及可选页面依赖：

```bash
conda activate libero
python -m pip install -e '.[live]' --no-deps
vla-engine live --host 127.0.0.1 --port 8765
```

若环境尚无 Starlette、Uvicorn 或 Pillow，先在同一环境安装 `pip install -e '.[live]'`。上面的 `--no-deps` 仅适用于已有固定环境且这三个包都已安装的机器。浏览器打开 `http://127.0.0.1:8765/`。端口被占用时用 `--port` 另选本机端口。首次加载模型约需数十秒，具体取决于磁盘与 GPU；页面先显示 LOADING，准备好后显示 READY 和初始双相机画面。

选择 suite、task、checkpoint、seed 和初态编号，点击“加载场景与模型”。任务说明来自 LIBERO 元信息。点击“模型接管”连续执行；“暂停”在当前步结束后生效；“继续”恢复；“执行一步”仅在 READY/PAUSED 时放行一步；“重置同一初态”新建同一配置的工作进程；“结束”关闭当前运行。页面刷新后尝试恢复当前活动会话；WebSocket 断开超过 3 秒后自动在步边界暂停。新建任务会结束旧会话。当前只允许一个活动会话。

## 记录与语义

每次加载或重置在 `runs/live/<session_id>/generation_<n>/` 写入 `config.json`、`worker.log`、`trace.jsonl`、`frames/*.jpg`、退出后的 `summary.json` 和双路 `image.mp4`、`image2.mp4`。MP4 以每仿真步一帧、固定 **10 fps** 回放，不表示真实墙钟速度；`summary.json` 记录 checkpoint SHA256、终态、视频名和编码错误。结果标为 `mode=interactive`、`formal_result=false`，**不参与正式 2400 episode 的分数**。页面给出三层 7D 数值：`normalized` 是策略输出，`processed` 是 checkpoint 后处理器输出，`environment` 是送入 `env.step` 的动作；它们不是直接测得的末端位姿。下方曲线显示当前页面收到的环境动作 XYZ 历史；刷新后曲线从新收到的步开始。两幅 JPEG 是仿真观测显示图，和模型实际读取的输入同源；每个工作进程只运行一个任务和 checkpoint。

若页面显示 ERROR，先打开该次 `worker.log`，再看同目录 `trace.jsonl` 的最后一条事件。服务由固定 `scripts/common.sh` 设置 Python、MuJoCo/EGL 和资产路径。为保持与正式测评一致，工作进程调用固定 LeRobot 的 `rollout`、processor 和 LIBERO 环境工厂；交互控制发生在 rollout 的逐步渲染回调处。

## 阅读顺序

1. `src/vla_inference_engine/live/events.py`：任务选择字段、READY/RUNNING/PAUSED 状态门。
2. `catalog.py` 和 `server.py`：40 个任务、模型标签、HTTP 与 WebSocket 接口。
3. `session.py`：单活动会话、generation、命令去重、工作进程创建和清理。
4. `worker.py`：固定初态、官方 `rollout`、动作观察包装、双图发布和 trace。
5. `static/app.js`：浏览器命令、断线重连、同 generation 的画面和动作更新。
6. 对照固定 LeRobot 的 `lerobot_eval.py::rollout`、`envs/libero.py` 及 `processor/env_processor.py`，追踪观测到仿真反馈的一步。

CLI 入口是 `src/vla_inference_engine/cli.py::live`；设计决策和阶段验收见 [实现规划](../design/LIBERO_LIVE_VIEWER_PLAN.zh-CN.md) 与 [项目主文档](../PROJECT_GUIDE.zh-CN.md)。
