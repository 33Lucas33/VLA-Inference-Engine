# 后续配置设计草案

这里保存原 `configs/{benchmark,env,inference,policy,safety}/` 的 YAML 设计样例。
当前没有配置加载器读取这些文件；修改它们不会改变 CLI、Mock 或 LeRobot 的运行行为。
其中 `status` 是草案建立时的记录，当前进度以 `docs/PROJECT_GUIDE.zh-CN.md` 为准。

实际生效的入口：

- 评测命令：`src/vla_inference_engine/benchmark/libero.py`；固定协议：同目录 `protocol.py`。
- 模型配置与归一化：所选 checkpoint 的 `config.json`、processor JSON 和 safetensors。
- CLI 参数：`src/vla_inference_engine/cli.py`。
- 环境变量与资源路径：`scripts/common.sh`。
- 公开比较基线：`configs/evaluation/baselines.json`。

将来引入配置加载器时，再把确实支持、经过验证的字段迁回运行配置目录；不要通过复制草案建立第二份实际协议。
