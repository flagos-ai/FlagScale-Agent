<!-- Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 -->

# 自动调优的有界单次运行

`flagscale_agent.skills.train-run.scripts.training_trial` 执行 `flagscale train [model] -c <yaml> --test`，限制单次时长并返回测量摘要。
仅适用单机 Megatron；多机或其他后端使用 [主流程的直接启动](../SKILL.md)。沿用调用方已确认的环境和设备分配。

## 1. 准备本次 YAML

复制原配方，保留相对路径依赖和相同训练起点，只改本次试验变量及独立输出、checkpoint 保存路径。配置须满足：

- 完整 YAML，`experiment.task.type: train`、`experiment.task.backend: megatron`；`defaults` 省略或为 `[_self_]`。
- `experiment.runner`：`type: ssh`、`backend: torchrun`、`nnodes: 1`、`hostfile: null`；runner 和 `train.system` 均不启用 `no_shared_fs`。
- `experiment.exp_dir`：不含插值的绝对路径，目录尚不存在，包括未被 dryrun 创建。
- `train.system.logging.log_interval: 1`，其余日志路径使用默认值。

## 2. 准备请求 JSON

将以下示例按本次配方和测量约定填写，保存为 `/workspace/configs/baseline-01.json`：

```json
{
  "config_path": "/workspace/configs/baseline.yaml",
  "cwd": "/workspace/FlagScale",
  "timeout_seconds": 300,
  "expected_ranks": 16,
  "run_id": "baseline-01",
  "role": "baseline",
  "first_iteration": 1,
  "end_iteration": 40,
  "warmup_steps": 10,
  "global_batch_size": 64,
  "sequence_length": 512
}
```

`config_path`、`cwd` 用绝对路径。`expected_ranks` 与 `experiment.runner.nproc_per_node` 一致；
`end_iteration`、`global_batch_size`、`sequence_length` 分别与 `train.model` 的 `train_iters`、`global_batch_size`、`seq_length` 一致。
`first_iteration` 是本次首个预期迭代，`warmup_steps` 是从该迭代起跳过的预热步数，`timeout_seconds` 不超过剩余预算。
候选设置 `role: "candidate"` 和独立 `run_id`。CLI 要求 MODEL 时，额外添加 `model` 字段，值为该配方的模型名称。

## 3. 执行并读取结果

用 `shell(background=True)` 执行内置 `train-run` 脚本：

```bash
PYTHONUNBUFFERED=1 python -m flagscale_agent.skills.train-run.scripts.training_trial --request /workspace/configs/baseline-01.json
```

保存 job id，用 `shell_jobs(action="wait", job_id=..., timeout=60)` 等待；仍在运行就继续等待，无需同时反复查询日志。
结束后读取返回摘要；详情在 `experiment.exp_dir/agent_run/run.json`，测量在同目录的 `measurement.json`。
将其中给出的 `log_path`、`exit_code_path` 返回调用方的比较流程；脚本用法见 [结果分析](result-analysis.md)。不猜测日志路径或为取结果重跑训练。

启动故障或退出状态不清时先处理本次问题；诊断从同目录 `launcher.log` 开始。确认自有 worker 已退出后，将 OOM、无收益或质量失败返回调优主流程，由其选择后续候选。
`status="measured"` 表示获得测量，不代表质量验收或全部 rank 成功退出；退出证据缺失时保留“未验证”。
