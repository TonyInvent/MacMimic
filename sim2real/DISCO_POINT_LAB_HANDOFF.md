# 实验室交接 · OpenSprite v5 迪斯科斜指

**任务**　在 Jetson 253 上接入 Disco Point，让现场操作员从命令行启动，在吊架上做一遍 32.8 秒的斜指舞，结束后回到站姿并确认 31 个驱动全部失能。舞蹈由策略逐帧播放，Xbox 只保留操作员停机功能。模型已经导出，接手从离线回放开始。

| 截至发布提交 `8e8de2d` | 已有结果 |
| --- | --- |
| Disco Point（本次） | Mac 上 MuJoCo 跑完 1,640 帧，保持直立。ONNX 对 MLX 的最大原始动作误差为 0.00000620。Jetson 原生回放、失能验证和吊架上电实测待做。[预览](../policies/opensprite-disco-point/disco_point_preview.mp4) · [仿真记录](../policies/opensprite-disco-point/disco_point_simulation.json) |
| Standing Groove（第一个舞蹈） | Jetson 253、OpenIMU CAN FD，已上电走完 600 帧，可复用其启动、站姿交接与停机流程。[原生回放](evidence/standing-groove-native-253.json) · [真机记录](evidence/standing-groove-powered-253.json) |

## 1. 在 Jetson 上核对发布包，运行离线回放

在 `TonyInvent/MacMimic` 包含本交接文档的检出目录运行。使用装有 NumPy 和 CPU ONNX Runtime 的 Python。舞蹈文件来自标签 `opensprite-disco-point-s49-v1`。[策略目录](../policies/opensprite-disco-point/) · [发布包](https://github.com/TonyInvent/MacMimic/releases/tag/opensprite-disco-point-s49-v1)

```bash
(
  cd policies/opensprite-disco-point &&
  sha256sum -c disco_point_SHA256SUMS &&
  python3 disco_point_replay.py
)
```

预期所有文件校验为 `OK`，回放输出 `PASS: 1640 frames`。这一步只读取记录的观测。保留 Jetson 实际打印的最大误差和推理 p99。

## 2. 把新动作接入原生运行时

以 [Disco Point contract](../policies/opensprite-disco-point/disco_point_contract.json) 为准。ONNX 已内嵌整段舞蹈的动作潜变量。每 20 ms 输入一次 `frame`（int64，形状 `[1]`）和 `history`（float32，形状 `[1,10,99]`）。历史由旧到新，每帧依次为 31 个相对默认姿态的关节角、31 个关节速度、3 个机身坐标系角速度、3 个投影重力分量和 31 个上一步原始动作。输出 `actions` 是按 contract 关节顺序排列的 31 个原始动作，目标角映射和增益使用发布的 contract 与控制配置。WAV 只供听节拍。

| 改动位置 | 实验室 Codex 要做的事 |
| --- | --- |
| [原生模型封装](../runtime/native/src/standing_groove.hpp)、[actor 主循环](../runtime/native/src/sprite_supermimic_replay.cpp) | 为 Disco Point 注册独立的模型绑定和帧数。沿用站姿检查，通过后用实测状态填满十帧历史，前一动作清零。依次执行帧 0 到 1,639 各一次，末帧后按已有两秒姿态交接回到零速站立。 |
| [时序常量](../runtime/native/src/sling_window.hpp)、[CAN 控制器](../runtime/native/src/sprite_can_shadow.cpp)、[commissioning](../runtime/tools/run_supermimic_commissioning.py) | 同步动作窗口、整场时限、故障注入时刻、回站过渡、帧数与事件验收。旧舞的 36.5 秒是含准备和收尾的整场时长；新舞的 32.8 秒只是动作段。给新模式单独的确认词和报告标识。 |
| [原生回放验证器](../runtime/tools/verify_standing_groove.py)、[旧舞启动器](../scripts/standing-groove-253.sh) | 从发布清单绑定新 ONNX、contract、控制配置和回放向量的 SHA-256。为新舞提供独立命令行入口，例如待实现的 `scripts/disco-point-253.sh`，默认做回放。沿用 commissioning 互斥锁、停机和驱动保护，按动作包及 IMU 后端隔离资格记录。保留旧舞 600 帧回归。 |

## 3. 分阶段验收并留下记录

1. **原生回放，不连接电机。** 用 `bash sim2real/build-jetson.sh` 从当前源码重建。用新入口在 Jetson 的受保护 actor 上跑两遍：第一遍读发布包的历史，第二遍用记录的状态和原生 actor 自己的上一步输出重建历史。两遍均须比较全部 1,640 × 31 个原始动作，均为有限值，沿用 `atol=rtol=2e-4`。保存两次误差。再跑 `bash scripts/standing-groove-253.sh --replay` 确认旧舞回归。
2. **驱动失能下的整机验证。** 采用现场实际 IMU 后端，先验证完整站立、舞蹈和回站的时序，再单独验证有意注入的超时触发现有保护。保存与当前模型、运行时构建、硬件配置和 IMU 匹配的新资格记录。核查正常流程中完整的 1,640 帧及末帧交接，避免到了旧舞第 600 帧就提前收尾。
3. **现场操作员确认后上吊架。** 沿用[已实测的 Jetson 操作流程](README.md#standing-groove-on-jetson-253)：确认吊架支撑、急停可及、手柄在线，由现场操作员在终端确认上电。保留 LB+RB 持续三秒请求停机。记录实际播放帧数、站姿交接、actor/CAN 超时次数、TX 审计及逐驱动失能反馈。发生保护停机时记录最后动作帧和触发原因。

**交回结果**　把新入口的实际命令、Git 提交、包指纹、IMU 后端和上述三阶段报告写回仓库。原生回放通过、失能验证通过、上电走完全程分别记录。真机完成的判据是舞蹈完整播放、按计划回站、时序及 TX 审计通过，并取得全部 31 个驱动的失能反馈。
