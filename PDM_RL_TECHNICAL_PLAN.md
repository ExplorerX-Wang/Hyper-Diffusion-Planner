# nuPlan IL 模型接入 NAVSIM PDMScore 强化学习，再回到 nuPlan 闭环仿真的技术方案

> 适用项目：`HDP-nuplan/` 与 `HDP-navsim/`
> 目标链路：**nuPlan 模仿学习 checkpoint → NAVSIM 场景与 PDM simulator/scorer 强化学习 → nuPlan 闭环仿真**
> 当前状态：这是实现设计文档；文中标记为“拟新增”的模块和命令尚未实现。

---

## 1. 目标的准确含义

本方案不是在任意 nuPlan 场景上重新实现一套 PDMScore，也不是把
`HDP-nuplan` checkpoint 填到 `dp_vla_rl_agent` 的
`pretrain_config.checkpoint_path` 中。

真正的数据流是：

```text
nuPlan 数据
    │
    ├─ 模仿学习
    ▼
HDP-nuplan IL checkpoint
    │
    │ 在相同驾驶时刻上构造 nuPlan 原生模型输入
    ▼
NAVSIM navtrain 场景 ── NAVSIM metric cache
    │                         │
    │ HDP 采样 K 条轨迹        │ observation / route / map / PDM reference
    └──────────┬──────────────┘
               ▼
       NAVSIM PDM simulator + PDMScorer
               │
               ▼
          PDMScore reward
               │
               ▼
     微调 HDP-nuplan trajectory decoder
               │
               ▼
      导出兼容 nuPlan planner 的 checkpoint
               │
               ▼
       nuPlan 原生闭环仿真与最终评估
```

这里有两个必须同时满足的约束：

1. **环境属于 NAVSIM**：训练场景、split、metric cache、PDM simulator 和
   PDMScorer 都来自 NAVSIM。
2. **模型接口保持 nuPlan 原生格式**：HDP 的输入预处理、网络结构、normalizer、
   输出轨迹定义和最终 checkpoint 格式不能被 Florence/DP-VLA 接口替换。

---

## 2. 结论：可行，但需要“跨框架桥接层”

这条链路在工程上可行，而且比“把 PDMScore 搬到任意 nuPlan 场景”更接近你的
目标。不过，不能把现有两个工程直接用一个参数连起来，原因如下：

- `HDP-nuplan` 是矢量场景模型，输入包括高频 ego/agent 历史、地图、route、
  traffic light 等结构化数据。
- `HDP-navsim` 当前 RL agent 是 DP-VLA/Florence 路线，输入、网络和 checkpoint
  都与 `HDP-nuplan` 不同。
- NAVSIM `AgentInput` 主要包含 ego status、camera 和 lidar，不能直接替代
  `HDP-nuplan` 的矢量输入。
- NAVSIM v1.1 场景原始帧通常是 0.5 秒间隔，而 `HDP-nuplan` 默认使用
  2 秒、20 个历史 pose，即 10 Hz 历史。直接把 NAVSIM 的低频历史塞入模型会
  产生明显输入分布偏移。
- NAVSIM 的 PDM reward 不可微，不能直接对 `PDMScore` 调
  `backward()`；必须采用轨迹采样、reward 加权的 diffusion loss 或其他
  policy-gradient 类目标。

因此，推荐新增一个独立的 `nuplan_hdp_rl` 桥接模块，复用 NAVSIM 的评分能力，
但不继承 DP-VLA 的 Florence 模型结构。

---

## 3. 三个阶段各自负责什么

| 阶段 | 场景/数据 | 模型输入 | 评分或 loss | 更新参数 |
|---|---|---|---|---|
| nuPlan IL | nuPlan train | nuPlan 原生矢量特征 | diffusion/trajectory imitation loss | encoder + decoder |
| NAVSIM RL | NAVSIM `navtrain` | 与 NAVSIM token 对齐的 nuPlan 原生矢量特征 | NAVSIM PDMScore reward + BC/anchor loss | 第一版只更新 decoder |
| nuPlan 闭环 | nuPlan simulation scenarios | nuPlan planner 在线构造的原生矢量特征 | nuPlan 闭环 metrics | 不更新，仅评估 |

第一版强烈建议冻结 encoder，只微调：

```text
Hyper_Diffusion_Planner.decoder
```

这样既能降低显存和训练不稳定性，也能最大程度保持从 NAVSIM RL 返回 nuPlan
闭环时的输入兼容性。

---

## 4. 当前代码中可直接复用的部分

### 4.1 HDP-nuplan 模型

模型入口：

```text
HDP-nuplan/hdp_nuplan/model/hyper_diffusion_planner.py
```

网络由：

```text
Hyper_Diffusion_Planner.encoder
Hyper_Diffusion_Planner.decoder
```

组成。

当前 decoder 推理和 diffusion 训练逻辑位于：

```text
HDP-nuplan/hdp_nuplan/model/module/decoder.py
```

默认预测 80 个点、间隔 0.1 秒，总时域 8 秒。单条预测的语义为：

```text
[x, y, cos(heading), sin(heading)]
```

nuPlan planner 在下面的位置把它转为：

```text
[x, y, heading]
```

并生成 `InterpolatedTrajectory`：

```text
HDP-nuplan/hdp_nuplan/planner/planner.py
```

### 4.2 HDP-nuplan 输入处理

离线数据处理和在线 observation adapter：

```text
HDP-nuplan/hdp_nuplan/data_process/data_processor.py
```

模仿学习数据集：

```text
HDP-nuplan/hdp_nuplan/utils/dataset.py
```

离线 `.npz` 中已经包含 token、ego/agent 历史、地图、route、speed limit 和
traffic light 等输入，但当前 Dataset 没有把 token 返回给训练循环。RL 数据集
必须显式返回 NAVSIM token。

### 4.3 NAVSIM PDM reward

现有项目已经调用：

```text
NAVSIM MetricCacheLoader
NAVSIM pdm_score
NAVSIM PDMSimulator
NAVSIM PDMScorer
```

相关调用位置：

```text
HDP-navsim/hdp_navsim/agent/dp_vla/scoring.py
HDP-navsim/hdp_navsim/agent/dp_vla/utils.py
HDP-navsim/hdp_navsim/run_pdm_score_ddp.py
```

现有 DP-VLA RL 中的 replay buffer、group reward normalization 和 Ray 并行
PDM scoring 可参考：

```text
HDP-navsim/hdp_navsim/agent/dp_vla/dp_vla_rl_agent.py
```

可以复用“算法结构”和评分工具，但不能复用它的 Florence feature、模型构造或
checkpoint 字段。

---

## 5. 最关键的数据对齐

### 5.1 推荐方法：用 NAVSIM token 找回同一时刻的 nuPlan 原生输入

NAVSIM 来源于 nuPlan/OpenScene 数据。对 NAVSIM `navtrain` 中的每个场景，
在原始 nuPlan 数据库中定位同一 log、同一 timestamp/initial lidar frame，
然后继续使用 `HDP-nuplan` 原来的 `DataProcessor` 生成模型输入。

最终形成两个由同一个 NAVSIM token 索引的 cache：

```text
<navsim_token>
    ├─ nuplan_feature_cache/<token>.npz
    └─ navsim_metric_cache/.../<token>/metric_cache.pkl
```

训练 batch 中：

```text
nuplan feature cache ──> HDP-nuplan model
navsim metric cache  ──> PDM simulator/scorer
```

这种做法的优点是：

- RL 时模型看到的输入定义与 nuPlan IL 和最终 nuPlan 闭环一致；
- 保留 10 Hz 历史，不需要伪造高频历史；
- 不修改 HDP encoder 的张量语义；
- RL 后的权重可以直接回写原 nuPlan 模型。

### 5.2 不要假设两个 token 字符串天然等价

NAVSIM 的 initial frame token、nuPlan scenario token 和 lidar-pc token 可能属于
不同层级。构建 cache 时不能只靠文件名字符串碰撞，应生成显式 manifest：

```text
navsim_token
navsim_log_name
navsim_timestamp
nuplan_log_name
nuplan_lidar_token
nuplan_scenario_token
nuplan_feature_path
metric_cache_path
map_name
route_id_hash
initial_rear_axle_x
initial_rear_axle_y
initial_rear_axle_heading
valid
invalid_reason
```

推荐匹配顺序：

1. log name；
2. initial lidar token；若 token 体系不同，则用 timestamp；
3. map name；
4. 初始 rear-axle pose；
5. route/roadblock 信息。

初始位姿建议检查：

```text
position error < 0.05 m
heading error  < 0.01 rad
timestamp error <= 一个数据库采样间隔
```

阈值可以根据实际数据库精度调整，但训练前必须统计误差分布，不能静默接受
错配。

### 5.3 缺少原始 nuPlan 数据库时的备选方案

可以从 NAVSIM `Scene` 中读取 annotations、map、route 和交通灯，再把 0.5 秒
帧插值为 HDP 所需的 10 Hz 历史。动态 agent 需要按 `track_token` 关联，位置和
速度线性插值，heading 需要 unwrap 后插值。

这个方案只能作为 fallback，因为：

- 插值得到的 10 Hz 历史不包含真实中间运动；
- 遮挡和目标出现/消失难以正确恢复；
- 构造结果可能与 nuPlan 闭环时的 DataProcessor 分布不同。

如果采用 fallback，必须先用相同 checkpoint 对一批对齐场景比较：

```text
原生 nuPlan feature 输入产生的轨迹
NAVSIM 插值 feature 输入产生的轨迹
```

若轨迹偏差明显，应先做 feature-domain adaptation，再做 PDM RL。

### 5.4 不推荐的方法

以下方法不应作为正式方案：

- 只把 `HDP-nuplan` checkpoint 路径传给 `dp_vla_rl_agent`；
- 把 Florence checkpoint 参数改成 HDP checkpoint；
- 只用 NAVSIM `AgentInput` 的 ego/camera/lidar，丢掉 HDP 的矢量地图和 agent
  历史；
- 把一个 nuPlan 场景的模型输入和另一个 NAVSIM token 的 metric cache 配对；
- 根据 token 文件名相似就认为场景已对齐。

---

## 6. 拟新增的桥接模块

推荐把训练侧桥接代码放在 `HDP-navsim`，因为 RL 使用 NAVSIM split、metric
cache、simulator 和 scorer；模型定义仍直接从 `HDP-nuplan` 导入。

拟新增目录：

```text
HDP-navsim/hdp_navsim/agent/nuplan_hdp_rl/
├── __init__.py
├── agent.py
├── model_loader.py
├── trajectory_adapter.py
├── dataset.py
├── scoring.py
├── replay_buffer.py
└── checkpoint_export.py

HDP-navsim/hdp_navsim/scripts/
├── build_nuplan_navsim_manifest.py
├── cache_nuplan_features_for_navsim.py
├── train_nuplan_hdp_rl.py
└── validate_nuplan_navsim_alignment.py

HDP-navsim/hdp_navsim/config/agent/
└── nuplan_hdp_rl_agent.yaml
```

各模块职责如下。

### 6.1 `model_loader.py`

- 读取 `HDP-nuplan` 的 `args.json`/Config；
- 构造原始 `Hyper_Diffusion_Planner`；
- 加载 IL checkpoint；
- 处理 `model`、`ema_state_dict` 和 `module.` 前缀；
- 默认使用 IL EMA 权重作为 RL 初始化；
- 冻结 encoder，只向 optimizer 注册 decoder 参数；
- 保持原 observation normalizer 和 trajectory normalizer 不变。

### 6.2 `dataset.py`

每个样本返回：

```python
{
    "token": navsim_token,
    "model_inputs": nuplan_native_features,
    "gt_trajectory": nuplan_gt_8s,
    "metric_cache_path": navsim_metric_cache_path,
}
```

训练 cache 与 metric cache 的职责不能混淆：

- `nuplan_feature_cache`：给模型前向和 BC loss 使用；
- `metric_cache`：给 NAVSIM PDM simulator/scorer 使用；
- replay buffer：保存当前策略采样轨迹及其 reward。

### 6.3 `trajectory_adapter.py`

负责把 HDP 输出转成 NAVSIM `Trajectory`。

转换步骤：

```python
x = prediction[..., 0]
y = prediction[..., 1]
heading = atan2(prediction[..., 3], prediction[..., 2])
poses = stack([x, y, heading], dim=-1)
```

HDP 默认 8 秒、0.1 秒间隔；NAVSIM PDM proposal 默认只评估约 4 秒。因此第一版
取：

```text
poses[:, :40, :]
sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
```

不要直接每 0.5 秒抽一个点，除非当前安装的 NAVSIM `pdm_score` 明确要求
4 秒 8 点输入。`pdm_score` 内部通常会根据 sampling 处理轨迹；实现时应通过
单元测试确认当前 v1.1 API。

坐标系必须保持：

```text
当前 ego rear axle 为原点
x 轴向前
y 轴向左
heading 为相对当前 ego 的角度
```

在开始训练前，用直行、左转、右转场景可视化检查；坐标系一旦错位，PDMScore
会变成无意义 reward。

### 6.4 `scoring.py`

可以封装或复用：

```text
HDP-navsim/hdp_navsim/agent/dp_vla/scoring.py
```

输入为：

```text
token
K 条 NAVSIM Trajectory
MetricCacheLoader
PDMSimulator
PDMScorer
```

输出至少保存：

```text
pdm_score
no_at_fault_collisions
drivable_area_compliance
driving_direction_compliance
ego_progress
time_to_collision_within_bound
comfort
```

训练 reward 可以先使用最终 `pdm_score`，但日志中必须保留各子指标，否则无法
判断模型究竟改善了 progress，还是通过保守停车避免碰撞。

### 6.5 `agent.py`

这是面向 NAVSIM 的模型包装器，但内部模型必须是
`HDP-nuplan/Hyper_Diffusion_Planner`。

职责：

- 接收 token 对齐的 nuPlan feature；
- 执行 encoder；
- 从 decoder 采样 K 条候选轨迹；
- 调用 trajectory adapter；
- 在训练时生成 replay 数据；
- 在 NAVSIM 评估时输出单条 `Trajectory`。

它不应实例化 Florence，也不应依赖：

```text
DP_VLA_ENCODER_PATH
microsoft/Florence-2-large
```

---

## 7. NAVSIM 中的 RL 到底是什么

第一版不是让车辆在 NAVSIM 中逐帧执行、再把新 observation 返回模型的在线
MDP/PPO。

更准确地说，它是：

```text
每个 NAVSIM 场景只在起始时刻规划一次
    └─ 模型采样 K 条未来轨迹
        └─ PDM 内部让 ego 跟踪每条候选轨迹
            └─ 与缓存的未来交通参与者轨迹、地图和 route 计算分数
```

TTC 能被计算，是因为 metric cache 中包含时间序列 observation，PDM simulator
也会在未来时间步上模拟 ego 状态，并不是因为模型每一步都重新规划。

所以第一版属于：

```text
offline one-shot trajectory optimization
或 contextual-bandit-style RL
```

它可以有效使用 PDMScore 微调轨迹分布，但不等价于完整的在线交互式强化学习。
如果后续需要真正的滚动 RL，必须另行实现：

```text
plan → execute Δt → update observation → replan
```

还需要 reactive traffic-agent 模型和 episode 状态管理，这超出 NAVSIM 当前
PDM metric-cache 评分范式。

---

## 8. 推荐的 RL 目标

### 8.1 Rollout

对每个 scene feature：

1. encoder 只计算一次；
2. decoder 使用不同随机噪声采样 `K` 条轨迹；
3. 前 4 秒送入 NAVSIM PDM scorer；
4. 保存轨迹、GT 和 reward 到 replay buffer。

第一版建议：

```text
K = 8 或 10
```

显存不足时可分块采样，例如每次 2 条，累计得到 K 条。

### 8.2 Reward 标准化

对同一场景的 K 个 reward 做 group-relative normalization：

```text
A_i = (R_i - mean(R_group)) / (std(R_group) + eps)
```

建议额外：

```text
clip(A_i, -3, 3)
```

这样模型学习的是同一驾驶场景下候选轨迹的相对优劣，能降低不同场景天然难度
差异带来的噪声。

### 8.3 训练 loss

PDMScore 不可微，推荐以采样轨迹为目标重新加噪，训练 decoder 提高高 reward
轨迹的概率：

```text
L_RL = mean(w(A_i) * L_diffusion(sample_i))
```

权重可以采用：

```text
w(A_i) = exp(clip(A_i / temperature, -c, c))
```

同时保留 nuPlan GT imitation loss：

```text
L_total =
    λ_rl * L_RL
  + λ_bc * L_BC
  + λ_anchor * L_anchor
```

其中：

- `L_BC`：原始 8 秒 GT diffusion/trajectory loss；
- `L_anchor`：可选，使 RL decoder 不要偏离 IL decoder 太快；
- PDM reward 只覆盖前约 4 秒，BC loss 继续约束后 4 秒。

第一版推荐：

```text
λ_rl = 1.0
λ_bc = 0.2 ~ 1.0
λ_anchor = 0 或一个很小的值
temperature = 1.0
```

实际权重应根据 reward、BC loss 和梯度范数的日志调整。

### 8.4 更新哪些权重

第一版：

```python
for parameter in model.encoder.parameters():
    parameter.requires_grad_(False)

for parameter in model.decoder.parameters():
    parameter.requires_grad_(True)
```

保持 encoder 为 `eval()`，避免 dropout 或其他随机行为改变缓存特征分布。decoder
训练时为 `train()`。

第二阶段稳定后才考虑解冻 encoder 顶部层；一旦解冻，必须确认 RL 使用的是原生
nuPlan 高频特征，否则返回 nuPlan 闭环时容易出现 domain shift。

---

## 9. PDMScore 在本方案中的作用与限制

PDM scorer 会综合：

- at-fault collision；
- drivable-area compliance；
- driving-direction compliance；
- progress；
- TTC；
- comfort。

它不是“当前轨迹和 GT 做点对点距离”。

PDM 内部过程大致是：

1. 把模型的局部相对轨迹转换到 metric cache 的全局坐标；
2. tracker 控制 ego 跟踪 proposal；
3. bicycle model 推进 ego 状态；
4. 与 cache 中的地图、route 和未来 observation 检查碰撞、TTC、进度、舒适性；
5. 汇总为 PDMScore。

主要限制：

- 其他交通参与者通常来自缓存，不会针对 ego 新策略完全交互式响应；
- 只评估 proposal horizon 内的行为；
- reward 可以被极端保守策略利用，因此必须监控 progress，并保留 BC 约束；
- NAVSIM navtrain 上的训练分数不能当作 navtest 的无偏最终结果。

---

## 10. Checkpoint 如何带回 nuPlan

### 10.1 保持架构和 Config 不变

RL 不应改变：

```text
future_len
hidden_dim
encoder/decoder 层数
normalizer
输入字段和 shape
```

否则 RL checkpoint 无法直接被 nuPlan planner 加载。

### 10.2 推荐保存内容

RL checkpoint 建议保存：

```python
{
    "base_il_checkpoint": "...",
    "model": full_model_state_dict,
    "ema_state_dict": full_ema_state_dict,
    "rl_decoder": decoder_state_dict,
    "rl_decoder_ema": decoder_ema_state_dict,
    "optimizer": optimizer_state_dict,
    "epoch": epoch,
    "global_step": global_step,
    "config": resolved_config,
    "bridge_manifest_hash": "...",
    "navsim_split": "navtrain",
}
```

### 10.3 EMA 是一个容易踩坑的位置

nuPlan planner 当前默认：

```text
enable_ema=True
```

并从 checkpoint 的：

```text
ema_state_dict
```

加载权重。因此，如果只把 RL decoder 写进 `model`，但没有更新
`ema_state_dict`，闭环仿真实际上仍会运行旧 IL 模型。

正确做法是：

1. 从 IL checkpoint 读取完整 model/EMA；
2. 用 RL decoder 覆盖完整 model 中的 decoder keys；
3. 用 RL decoder EMA 覆盖完整 EMA 中的 decoder keys；
4. 保留 encoder 的 IL 权重；
5. 保持当前 `planner.py` 所期望的 `module.` key 前缀；
6. 严格加载并检查 missing/unexpected keys 均为空。

如果 RL 阶段没有维护 EMA，则最终评估时应：

- 要么用 RL decoder 当前权重同时覆盖 model 和 EMA；
- 要么显式设置 `enable_ema=False`。

第一种更方便与现有 `sim_hdp_runner.sh` 保持一致。

---

## 11. 推荐的完整执行顺序

### 阶段 A：准备 nuPlan IL checkpoint

确认至少具备：

```text
IL checkpoint
对应 args.json / model config
训练时使用的 normalizer
原始 nuPlan DB 与 maps
```

先用当前 `HDP-nuplan` planner 跑小规模 nuPlan 闭环，记录 IL baseline。

### 阶段 B：准备 NAVSIM metric cache

对 RL train split 生成或检查 NAVSIM metric cache：

```bash
cd /mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim
source /mnt/workspace/miniconda3/bin/activate navsim

bash ./scripts/evaluation/run_metric_caching.sh navtrain
```

metric cache 只服务于 PDM simulation/scoring，不是模型输入 cache。

建议从 `navtrain` 中划出固定的 RL-validation token，避免所有训练 token 都参与
optimizer 更新。

### 阶段 C：建立 NAVSIM ↔ nuPlan manifest

拟新增命令：

```bash
python -m hdp_navsim.scripts.build_nuplan_navsim_manifest \
  split=navtrain \
  navsim_data_root="${NAVSIM_EXP_ROOT}" \
  nuplan_data_root="${NUPLAN_DATA_ROOT}" \
  output_path="${CACHE_ROOT}/nuplan_navsim_manifest.parquet"
```

先运行只读验证：

```bash
python -m hdp_navsim.scripts.validate_nuplan_navsim_alignment \
  manifest_path="${CACHE_ROOT}/nuplan_navsim_manifest.parquet"
```

只有 `valid=true` 的 token 才可进入 RL。

### 阶段 D：缓存与 NAVSIM token 对齐的 nuPlan 原生输入

拟新增命令：

```bash
python -m hdp_navsim.scripts.cache_nuplan_features_for_navsim \
  manifest_path="${CACHE_ROOT}/nuplan_navsim_manifest.parquet" \
  hdp_config="${HDP_NUPLAN_ROOT}/checkpoints/args.json" \
  output_dir="${CACHE_ROOT}/nuplan_hdp_features/navtrain"
```

cache 结束后应检查：

```text
manifest valid token 数
feature cache token 数
metric cache token 数
三者交集 token 数
缺失/重复/错配 token 数
```

### 阶段 E：先做零训练一致性测试

在不更新参数的情况下：

1. 加载 IL checkpoint；
2. 从 bridge dataset 取 16～100 个场景；
3. 生成轨迹；
4. 转成 NAVSIM `Trajectory`；
5. 计算 PDMScore；
6. 可视化局部轨迹和全局轨迹；
7. 确认所有子指标不是全 0、NaN 或异常常数。

这一步通过后才能开始 RL。

### 阶段 F：NAVSIM PDM RL

拟新增命令：

```bash
cd /mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim
source /mnt/workspace/miniconda3/bin/activate navsim

export PYTHONPATH="${PWD}/../HDP-nuplan:${PYTHONPATH}"

torchrun --standalone --nproc_per_node=8 \
  -m hdp_navsim.scripts.train_nuplan_hdp_rl \
  agent=nuplan_hdp_rl_agent \
  train_test_split=navtrain \
  agent.config.il_checkpoint="${HDP_NUPLAN_CKPT}" \
  agent.config.il_args="${HDP_NUPLAN_ARGS}" \
  agent.config.bridge_manifest="${CACHE_ROOT}/nuplan_navsim_manifest.parquet" \
  agent.config.feature_cache="${CACHE_ROOT}/nuplan_hdp_features/navtrain" \
  agent.config.metric_cache="${NAVSIM_EXP_ROOT}/metric_cache" \
  agent.config.freeze_encoder=true \
  agent.config.num_rollouts=8 \
  dataloader.params.batch_size=1
```

这是目标接口示例；在相应脚本实现前不能直接执行。

DDP 每张卡 batch size 建议从 1 开始，PDM scoring 通过 CPU/Ray 并行。不能只设置：

```bash
DP_VLA_NPROC=8
```

然后启动一个没有读取该变量的 Python 进程；最终日志必须出现：

```text
MEMBER: 1/8
...
MEMBER: 8/8
```

### 阶段 G：导出 nuPlan-compatible checkpoint

拟新增命令：

```bash
python -m hdp_navsim.agent.nuplan_hdp_rl.checkpoint_export \
  --base-il-checkpoint "${HDP_NUPLAN_CKPT}" \
  --rl-checkpoint "${RL_CKPT}" \
  --output "${OUTPUT_DIR}/hdp_nuplan_pdm_rl.pth"
```

导出脚本必须立即用 `HyperDiffusionPlanner.initialize()` 的同一路径做一次严格
加载测试。

### 阶段 H：回到 nuPlan 做闭环

把 `HDP-nuplan/sim_hdp_runner.sh` 中：

```text
CKPT_FILE
```

指向：

```text
hdp_nuplan_pdm_rl.pth
```

并保持与 IL baseline 完全相同的：

```text
scenario filter
random seed
planner sampling
reactive/non-reactive 配置
metric 配置
worker 数量
```

最终至少比较：

```text
IL baseline
PDM-RL checkpoint
```

不要只比较 NAVSIM train reward。

---

## 12. 环境兼容性

NAVSIM RL 进程需要同时 import：

```text
navsim
nuplan
hdp_navsim
hdp_nuplan
```

建议只使用一个 RL conda 环境，并把两个工程以 editable package 或明确
`PYTHONPATH` 接入。需要检查：

```bash
python -c "import navsim, nuplan, hdp_navsim, hdp_nuplan; print('imports ok')"
python -c "import torch; print(torch.__version__, torch.cuda.device_count())"
```

需要特别避免：

- 同一环境出现两套不同版本的 `nuplan`；
- NAVSIM 与 HDP-nuplan 分别依赖不兼容的 torch/numpy；
- 训练配置加载了另一套同名 Hydra config；
- Ray worker 没有继承 `PYTHONPATH`、数据根目录或 cache 路径。

Ray worker 应显式传递所需 `runtime_env`，不要依赖登录 shell 中偶然存在的环境
变量。

---

## 13. 实施前必须通过的检查

### 13.1 数据检查

- NAVSIM token 能唯一定位对应的 nuPlan 原始帧；
- feature cache 与 metric cache 的 token 完全对齐；
- 初始时间、rear-axle pose、map 和 route 一致；
- 输入张量 shape、dtype 与原 IL Dataset 一致；
- 没有把未来信息混入 encoder 输入。

### 13.2 轨迹检查

- `atan2(sin, cos)` 顺序正确；
- x/y 单位为米；
- heading 单位为弧度；
- 局部坐标原点和朝向一致；
- 4 秒裁剪没有 off-by-one；
- NAVSIM sampling 与数组长度一致。

### 13.3 参数检查

- encoder `requires_grad=False`；
- optimizer 中只有 decoder 参数；
- frozen encoder 没有梯度；
- IL、RL 和导出 checkpoint 均可严格加载；
- nuPlan planner 实际加载的是 RL EMA，而不是旧 IL EMA。

### 13.4 Reward 检查

- reward 与每条 rollout 一一对应；
- Ray 返回结果后没有打乱 token/rollout 顺序；
- PDM 子指标不是全 0、NaN 或常数；
- reward normalization 只在同一场景的 K 条轨迹内进行；
- progress、collision、TTC、comfort 都被单独记录。

---

## 14. 推荐的渐进式里程碑

### M1：跨框架前向

目标：

```text
一个 NAVSIM token
→ 找到 nuPlan 原生 feature
→ HDP IL 生成轨迹
→ NAVSIM PDMScore 成功返回
```

### M2：批量对齐

目标：

```text
至少 100 个 token
→ feature/metric cache 100% 对齐
→ PDMScore 无 NaN
→ 轨迹可视化正确
```

### M3：单卡过拟合测试

用很小的 token 集合验证：

```text
decoder 确实更新
reward-weighted loss 下降
高 reward 轨迹概率上升
checkpoint 可恢复
```

### M4：8 卡正式训练

验证：

```text
DDP rank 为 8
每个 token 不被错误重复
Ray scoring 没有 CPU 内存泄漏
replay buffer 不无限增长
```

### M5：nuPlan 闭环回归

先跑少量场景确认：

```text
planner 可加载
输出无 NaN
轨迹时域正确
闭环能完成
```

再运行完整 benchmark。

---

## 15. 最终推荐

建议采用下面这条主路线：

```text
HDP-nuplan IL checkpoint
    +
NAVSIM navtrain token
    +
同一 token 对应的 nuPlan 原生 10 Hz 矢量 feature
    +
NAVSIM metric cache / PDMSimulator / PDMScorer
    ↓
冻结 encoder，PDM reward + BC loss 微调 decoder
    ↓
合并 model 与 EMA decoder 权重
    ↓
HDP-nuplan 原生 planner 做闭环仿真
```

这条路线的核心不是“把 nuPlan 模型改成 NAVSIM 模型”，而是：

> **让 nuPlan 模型保持原生输入输出和 checkpoint 兼容性，只在训练中借用
> NAVSIM 的场景索引、PDM 内部仿真和 PDMScore 作为 reward。**

最大的前置条件是能把 NAVSIM 场景精确映射回同一时刻的原始 nuPlan 数据。
如果这一步无法保证，PDM reward 会对应错误场景，或者模型会在插值伪造的输入
分布上被微调，最终回到 nuPlan 闭环时效果没有可信保证。
