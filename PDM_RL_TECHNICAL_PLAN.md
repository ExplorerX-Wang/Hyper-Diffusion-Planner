# HDP-nuPlan 使用 PDMScore 进行强化学习微调的技术方案

> 适用范围：`HDP-nuplan/` 中已经完成模仿学习训练的二阶段端到端规划模型。  
> 目标：在相同的 nuPlan 训练场景上，通过 PDM 仿真为模型生成的轨迹计算 reward，并只微调第二阶段轨迹生成器。  
> 当前状态：本文是实现方案，文中标记为“拟新增”的脚本和模块尚未创建。

## 1. 目标与非目标

### 1.1 目标

1. 使用与 HDP-nuPlan 模仿学习完全相同的 nuPlan scenario 生成 PDM metric cache。
2. 使用相同的 `scenario.token` 对齐：
   - HDP 训练 `.npz`；
   - PDM `metric_cache.pkl`；
   - RL replay buffer。
3. 第一阶段场景编码器保持冻结，第二阶段 diffusion trajectory decoder 使用 PDMScore 微调。
4. 保留 GT behavior-cloning loss，降低 reward hacking 和训练崩溃风险。
5. 支持单机多卡 DDP，并使用 Ray/CPU 并行执行 PDM scoring。
6. 最终仍使用 nuPlan 原生 non-reactive/reactive closed-loop simulation 验证效果。

### 1.2 非目标

1. 不直接复用 NAVSIM `navtest` 的 metric cache。
2. 不把任意 nuPlan 场景轨迹与不匹配的 NAVSIM token 组合评分。
3. 第一版不实现逐帧重新规划的在线 RL。
4. 第一版不对 PDM simulator/scorer 进行学习。
5. 第一版不追求和 NAVSIM leaderboard 完全可比的官方 navtest PDMS。

本方案中的 PDMScore 是 nuPlan 训练场景上的离线轨迹 reward。只有在 NAVSIM 官方 navtest 场景和官方配置上计算的分数，才能作为 NAVSIM benchmark PDMS 与其他 NAVSIM agent 直接比较。

---

## 2. 当前 HDP-nuPlan 模型和数据接口

### 2.1 模型结构

当前模型入口：

```text
hdp_nuplan/model/hyper_diffusion_planner.py
```

模型由两部分组成：

```text
第一阶段：Hyper_Diffusion_Planner_Encoder
    - 动态交通参与者编码
    - 静态物体编码
    - lane/map 编码
    - token fusion

第二阶段：Hyper_Diffusion_Planner_Decoder
    - route encoding
    - diffusion DiT
    - DPM-Solver 轨迹采样
```

当前推理输出位于：

```text
hdp_nuplan/model/module/decoder.py
```

默认输出：

```text
80 × 4
[x, y, cos(heading), sin(heading)]
```

对应时域：

```text
8 秒 / 80 点 = 0.1 秒每点
```

输出中的 `x, y` 是当前 ego rear axle 局部坐标系中的未来位置。

### 2.2 模仿学习数据

当前预处理入口：

```text
hdp_nuplan/data_process/data_processor.py
```

每个 `.npz` 已经保存：

```text
map_name
token
ego_current_state
ego_agent_future
neighbor_agents_past
neighbor_agents_future
static_objects
lanes
route_lanes
speed_limit
traffic_light
```

文件名为：

```text
<map_name>_<scenario_token>.npz
```

当前 Dataset：

```text
hdp_nuplan/utils/dataset.py
```

虽然 `.npz` 中存在 `token`，但 Dataset 返回值中没有 token。RL 阶段必须修改 Dataset，使每个 batch 同时返回 `scenario_token`。

---

## 3. 总体方案

```text
同一个 nuPlan AbstractScenario
    ├──→ HDP data process
    │       └── <map>_<token>.npz
    │
    └──→ PDM metric cache process
            └── <log>/<scenario_type>/<token>/metric_cache.pkl

训练 batch
    ↓
第一阶段 encoder（冻结）
    ↓
第二阶段 decoder 为每个场景生成 K 条轨迹
    ↓
根据 token 加载 metric_cache.pkl
    ↓
PDMSimulator + PDMScorer
    ↓
每条轨迹得到 PDMScore 和子指标
    ↓
写入 replay buffer
    ↓
组内 reward 标准化
    ↓
reward-weighted diffusion loss + GT BC loss
    ↓
只更新第二阶段 decoder
```

核心原则：

```text
metric cache 与模型无关，但与场景、token、时间戳和 ego 初始状态强绑定。
```

---

## 4. PDM Metric Cache 设计

### 4.1 为什么不能直接运行 NAVSIM 的缓存脚本

NAVSIM v1.1 的：

```text
navsim/planning/script/run_metric_caching.py
```

外层数据入口固定使用 NAVSIM `SceneLoader`，因此不能直接读取 HDP-nuPlan 使用的全部 nuPlan scenario。

但是内部：

```python
MetricCacheProcessor.compute_metric_cache(scenario)
```

接收的是 nuPlan 通用接口：

```python
AbstractScenario
```

因此不需要重写 PDM cache 的核心算法，只需要新增一个 nuPlan 场景入口。

### 4.2 拟新增缓存脚本

建议新增：

```text
hdp_nuplan/rl/cache_pdm_metrics.py
```

职责：

1. 使用与 `data_process.py` 相同的 `NuPlanScenarioBuilder`。
2. 使用相同的 `ScenarioFilter`、log list 和 scenario token list。
3. 对每个 `AbstractScenario` 调用：

```python
MetricCacheProcessor.compute_metric_cache(scenario)
```

4. 写入 cache metadata。
5. 统计成功、失败和缺失 token。

建议目录：

```text
<PDM_METRIC_CACHE_ROOT>/
└── <log_name>/
    └── <scenario_type>/
        └── <scenario_token>/
            └── metric_cache.pkl
```

### 4.3 Cache 内容

每个 `metric_cache.pkl` 保存：

- `ego_state`
  - 当前场景的 ego 初始状态；
- `trajectory`
  - PDM-Closed planner 生成的基准轨迹；
- `observation`
  - 未来交通参与者的插值占用区域；
- `centerline`
  - route 对应的道路中心线；
- `route_lane_ids`
  - 当前路线包含的 lane/lane connector；
- `drivable_area_map`
  - 道路、车道、路口和可行驶区域多边形。

Metric cache 不保存：

- 模型输入 embedding；
- 第一阶段 encoder 输出；
- 模型 checkpoint；
- 模型生成轨迹；
- 固定不变的 PDMScore。

PDMScore 必须在模型生成轨迹后动态计算。

### 4.4 Token 对齐

生成 metric cache 前，应从训练数据列表构造 token 白名单。

推荐流程：

```text
nuplan_train.json
    ↓
读取每个 .npz 中的 token
    ↓
形成唯一 token 集合
    ↓
ScenarioFilter(scenario_tokens=...)
    ↓
只生成训练实际使用场景的 metric cache
```

不要只依靠随机 ScenarioFilter 再跑一次，因为第二次随机选择的场景可能与 `.npz` 训练集不同。

生成后必须检查：

```text
training token set - metric cache token set = empty
```

任何缺失 token 都应在 RL 启动前报告，而不是在训练中静默跳过。

---

## 5. 时域与轨迹格式

### 5.1 当前时域差异

HDP-nuPlan 模型：

```text
8 秒，80 点，0.1 秒间隔
```

NAVSIM v1.1 默认 PDM proposal：

```text
4 秒，40 步，0.1 秒间隔
```

第一版推荐只使用模型轨迹的前 4 秒计算 PDM reward：

```python
prediction_4s = prediction[..., :40, :]
```

然后转换 heading：

```python
heading = torch.atan2(
    prediction_4s[..., 3],
    prediction_4s[..., 2],
)

poses = torch.cat(
    [prediction_4s[..., :2], heading[..., None]],
    dim=-1,
)
```

PDM 输入格式：

```text
40 × 3
[x, y, heading]
```

并显式指定：

```python
TrajectorySampling(
    num_poses=40,
    interval_length=0.1,
)
```

因为 HDP-nuPlan 已经输出 0.1 秒间隔的轨迹，所以不需要先下采样到 0.5 秒再由 PDM 插值。

### 5.2 为什么第一版不直接扩展到 8 秒

PDM 的 TTC 会在当前仿真时刻继续查看一段未来 observation。默认 4 秒 proposal 通常需要约 5 秒的未来交通参与者数据。

如果扩展到 8 秒，需要同步修改：

- `proposal_sampling`；
- PDM-Closed 基准轨迹长度；
- metric cache future observation 长度；
- TTC 未来查看窗口；
- 原始 scenario 的未来有效长度检查。

8 秒 PDM 通常需要接近 9 秒的未来 observation。部分 scenario 可能不满足，需要过滤或截断。

### 5.3 后 4 秒如何训练

如果 PDM 只评分前 4 秒，模型后 4 秒不能完全失去监督。建议总 loss 保留完整 8 秒 GT behavior-cloning：

```text
L_total =
    λ_rl × L_reward_weighted_first_4s
    + λ_bc × L_GT_full_8s
```

推荐初始值：

```text
λ_rl = 1.0
λ_bc = 0.1 ~ 1.0
```

具体比例应通过 validation PDMScore 和 nuPlan closed-loop 指标共同选择。

---

## 6. 二阶段模型的冻结与缓存策略

### 6.1 第一版更新范围

建议：

```text
第一阶段 encoder：冻结
第二阶段 diffusion decoder：更新
```

实现：

```python
for parameter in model.encoder.parameters():
    parameter.requires_grad = False

model.encoder.eval()

optimizer = torch.optim.AdamW(
    model.decoder.parameters(),
    lr=rl_learning_rate,
)
```

冻结第一阶段的原因：

- PDM reward 直接作用于最终轨迹；
- 降低训练显存；
- 避免破坏已经学到的场景表示；
- 可以预计算 encoder context；
- replay buffer 中不需要保存完整计算图。

### 6.2 可选的第一阶段 Feature Cache

如果 encoder 前向仍然占用较多时间，可以新增：

```text
<RL_FEATURE_CACHE_ROOT>/<token>.pt
```

内容：

```text
encoding
route_lanes
ego_current_state
```

RL rollout 和 optimization 直接加载这些固定条件，只运行 decoder。

第一版也可以先不做 feature cache，冻结 encoder 后在线前向；验证流程正确后再优化吞吐。

### 6.3 何时允许更新第一阶段

只有在第二阶段微调稳定后，才考虑：

- 解冻 encoder 最后若干层；
- encoder 使用 decoder 学习率的 `0.01 ~ 0.1`；
- 保留较大的 BC regularization；
- 监控 closed-loop collision 和 off-road 回退。

不建议第一版同时更新全模型。

---

## 7. Rollout 与 PDM Reward

### 7.1 每个场景生成 K 条轨迹

默认建议：

```text
group_size K = 8 或 10
```

第一阶段 context 只计算一次，然后扩展成 K 份：

```python
encoding_k = encoding.repeat_interleave(K, dim=0)
route_k = route_lanes.repeat_interleave(K, dim=0)
ego_k = ego_current_state.repeat_interleave(K, dim=0)
```

diffusion decoder 内部为每份 context 采样不同的初始噪声，从而得到 K 条不同轨迹。

输出整理为：

```text
[batch_size, K, 80, 4]
```

### 7.2 PDM 内部仿真

对每条轨迹：

1. 截取前 4 秒；
2. 从 `[x,y,cos,sin]` 转换为 `[x,y,heading]`；
3. 读取相同 token 的 `metric_cache.pkl`；
4. 将轨迹转换到世界坐标；
5. LQR tracker 跟踪参考轨迹；
6. kinematic bicycle model 每 0.1 秒推进 ego；
7. 与 cache 中的动态交通参与者和地图比较；
8. 计算 PDM 子指标和最终 PDMScore。

主要子指标：

- no-at-fault collision；
- drivable-area compliance；
- progress；
- TTC；
- comfort；
- driving-direction compliance。

其他交通参与者按照 metric cache 中记录的未来状态运动，不会响应 ego。因此这是 non-reactive、有限时域的离线仿真 reward。

### 7.3 PDM 评分并行化

PDM scoring 主要消耗 CPU，包含 Shapely、多边形碰撞和地图查询。建议：

- GPU rank 负责模型 rollout；
- Ray CPU workers 负责每条轨迹的 PDM scoring；
- 每个 unique token 的 metric cache 在一个 batch 内只解压一次；
- 通过 Ray object store 共享 simulator、scorer 和 cache；
- reward 返回时恢复原始轨迹顺序。

可参考：

```text
../HDP-navsim/hdp_navsim/agent/dp_vla/scoring.py
```

建议新增：

```text
hdp_nuplan/rl/pdm_reward.py
```

主要接口：

```python
def parallel_pdm_scores(
    trajectories_xyh,
    tokens,
    metric_cache_refs,
    simulator_ref,
    scorer_ref,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ...
```

---

## 8. Replay Buffer

### 8.1 建议保存的数据

每个 replay item：

```text
scenario_token
rollout_trajectories: [K, 80, 4]
rewards: [K]
PDM sub-metrics
可选：encoder context 或对应 feature-cache path
```

GT 不需要复制到每个 replay item，可以根据 token 从原始 `.npz` 重新读取。

### 8.2 Buffer 生命周期

建议每隔固定 epoch 使用当前模型重新生成 rollout：

```text
epoch 0、10、20、30...
```

在新的 rollout epoch：

1. 清空旧 buffer；
2. 使用当前 decoder 重新采样；
3. 重新计算 PDMScore；
4. 写入新 buffer。

其余 epoch 从 buffer 随机采样并更新 decoder。

这样可以避免长期使用已经过时的旧策略轨迹。

### 8.3 DDP 行为

第一版建议每个 DDP rank 维护自己的本地 replay buffer：

- DistributedSampler 给每个 rank 分配不同场景；
- 每个 rank 为自己的场景生成 rollout；
- reward loss 正常通过 DDP 同步梯度；
- 不在 rank 之间同步完整 replay buffer，降低通信开销。

需要保证所有 rank 在 rollout/optimization 阶段保持相同节奏，避免部分 rank 返回 loss、部分 rank 不返回 loss 造成 DDP hang。

---

## 9. Reward 与 RL Loss

### 9.1 PDMScore 不可微

PDMScore 由车辆运动学、几何碰撞和规则评分产生，不能直接对模型执行：

```python
pdm_score.backward()
```

第一版使用 reward-weighted diffusion regression，不通过 PDM 反向传播。

### 9.2 组内 Advantage

一个场景的 K 条轨迹 reward：

```text
R1, R2, ..., RK
```

标准化：

```text
Ai = (Ri - mean(R)) / (std(R) + epsilon)
```

权重：

```text
wi = exp(beta × Ai)
```

建议：

```text
beta = 0.5 ~ 1.0
weight clip = [0.1, 10.0]
```

clip 可以避免某个异常 reward 产生过大的梯度。

### 9.3 Reward-Weighted Diffusion Loss

当前 `hdp_nuplan/loss.py` 在内部直接执行 batch mean。为了按每条 rollout 加权，需要重构为先返回：

```text
per_sample_diffusion_loss: [B × K]
per_sample_hybrid_loss: [B × K]
```

RL loss：

```text
L_diff_rl = mean(wi × per_sample_diffusion_loss_i)
L_hybrid_rl = mean(wi × per_sample_hybrid_loss_i)

L_RL =
    L_diff_rl
    + planning_hybrid_loss × L_hybrid_rl
```

这里的监督轨迹是模型 rollout trajectory，不是 GT trajectory。高 reward rollout 获得更大训练权重。

### 9.4 GT Behavior-Cloning Regularization

同一个 batch 继续使用原始 GT 计算：

```text
L_BC =
    diffusion_loss(GT)
    + planning_hybrid_loss × waypoint_loss(GT)
```

总 loss：

```text
L_total =
    λ_rl × L_RL
    + λ_bc × L_BC
```

第一版建议不要去掉 `L_BC`。原因：

- PDMScore 可能稀疏；
- collision/DAC 乘法指标可能让大量轨迹 reward 为 0；
- 只优化 PDM reward 容易产生不自然轨迹；
- 后 4 秒暂时没有 PDM reward；
- 保留 GT 可以防止模型远离数据分布。

### 9.5 Reward Shaping

第一版优先直接使用最终 PDMScore：

```text
reward = pdm_result.score
```

如果同组 K 条轨迹分数几乎完全相同，组内标准差接近 0，训练没有有效排序信号。此时可以增加稠密子指标：

```text
reward =
    pdm_score
    + α × progress
    + β × ttc
    + γ × comfort
```

使用 shaped reward 时必须分别记录原始 PDMScore，不能只记录组合 reward，否则可能掩盖 reward hacking。

---

## 10. 训练阶段调度

推荐两阶段 epoch 调度：

### 10.1 Rollout Epoch

```text
model.encoder.eval()
model.decoder.eval()
torch.no_grad()
```

执行：

```text
训练场景
    ↓
K 条 trajectory rollout
    ↓
PDM scoring
    ↓
写 replay buffer
```

不执行：

- backward；
- optimizer step；
- EMA update；
- scheduler step。

### 10.2 Optimization Epoch

```text
model.encoder.eval()
model.decoder.train()
```

执行：

```text
replay buffer sample
    ↓
reward-weighted diffusion loss
    +
GT BC loss
    ↓
backward
    ↓
gradient clip
    ↓
optimizer step
    ↓
EMA update
```

需要保证冻结 encoder 后不会因为 `model.train()` 被重新切换到训练模式。建议分别调用：

```python
model.encoder.eval()
model.decoder.train()
```

---

## 11. 建议的代码结构

建议新增：

```text
HDP-nuplan/
├── cache_pdm_metrics.py
├── train_predictor_rl.py
└── hdp_nuplan/
    └── rl/
        ├── __init__.py
        ├── metric_cache_loader.py
        ├── pdm_reward.py
        ├── replay_buffer.py
        ├── rollout.py
        └── rl_loss.py
```

建议修改：

```text
hdp_nuplan/utils/dataset.py
    - 返回 scenario token
    - 可选返回原始 GT 和 cache path

hdp_nuplan/loss.py
    - 支持 reduction="none"
    - 返回 per-sample diffusion/hybrid loss

hdp_nuplan/model/module/decoder.py
    - 支持 group_size/K 次并行采样
    - 明确输出 [B,K,T,4]

train_predictor.py
    - 保持原 IL 训练入口不变

train_predictor_rl.py
    - 新增独立 RL 入口，避免破坏现有 IL 训练
```

不建议直接在现有 `train_epoch.py` 中混入大量 rollout/PDM 逻辑。独立 RL 入口更容易调试、复现和回退。

---

## 12. 配置参数

建议 RL 配置至少包含：

```text
pretrained_checkpoint
train_set
train_set_list
pdm_metric_cache_path
rl_feature_cache_path

group_size
rollout_update_epoch
rollout_steps
replay_buffer_size

rl_learning_rate
lambda_rl
lambda_bc
reward_temperature_beta
reward_weight_min
reward_weight_max

pdm_horizon
pdm_interval
ray_reserved_cpus

freeze_encoder
use_ema
```

第一版建议：

```text
group_size = 8
rollout_update_epoch = 10
pdm_horizon = 4.0
pdm_interval = 0.1
freeze_encoder = true
lambda_rl = 1.0
lambda_bc = 0.5
reward_temperature_beta = 1.0
reward_weight_min = 0.1
reward_weight_max = 10.0
```

真实 batch size 需要结合：

```text
每卡显存
group_size
decoder hidden size
PDM CPU throughput
```

共同确定。应先使用 `batch_size=1、group_size=2` 做完整 smoke test，再逐步增大。

---

## 13. Checkpoint 加载与保存

### 13.1 RL 初始化

加载 IL checkpoint：

```text
encoder weights
decoder weights
EMA weights（如果 IL 推理默认使用 EMA）
normalizer/config
```

必须确保 RL 训练选择的权重版本与 IL evaluation 一致。例如 IL evaluation 使用 `ema_state_dict`，RL 初始化也应使用同一份 EMA 权重，不能一个使用 raw model、另一个使用 EMA。

### 13.2 RL 保存

RL checkpoint 建议保存：

```text
decoder state_dict
decoder EMA state_dict
optimizer
scheduler
epoch
global step
RL config
IL checkpoint path/hash
normalization config
metric-cache config
```

如果只保存 decoder，最终 evaluation 需要：

```text
IL checkpoint 的 encoder
    +
RL checkpoint 的 decoder
```

也可以额外保存完整合并模型，便于部署。

---

## 14. 日志和监控

至少记录：

### Reward

```text
reward/mean
reward/std
reward/group_max
reward/group_min
reward/best_minus_worst
reward/invalid_ratio
```

### PDM 子指标

```text
metric/pdms
metric/no_collision
metric/drivable_area
metric/progress
metric/ttc
metric/comfort
metric/driving_direction
```

### Loss

```text
loss/total
loss/rl_diffusion
loss/rl_hybrid
loss/bc_diffusion
loss/bc_hybrid
```

### 系统

```text
system/replay_buffer_size
system/pdm_scenarios_per_second
system/pdm_failure_count
system/cache_hit_rate
system/gpu_memory
system/ray_pending_tasks
```

如果 `reward/std` 长期接近 0，应检查：

- group 内轨迹是否真的不同；
- decoder 是否重复使用同一份噪声；
- PDM cache 是否匹配正确 token；
- PDMScore 是否全部被 collision/DAC 门控为 0；
- 是否需要 reward shaping。

---

## 15. 失败处理

以下情况不应直接参与训练：

- token 找不到 metric cache；
- PDM scoring 抛异常；
- reward 为 NaN/Inf；
- 轨迹包含 NaN/Inf；
- 轨迹点数或时间间隔不匹配；
- 所有 K 条轨迹完全相同；
- metric cache 的 ego initial state 与训练样本不一致。

建议：

1. Rollout 失败返回无效标记，不使用伪造 reward。
2. 一个场景部分轨迹失败时，只在有效轨迹足够时保留该组。
3. 有效轨迹少于 2 条时无法计算可靠组内标准差，应跳过。
4. 缺失 cache 的 token 在训练启动前一次性列出。
5. 每次训练保存失败 token 文件，便于离线排查。

---

## 16. 验证与验收计划

### 16.1 单元测试

### Token 测试

- `.npz` token 能在 metric cache loader 中唯一找到；
- 训练 token coverage 为 100%；
- 不允许不同 log 的重复 token 指向错误 cache。

### 轨迹转换测试

- 输入 `[80,4]`；
- 输出前 4 秒 `[40,3]`；
- `heading = atan2(sin,cos)`；
- 坐标系为 ego rear axle local frame；
- 第 40 点对应 4.0 秒。

### Freeze 测试

一次 RL backward 后：

```text
encoder grad 全部为 None
decoder 至少一个参数 grad 非零
```

### Reward 测试

人工构造：

- 明显碰撞轨迹；
- 停在原地轨迹；
- 沿 route 正常前进轨迹；
- 驶出道路轨迹。

验证 PDMScore 排序符合预期。

### 16.2 单场景集成测试

建议参数：

```text
1 GPU
1 scenario
group_size=2
batch_size=1
1 rollout epoch
1 optimization step
```

通过条件：

- 两条轨迹不同；
- metric cache 成功加载；
- PDM reward 有限；
- replay buffer 写入成功；
- decoder 完成一次 optimizer step；
- encoder 权重不变。

### 16.3 小数据 Smoke Test

建议：

```text
32 ~ 128 scenarios
group_size=4
2 GPUs
10 ~ 20 epochs
```

检查：

- reward 是否上升；
- BC loss 是否保持稳定；
- collision/DAC 是否恶化；
- PDM scoring 是否成为吞吐瓶颈；
- DDP 是否出现 rank 不同步。

### 16.4 全量训练

全量训练前必须固定：

- IL checkpoint；
- scenario token list；
- metric cache 版本；
- nuPlan/NAVSIM/PDM scorer 版本；
- map version；
- normalizer；
- RL config。

否则不同实验之间的 PDMScore 不可严格比较。

---

## 17. 最终评测

PDMScore 是训练 reward 和快速离线评估指标，不应替代 nuPlan 官方闭环评测。

建议评测顺序：

```text
1. held-out nuPlan 场景 PDMScore
2. nuPlan closed_loop_nonreactive_agents
3. nuPlan closed_loop_reactive_agents
```

最终重点比较：

- collision；
- drivable area；
- progress；
- TTC；
- comfort；
- closed-loop planner failure；
- 原 IL checkpoint 与 RL checkpoint 的差异。

运行原生 nuPlan 仿真继续使用：

```text
sim_hdp_runner.sh
```

如果 RL 只保存 decoder，需要在 planner 初始化时先加载 IL encoder，再覆盖 RL decoder。

---

## 18. 实施顺序

建议按以下顺序开发：

### Phase 1：PDM Cache

1. 新增 nuPlan scenario metric-cache 入口；
2. 使用训练 token 白名单；
3. 完成 cache coverage 检查；
4. 单 token PDMScore 验证。

### Phase 2：数据和轨迹接口

1. Dataset 返回 token；
2. decoder 支持 K 条并行采样；
3. 完成 `[80,4] → [40,3]` 转换；
4. 完成 Ray PDM scoring。

### Phase 3：RL Loss

1. `loss.py` 支持 per-sample loss；
2. 实现 group advantage；
3. 实现 reward-weighted loss；
4. 加入完整 8 秒 BC regularization。

### Phase 4：训练框架

1. 新建 `train_predictor_rl.py`；
2. 冻结 encoder；
3. 接入 replay buffer；
4. 接入 DDP、EMA、checkpoint 和日志。

### Phase 5：验证与扩展

1. 小规模 smoke test；
2. 全量 4 秒 PDM reward；
3. nuPlan non-reactive/reactive validation；
4. 根据结果决定是否扩展到 8 秒 PDM。

---

## 19. 最小可行版本定义

MVP 应满足：

```text
同一批 nuPlan 训练 token
    +
4 秒 PDM metric cache
    +
冻结第一阶段 encoder
    +
每场景 4 条 diffusion rollout
    +
最终 PDMScore 组内标准化
    +
reward-weighted decoder loss
    +
完整 8 秒 GT BC regularization
```

MVP 不包含：

- 8 秒 PDM；
- reactive traffic；
- encoder 解冻；
- PPO/GRPO diffusion log-prob；
- 跨节点共享 replay buffer；
- 在线逐帧重新规划。

先完成 MVP 可以最低成本验证：PDMScore 作为 reward 是否能改善 HDP-nuPlan 第二阶段规划器。

---

## 20. 结论

HDP-nuPlan 使用 PDMScore 进行强化学习微调是可行的，推荐实现方式不是把 nuPlan checkpoint 直接接到 NAVSIM navtest，而是：

```text
在完全相同的 nuPlan 训练 scenario 上生成 PDM metric cache
    ↓
用 token 对齐训练样本和 metric cache
    ↓
冻结第一阶段场景编码器
    ↓
第二阶段为每个场景生成多条轨迹
    ↓
PDM 对前 4 秒轨迹计算 reward
    ↓
reward-weighted diffusion loss + 8 秒 GT BC loss
    ↓
只更新第二阶段 diffusion decoder
```

该方案能够复用当前 HDP-NAVSIM 的 PDM scoring、replay buffer 和 reward-weighted diffusion 思路，同时保留 HDP-nuPlan 原有的向量场景输入、8 秒轨迹输出和 nuPlan closed-loop evaluation。
