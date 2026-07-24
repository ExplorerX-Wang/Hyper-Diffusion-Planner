### Package Version ：
- navsim v1.1
- nuplan 1.2.2

NOTE: 相关数据存放路径定义在env.sh中，使用前source一下

## 1. IL training cache preparation
图像路径、ego 状态、GT 未来轨迹

add “+” to agent.config.pretrain.checkpoint_path in `run_cache_training.sh`
```bash
"${LAUNCHER[@]}" "${HDP_NAVSIM_ROOT}/${PY_ENTRY}" \
    agent="${AGENT}" \
    experiment_name="cache_${AGENT}" \
    train_test_split="${SPLIT}" \
    cache_path="${CACHE_PATH}" \
    force_cache_computation=true \
    +agent.config.pretrain_config.checkpoint_path="${CHECKPOINT_PATH}" \
    +cache_data_list_path="${CACHE_DATA_LIST_PATH}" \
    "$@"
```

```bash
cd /mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim
source /mnt/workspace/miniconda3/bin/activate navsim
bash ./scripts/training/run_cache_training.sh dp_vla_agent navtrain
```

## 2. IL training

download florence-2 encoder pretrained model from
`
https://huggingface.co/microsoft/Florence-2-large
`
- NOTE: model.safetensors is about 1.5Gb

set `export DP_VLA_ENCODER_PATH="/mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim/Florence-2-large"` 

change total_epochos and learning rate and launch training

```bash
cd /mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim
source /mnt/workspace/miniconda3/bin/activate navsim
DP_VLA_NPROC=8 \
bash ./scripts/training/run_training.sh \
  train_test_split=navtrain \
  dataloader.params.batch_size=2
```


## 3. navtest and navtrain metric caching
observation、道路、centerline、route、交通参与者等

```
cd /mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim
source /mnt/workspace/miniconda3/bin/activate navsim
bash ./scripts/evaluation/run_metric_caching.sh 
bash ./scripts/evaluation/run_metric_caching.sh navtrain

```


## 4. RL training caching

Florence encoder 特征、ego 状态、GT 轨迹
这里使用了DP-VLA预训练180epoch
```
cd /mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim
source /mnt/workspace/miniconda3/bin/activate navsim

DP_VLA_NPROC=1 bash ./scripts/training/run_cache_training.sh \
  dp_vla_rl_agent navtrain \
  agent.config.pretrain_config.checkpoint_path=/mnt/workspace/users/ExplorerX/NAVSIM/Hyper-Diffusion-Planner/HDP-navsim/navsim-exp/dp_vla_agent/2026.07.20.22.03.13/hf_checkpoints/epoch_179
```

## 5. RL fine-tune



## 6. IL evaluation



## 7. RL evaluation


# IL、RL Training Cache 与 Metric Cache 说明

## 1. 三种 Cache 的总体区别

| Cache 类型 | 默认环境变量/目录 | 主要保存内容 | 主要作用 |
| --- | --- | --- | --- |
| IL training cache | `$HDP_NAVSIM_CACHE_PATH`，默认位于 `$NAVSIM_EXP_ROOT/training_cache` | 相机图片路径、历史 ego 状态、GT 未来轨迹 | 给监督式模仿学习提供模型输入和监督标签 |
| RL training cache | `$HDP_RL_CACHE_PATH`，默认位于 `$NAVSIM_EXP_ROOT/rl_training_cache` | Florence encoder 输出、历史 ego 状态、GT 未来轨迹 | RL 微调时跳过 Florence，只训练 diffusion decoder |
| PDM metric cache | `$NAVSIM_METRIC_CACHE_PATH`，NAVSIM v1.1 默认位于 `$NAVSIM_EXP_ROOT/metric_cache` | ego 初始状态、未来交通参与者、道路、route、centerline、可行驶区域和 PDM-Closed 轨迹 | 为 RL rollout 和最终评测动态计算 PDMScore |

一句话概括：

```text
IL/RL training cache：描述模型吃什么、训练什么。
PDM metric cache：描述生成的轨迹在什么道路和交通环境中被仿真、评分。
```

这三种 cache 通过同一个场景 `token` 对齐，但文件格式和用途完全不同，不能互相替代。

---

## 2. IL Training Cache

### 2.1 生成位置和文件结构

IL cache 默认写入：

```text
$HDP_NAVSIM_CACHE_PATH
```

单个场景的目录结构为：

```text
training_cache/
└── <log_name>/
    └── <token>/
        ├── dp_vla_feature.gz
        └── dp_vla_target.gz
```

此外会生成或更新一个 JSON 数据列表，例如：

```text
hdp_navsim/training/training_utils/navtrain.json
```

JSON 中保存的是 `<log_name>/<token>` 相对路径列表，它不是特征数据本身。训练时先从 JSON 取得样本路径，再到 `$HDP_NAVSIM_CACHE_PATH` 中读取对应的 `.gz` 文件。

### 2.2 `dp_vla_feature.gz` 缓存哪些数据

主要包含：

- `meta_images`
  - 当前场景三路相机图片的文件路径。
  - 缓存的是图片路径，不是解码后的像素数组，也不是 Florence 特征。
  - 训练 DataLoader 读取样本时才打开图片、resize、normalize。
- `meta_status`
  - 4 帧历史 ego 状态。
  - 每帧包含局部坐标下的 ego pose、驾驶命令、速度和加速度等信息。
  - 其中 ego pose 会整理为 diffusion decoder 使用的 `history/proprio`，其他状态还会用于生成语言 prompt。

### 2.3 `dp_vla_target.gz` 缓存哪些数据

主要包含：

- `ego_future_trajectory`
  - 数据集中的 ego GT 未来轨迹。
  - 默认未来 4 秒、间隔 0.5 秒，共 8 个轨迹点。
  - 每个点表示为：

```text
[x, y, cos(heading), sin(heading)]
```

### 2.4 IL cache 的作用

IL 训练数据流为：

```text
dp_vla_feature.gz
    ↓
读取三路图片并生成语言 prompt
    ↓
Florence-2 在线编码图片和文本
    ↓
DiT diffusion decoder 预测未来轨迹
    ↓
与 dp_vla_target.gz 中的 GT 轨迹计算 diffusion loss
```

因此，IL 训练期间 Florence-2 仍然参与前向传播和训练。IL cache 的主要作用是避免每次训练启动时重新解析原始 NAVSIM scene，同时避免把巨大的图片像素直接写入 cache。

IL cache 与具体的 Florence checkpoint 通常无关，因为它没有保存 Florence hidden states。只有原始数据、相机选择、历史帧数、未来轨迹采样或预处理逻辑发生变化时，才需要重新生成。

---

## 3. RL Training Cache

### 3.1 生成位置和文件结构

RL cache 默认写入：

```text
$HDP_RL_CACHE_PATH
```

单个场景的目录结构为：

```text
rl_training_cache/
└── <log_name>/
    └── <token>/
        ├── dp_vla_rl_feature.gz
        └── dp_vla_rl_target.gz
```

### 3.2 `dp_vla_rl_feature.gz` 缓存哪些数据

主要包含：

- `encoder_output`
  - 三路相机和语言 prompt 经过 Florence-2 后得到的 encoder hidden states。
  - 这是 RL cache 与 IL cache 最主要的区别。
- `meta_status`
  - 历史 ego 状态。
  - RL 训练时从中提取 ego pose，形成 diffusion decoder 的 `history/proprio` 条件。

RL feature cache 的生成过程是：

```text
相机图片 + ego 状态
    ↓
生成语言 prompt
    ↓
Florence-2 encoder
    ↓
encoder_output 写入 dp_vla_rl_feature.gz
```

### 3.3 `dp_vla_rl_target.gz` 缓存哪些数据

主要包含：

- `ego_future_trajectory`
  - 与 IL target 相同格式的 GT 未来轨迹。
  - 默认形状为 `8 × 4`。

需要注意：当前 RL 实现虽然会读取并把 GT 放入 replay buffer，但在真正的 reward-weighted diffusion 更新中，GT 被忽略。当前 RL loss 主要使用模型生成的 rollout trajectory 作为训练目标，再使用 PDM reward 加权。

### 3.4 RL cache 的作用

RL 训练需要为一个场景生成多条候选轨迹并计算 reward。如果每次 rollout 都重新运行 Florence-2，会占用大量显存和时间。因此仓库提前缓存 Florence encoder 输出，RL 训练时只加载 diffusion decoder：

```text
dp_vla_rl_feature.gz 中的 encoder_output
    ↓
diffusion decoder 生成一组候选轨迹
    ↓
根据 token 查找对应的 metric cache
    ↓
PDM 仿真并计算 reward
    ↓
候选轨迹和 reward 写入 replay buffer
    ↓
reward-weighted diffusion loss 更新 decoder
```

RL cache 依赖生成它时使用的 Florence encoder 和监督训练 checkpoint。更换以下任意一项后，建议重新生成 RL cache：

- Florence-2 模型或 Florence 权重；
- IL/监督训练后的 DP-VLA checkpoint；
- 图片或语言预处理方式；
- encoder hidden size、tokenizer 或模型结构。

`agent.config.pretrain_config.checkpoint_path` 应当指向完整的监督训练 DP-VLA checkpoint，例如 Lightning `last.ckpt` 或 DP-VLA Hugging Face 导出目录，而不是只指向原始 `Florence-2-large` 目录。Florence 的单独路径由 `DP_VLA_ENCODER_PATH` 指定。

---

## 4. PDM Metric Cache

### 4.1 生成位置和文件结构

`navtrain` 和 `navtest` 的 metric cache 都可以保存在同一个根目录：

```text
$NAVSIM_METRIC_CACHE_PATH
```

NAVSIM v1.1 的单场景目录结构通常为：

```text
metric_cache/
└── <log_name>/
    └── <scenario_type>/
        └── <token>/
            └── metric_cache.pkl
```

虽然 `navtrain` 和 `navtest` 的文件格式相同，但场景内容不同：

- `navtrain`
  - 数据来自 `navsim_logs/trainval` 中由 navtrain scene filter 选中的场景。
  - 用于 RL 训练期间计算 reward。
- `navtest`
  - 数据来自 `navsim_logs/test` 中由 navtest scene filter 选中的场景。
  - 用于 IL/RL 模型的最终 PDM 仿真评测。

两者的 `log_name` 和 `token` 通常不同，因此可以共存在同一个 `metric_cache` 根目录，不会互相覆盖。

### 4.2 `metric_cache.pkl` 缓存哪些数据

主要包含：

- `ego_state`
  - 当前场景起点的 ego 状态和车辆参数。
- `trajectory`
  - PDM-Closed planner 为该场景计算的参考轨迹。
- `observation`
  - 未来时间范围内其他交通参与者的位置、朝向和占用区域。
  - 用于碰撞和 TTC 检查。
- `centerline`
  - 当前路线对应的道路中心线。
  - 用于计算沿路线的前进距离。
- `route_lane_ids`
  - 路线包含的 lane/lane connector ID。
  - 用于判断行驶方向和路线合规性。
- `drivable_area_map`
  - 道路、车道、路口和可行驶区域的多边形地图。
  - 用于判断车辆是否驶出可行驶区域或进入逆向车道。

Metric cache 不包含：

- Florence encoder 特征；
- IL/RL 模型权重；
- 模型预测轨迹；
- 已经固定算好的 PDMScore。

### 4.3 为什么 metric cache 中没有固定 PDMScore

PDMScore 依赖模型当前生成的轨迹。模型不断更新，同一个场景每次生成的候选轨迹也不同，因此不能在 metric caching 阶段预先写死一个分数。

metric caching 阶段只保存不随模型变化的场景环境：

```text
道路 + 路线 + 交通参与者未来状态 + ego 初始状态
```

RL rollout 或模型评测时才执行：

```text
当前模型生成的轨迹
    +
metric_cache.pkl 中的场景环境
    ↓
PDMSimulator
    ↓
PDMScorer
    ↓
本次轨迹对应的 PDMScore
```

### 4.4 PDM 内部如何使用 metric cache

模型默认输出未来 4 秒、0.5 秒间隔的 8 个轨迹点。PDM 会：

1. 将模型的 ego 相对坐标轨迹转换成世界坐标。
2. 将轨迹插值为 0.1 秒间隔。
3. 使用 LQR tracker 跟踪参考轨迹。
4. 使用 kinematic bicycle model 每 0.1 秒推进一次 ego 状态。
5. 将模拟得到的 ego 多边形与 metric cache 中对应时刻的交通参与者和地图进行比较。
6. 计算碰撞、可行驶区域、前进距离、TTC、舒适性和行驶方向等指标。
7. 聚合得到该候选轨迹的最终 PDMScore。

因此，PDM 中的“滚动”是车辆运动学状态沿 4 秒时间轴逐步推进，不是每 0.1 秒重新读取相机并再次调用模型规划。其他交通参与者按照 metric cache 中记录的未来状态运动，不会对 ego 的新行为做出实时反应。

---

## 5. 三种 Cache 在完整训练流程中的关系

### 5.1 IL 阶段

```text
NAVSIM 原始 navtrain 数据
    ↓
IL training cache
    ↓
Florence-2 + diffusion decoder
    ↓
使用 GT future trajectory 训练
    ↓
得到 IL checkpoint
```

IL 模型在 navtest 上评测时：

```text
navtest 原始传感器数据
    ↓
IL 模型生成轨迹
    ↓
navtest metric cache
    ↓
计算 PDMScore
```

### 5.2 RL 阶段

RL 训练需要同时满足：

```text
1. IL/监督训练 checkpoint
2. navtrain 的 RL training cache
3. navtrain 的 PDM metric cache
```

完整数据流为：

```text
RL training cache：读取 encoder_output 和 ego history
    ↓
当前 decoder 为每个场景生成多条未来轨迹
    ↓
使用相同 token 读取 navtrain metric cache
    ↓
PDM 对每条轨迹进行 4 秒仿真并计算 PDMScore
    ↓
轨迹和 reward 写入 replay buffer
    ↓
组内标准化 reward
    ↓
exp(reward) × diffusion MSE
    ↓
更新 diffusion decoder
```

RL 模型最终评测时使用：

```text
IL checkpoint
    +
RL fine-tune checkpoint
    +
navtest metric cache
    ↓
navtest PDMScore
```

---

## 6. 什么时候需要重新生成 Cache

| 变化内容 | IL cache | RL cache | Metric cache |
| --- | --- | --- | --- |
| 更换 IL/RL decoder checkpoint | 不需要 | 如果影响生成 RL cache 时加载的完整监督模型，建议重建 | 不需要 |
| 更换 Florence encoder 权重 | 通常不需要 | 需要 | 不需要 |
| 修改图片、语言或历史状态预处理 | 需要 | 需要 | 不需要 |
| 修改未来轨迹长度或采样间隔 | 需要 | 需要 | 需要保持 PDM 配置一致 |
| 修改 NAVSIM 原始数据或 scene filter | 需要 | 需要 | 需要 |
| 修改地图、route 或 PDM scoring 配置 | 不需要 | 不需要 | 需要 |
| 只修改训练 batch size、学习率或 epoch | 不需要 | 不需要 | 不需要 |

最终可以将三种 cache 理解为：

```text
IL cache     = 原始视觉/状态输入 + GT 轨迹标签
RL cache     = 预计算的 Florence 表征 + GT 轨迹标签
Metric cache = 与模型无关的道路和交通仿真环境
```


# 各阶段加载和更新哪些模型

## 1. 总览

| 阶段 | 加载/使用的模型 | 是否反向传播 | 实际更新的参数 | 阶段输出 |
| --- | --- | --- | --- | --- |
| IL training cache | 不需要运行 Florence 或 diffusion decoder | 否 | 不更新任何模型 | 图片路径、ego 状态、GT 轨迹 |
| IL training | Florence-2 encoder + diffusion decoder | 是 | Florence encoder 和 diffusion decoder | 完整 IL checkpoint |
| navtrain/navtest metric caching | PDM-Closed planner，以及地图和未来 observation 处理 | 否 | 不更新任何模型 | `metric_cache.pkl` |
| RL training cache | 完整 IL checkpoint 中的 Florence encoder | 否 | 不更新任何模型 | Florence `encoder_output`、ego 状态、GT 轨迹 |
| RL rollout epoch | IL 初始化后的 diffusion decoder + PDM simulator/scorer | 否 | 不更新任何模型 | 候选轨迹、PDM reward、replay buffer |
| RL optimization epoch | diffusion decoder | 是 | 只更新 diffusion decoder | RL fine-tune checkpoint |
| IL evaluation | IL checkpoint 中的 Florence encoder + diffusion decoder | 否 | 不更新任何模型 | navtest 轨迹和 PDMScore |
| RL evaluation | IL checkpoint 中的 Florence encoder + RL checkpoint 中的 decoder | 否 | 不更新任何模型 | navtest 轨迹和 PDMScore |

---

## 2. IL Training Cache 阶段

入口：

```bash
./scripts/training/run_cache_training.sh dp_vla_agent navtrain
```

这个阶段只进行 NAVSIM 原始数据解析和序列化：

```text
相机文件路径
历史 ego 状态
GT 未来轨迹
    ↓
写入 dp_vla_feature.gz 和 dp_vla_target.gz
```

该阶段：

- 不运行 Florence-2 前向传播；
- 不运行 diffusion decoder；
- 不构造 optimizer；
- 不进行反向传播；
- 不更新任何神经网络参数。

因此，IL training cache 不是模型 checkpoint，只是训练数据的预处理结果。

---

## 3. IL Training 阶段

入口：

```bash
./scripts/training/run_training.sh
```

IL 训练构建完整的 DP-VLA：

```text
Florence-2 vision/language encoder
    +
CustomDiT diffusion decoder
```

### 3.1 Florence-2 是否更新

会更新。

当前代码把 Florence encoder 的全部可训练参数加入 AdamW optimizer：

```text
Florence encoder learning rate = lightning_agent.params.lr / 10
Diffusion decoder learning rate = lightning_agent.params.lr
```

例如：

```bash
lightning_agent.params.lr=1e-4
```

实际学习率为：

```text
Florence encoder = 1e-5
Diffusion decoder = 1e-4
```

如果不在命令行覆盖学习率，当前默认配置中的基础学习率为 `1e-3`，对应：

```text
Florence encoder = 1e-4
Diffusion decoder = 1e-3
```

建议显式设置训练学习率，避免误用默认值。

### 3.2 Florence 中哪些部分参与训练

DP-VLA 使用 Florence 的视觉编码和语言 encoder。Florence 原始的 language decoder 和 LM head 在模型初始化时会被删除，因为轨迹规划不需要生成文本。

因此：

- Florence vision tower：参与前向传播并更新；
- Florence language encoder：参与前向传播并更新；
- Florence language decoder：已删除，不参与；
- Florence LM head：已删除，不参与；
- tokenizer/image processor：不是可训练参数，不更新；
- CustomDiT diffusion decoder：参与前向传播并更新。

### 3.3 IL checkpoint 包含什么

IL 训练保存的完整 checkpoint 包含：

```text
微调后的 Florence encoder
    +
训练后的 diffusion decoder
```

Lightning checkpoint 通常位于：

```text
<IL output_dir>/checkpoints/last.ckpt
```

Hugging Face 格式导出通常位于：

```text
<IL output_dir>/hf_checkpoints/last/
```

这个完整 IL checkpoint 才是 RL cache 和 RL fine-tune 的 `pretrain_config.checkpoint_path`。

---

## 4. Metric Caching 阶段

入口：

```bash
./scripts/evaluation/run_metric_caching.sh navtrain
./scripts/evaluation/run_metric_caching.sh navtest
```

该阶段会运行 PDM-Closed planner，计算并保存之后进行仿真评分所需的场景信息。这里主要使用：

- PDM-Closed planner；
- 地图、route 和 centerline 处理；
- 未来交通参与者 observation 处理。

LQR tracker、kinematic bicycle model 和 PDM scorer 不在 metric caching 阶段执行。它们是在后续 RL rollout 或 IL/RL evaluation 阶段读取 `metric_cache.pkl` 后才运行。

这些组件不是在此阶段训练的神经网络。该阶段：

- 不加载 IL/RL checkpoint；
- 不更新 Florence；
- 不更新 diffusion decoder；
- 不训练 PDM scorer；
- 不进行反向传播。

输出是与模型无关的：

```text
metric_cache.pkl
```

同一个 metric cache 可以被多个不同的 IL/RL checkpoint 重复使用。

---

## 5. RL Training Cache 阶段

入口：

```bash
./scripts/training/run_cache_training.sh dp_vla_rl_agent navtrain ...
```

这个阶段需要两个不同路径：

```bash
# 原始 Florence-2 模型目录
export DP_VLA_ENCODER_PATH=/path/to/Florence-2-large

# 完整 IL/DP-VLA checkpoint
agent.config.pretrain_config.checkpoint_path=/path/to/IL/checkpoints/last.ckpt
```

不能把 `pretrain_config.checkpoint_path` 指向原始 `Florence-2-large` 目录。

模型加载过程为：

```text
DP_VLA_ENCODER_PATH
    ↓
构建 Florence-2 encoder
    ↓
加载完整 IL checkpoint
    ↓
用 IL 中微调后的 Florence 权重覆盖原始 Florence 权重
    ↓
运行 Florence 前向传播
    ↓
缓存 encoder_output
```

diffusion decoder 在加载 checkpoint 后会被删除，因为生成 RL feature cache 时只需要 Florence encoder。

该阶段使用：

```python
model.eval()
```

并且不会创建训练 optimizer。因此：

- Florence encoder：只做推理，不更新；
- diffusion decoder：不参与 cache 计算，不更新；
- IL checkpoint：只读取，不修改；
- 输出的 `encoder_output`：写入 RL training cache。

因为 RL cache 保存的是 Florence hidden states，所以它依赖 IL checkpoint 中的 Florence 权重。更换 IL checkpoint 后需要重新生成 RL cache。

---

## 6. RL Fine-Tune 阶段

入口：

```bash
./scripts/training/run_training_rl.sh
```

RL 训练启动时采用：

```text
with_encoder=False
```

也就是说，训练进程中不会构建 Florence encoder，而是直接读取 RL cache 中已经保存的 `encoder_output`。

模型初始化过程为：

```text
构建 diffusion decoder
    ↓
从完整 IL checkpoint 中加载 IL decoder 权重
    ↓
读取 RL training cache 中的 encoder_output
    ↓
开始 rollout 和 reward-weighted 训练
```

### 6.1 RL rollout epoch 更新什么

默认每 `10` 个 epoch 重新执行一次 rollout，例如：

```text
epoch 0、10、20、30...
```

rollout 阶段使用：

```python
@torch.no_grad()
```

流程为：

```text
当前 diffusion decoder
    ↓
每个场景生成一组候选轨迹
    ↓
PDM simulator/scorer 计算 reward
    ↓
轨迹和 reward 写入 replay buffer
```

rollout epoch：

- 不更新 Florence；
- 不更新 diffusion decoder；
- 不更新 PDM simulator/scorer；
- 不执行 optimizer step。

它只负责使用当前模型采样新轨迹、计算 reward 和刷新 replay buffer。

### 6.2 RL optimization epoch 更新什么

除 rollout epoch 外，其余 epoch 会从 replay buffer 随机采样轨迹：

```text
rollout trajectory + PDM reward
    ↓
组内 reward 标准化
    ↓
weight = exp(normalized_reward)
    ↓
reward-weighted diffusion loss
    ↓
反向传播
```

RL optimizer 中只有：

```text
model.decoder.parameters()
```

因此：

- Florence encoder：不存在于 RL 训练模型中，不更新；
- CustomDiT diffusion decoder：更新；
- PDM simulator/scorer：只负责产生 reward，不更新；
- replay buffer：保存数据，不包含可训练参数；
- tokenizer：RL 训练直接读取 encoder cache，不更新。

当前实现属于 diffusion decoder 的完整参数微调，不是只训练 Florence，也不是只更新某个 reward head。

### 6.3 RL checkpoint 包含什么

RL checkpoint 主要包含：

```text
RL fine-tune 后的 diffusion decoder
```

因为 RL 训练时 `with_encoder=False`，RL checkpoint 不应被单独当作完整在线推理模型。评测时还需要 IL checkpoint 提供 Florence encoder。

---

## 7. IL Evaluation 阶段

IL 评测时：

```text
加载完整 IL checkpoint
    ↓
Florence 在线编码 navtest 相机和语言输入
    ↓
IL diffusion decoder 生成未来轨迹
    ↓
navtest metric cache + PDM simulator/scorer
    ↓
PDMScore
```

整个评测过程使用 `eval/no_grad`：

- Florence encoder：不更新；
- diffusion decoder：不更新；
- PDM simulator/scorer：不更新；
- metric cache：只读取。

---

## 8. RL Evaluation 阶段

RL checkpoint 只提供微调后的 decoder，因此 RL 评测需要组合两个 checkpoint：

```text
IL checkpoint
    ├── 提供微调后的 Florence encoder
    └── 提供 IL decoder 初始权重

RL checkpoint
    └── 覆盖 IL decoder，得到 RL fine-tune 后的 decoder
```

完整加载顺序为：

```text
构建带 Florence encoder 的完整 DP-VLA
    ↓
加载 IL checkpoint
    ↓
加载 RL checkpoint，覆盖 decoder
    ↓
在线生成 navtest 轨迹
    ↓
使用 navtest metric cache 计算 PDMScore
```

RL 评测时：

- Florence encoder：只推理，不更新；
- RL diffusion decoder：只推理，不更新；
- PDM simulator/scorer：只计算分数，不更新。

---

## 9. 模型参数随阶段的传递关系

```text
原始 Florence-2
    ↓ IL training：Florence + decoder 同时更新
完整 IL checkpoint
    ├──→ RL cache：只读取 IL Florence，生成 encoder_output
    └──→ RL training：只读取 IL decoder，作为 RL decoder 初始权重
                                ↓
                         只更新 diffusion decoder
                                ↓
                         RL fine-tune checkpoint

RL evaluation：
IL checkpoint 的 Florence encoder
    +
RL checkpoint 的 diffusion decoder
    ↓
完整 RL 推理模型
```


```text

IL training：更新 Florence encoder + diffusion decoder

RL optimization：只更新 diffusion decoder

```
