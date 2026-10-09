# 2026-06-28 实验笔记

## 今日目标

停止继续修补 `image -> CVAE topology -> diffusion` 的独立 two-step 路线，重新确认 topology 在 diffusion 中的真实作用，并规划一个可微、可联合训练的 soft topology guidance 方案。

## 方向反思

- 长期目标仍是从单张 real photo 生成合法且与图像一致的 B-Rep；最终判断标准是 valid STEP rate 与条件一致性，而不是 topology exact accuracy。
- [06-23 two-step 下游实验](../2026-06-23/NOTE.md)已经说明：独立 topology predictor 的输出直接接入 diffusion 会明显伤害下游，但此前将问题主要归因于 one-to-many 和 topology fidelity，并未先审计 topology bias 的真实数值语义。
- 今天确认当前 `scale=2.0` 不是“相邻 attention 乘 2”，而是在 attention logit 上加 `2`；高噪声阶段相当于将相邻 pair 的未归一化 attention mass 乘约 `exp(2)=7.39`。这使 false-positive topology edge 的影响远强于原先理解。
- 因此当前应跳出“继续提升 CVAE 局部指标”的循环，先验证 topology-to-diffusion 接口，再讨论 topology predictor、one-to-many 和 visibility-weighted supervision。
- 之前基于混合 checkpoint、后处理和 evaluator 协议得到的定量表格不再作为新路线的决策依据；后续实验必须使用固定 checkpoint、固定 legacy post-process 和明确区分 evaluation coverage 与 valid solid rate 的协议。

## 实验记录

### 1. 审计当前 diffusion topology mask 的真实语义

**为什么这么做**

- two-step 失败可能不仅来自 topology predictor，还可能来自 topology guidance 本身过强；必须先确认“不相连”和“相连”在 self-attention 中到底发生了什么。

**具体实施**

- 审计 [diffusion_denoiser.py](../../src/brepnet/models/diffusion_denoiser.py) 中 `OracleTopologySelfAttentionMask`、`LearnedTopologySelfAttentionMask` 和 `BRepDenoiser.run_backbone`。
- 当前 `soft_bias` 实际计算：

```text
bias_ij = adjacency_ij * (t / num_train_timesteps) * scale
attention_ij = softmax(base_logit_ij + bias_ij)
```

- 当前 oracle 配置 [train_diffusion_oracle_topology.yaml](../../configs/train_diffusion_oracle_topology.yaml) 使用 `scale=2.0`。

**结果**

- 对二值 GT adjacency：
  - 不相连 pair：additive bias 为 `0`，不会被 mask，仍可正常 attention。
  - 相连 pair：最高获得约 `+2` logit，相对同 base logit 的非邻接 pair，其未归一化 attention mass 约乘 `7.39`。
- 当前所谓 soft topology mask 更准确地说是一个强 positive adjacency bias，不表达“禁止连接”。
- `hard_mask` 才会把不相连 pair 置为禁止并令 softmax 后权重为 0。
- topology bias 会进入 denoiser 全部 self-attention 层；错误 topology 的影响会在多层、多个 diffusion timestep 中持续出现。

**参考 / 关联**

- 实现：[diffusion_denoiser.py](../../src/brepnet/models/diffusion_denoiser.py)
- 模型装配：[diffusion.py](../../src/brepnet/models/diffusion.py)
- 当前 oracle 配置：[train_diffusion_oracle_topology.yaml](../../configs/train_diffusion_oracle_topology.yaml)
- two-step 失败记录：[2026-06-23 NOTE](../2026-06-23/NOTE.md)

### 2. 可微 multiplicative topology prior 设计

**为什么这么做**

- 目标不是由独立 CVAE 先生成一个必须完全正确的离散 topology，而是让图像条件模型输出连续 pairwise affinity，并在 diffusion 内部作为弱结构引导联合优化。
- 该方案保留 one-to-many 所需的不确定性：无证据的 pair 不被禁止，只是不获得额外 attention。

**具体实施**

- topology head 计划输出：
  - `30` 个 face-existence logits；
  - `30*29/2=435` 个严格上三角 edge logits。
- edge logits 经 sigmoid 后得到 `[0,1]` 的 edge probability，再镜像到下三角，构造对称矩阵 `P`；diagonal 固定为 0。
- attention prior 使用：

```text
P = symmetric(sigmoid(edge_logits))
weight = 1 + P
bias = timestep_weight * log(weight)
attention = softmax(base_logits + bias)
```

- 数值语义：

```text
P=0.0 -> attention mass 乘 1.0
P=0.2 -> attention mass 乘 1.2
P=0.4 -> attention mass 乘 1.4
P=1.0 -> attention mass 乘 2.0
```

- 这里必须加全 1 矩阵，而不是 Identity；`log(Identity + P)` 会使非对角 `P=0` 变成 `-inf`，重新退化为 hard mask。
- 已在当前 PyTorch `2.4.1+cu124` 环境验证：float attention mask 的梯度可经过 `log(1+P)`、对称化和 sigmoid 回传至 edge logits。
- 需要明确：`[1,2]` 限制的是 pairwise 未归一化 attention mass 的倍率，不保证最终输出 feature 的每个分量最多变化两倍。

**结果**

- 方案在数学和当前 PyTorch 实现上可微，可支持 topology predictor 与 diffusion 统一训练。
- 仅靠 diffusion loss 可能让 `P` 退化为不具备 topology 语义的 routing matrix，因此第一版保留 exact GT adjacency BCE 与 face-valid BCE。
- 当前已完成设计、attention-mask 梯度 smoke verification，以及实验 0 的 topology-bias 实现；image-conditioned topology head 留到实验 1。

**参考 / 关联**

- self-attention 接入点：[diffusion_denoiser.py](../../src/brepnet/models/diffusion_denoiser.py)
- diffusion 主流程：[diffusion.py](../../src/brepnet/models/diffusion.py)
- visibility 数据设计：[2026-06-18 NOTE](../2026-06-18/NOTE.md)
- 条件控制参考：[ControlNet](https://arxiv.org/abs/2302.05543)、[T2I-Adapter](https://arxiv.org/abs/2302.08453)

### 3. 分阶段实验路线

**为什么这么做**

- attention 参数化、topology predictor、diffusion-to-topology 梯度和 visibility weighting 是四个独立变量；一次全部加入将无法判断收益或失败来自哪里。

**具体实施**

1. **实验 0：GT oracle + log-multiplier**
   - 不加入 topology predictor。
   - 将 GT adjacency 从 additive `+2` 改为 `log(1+P)`，使邻接 pair 最多乘 2，非邻接 pair 乘 1。
   - 只回答 topology-to-diffusion 接口是否有效。
2. **实验 1：exact topology predictor，阻断 diffusion 梯度**
   - topology head 用 exact adjacency BCE 和 face-valid BCE 训练。
   - diffusion 使用 `log(1+P.detach())`。
   - 只回答监督得到的连续 topology prior 是否可被 diffusion 使用。
3. **实验 2：打开联合梯度**
   - 唯一变化是移除 `detach()`。
   - 比较 diffusion loss 回传 topology head 是否改善最终 valid STEP 和条件一致性。
4. **实验 3：visibility-weighted exact topology loss**
   - 模型结构不变，只修改 topology BCE 的 pair weight。
   - 使用连续 face visibility `v_i`、`v_j`：

```text
both_visible   = v_i * v_j
visible_hidden = v_i*(1-v_j) + (1-v_i)*v_j
both_hidden    = (1-v_i) * (1-v_j)

pair_weight =
    1.0 * both_visible
  + 0.4 * visible_hidden
  + 0.1 * both_hidden
```

   - 所有 valid face pair 仍计算 exact BCE；visibility 只控制梯度优先级。
   - edge/non-edge class balancing 与 visibility weight 分开处理，防止稀疏 adjacency 导致全 0 解。

**结果**

- 已确定按实验 0 -> 1 -> 2 -> 3 顺序推进，每一步只改变一个核心变量。
- 实验 0 是下一项正式实验；实验 1 之后才需要确定 topology head 的具体结构，实验 3 之前不调 visibility 权重。

**参考 / 关联**

- 当前路线图：[ROADMAP.md](../ROADMAP.md)
- 当前 topology bias：[diffusion_denoiser.py](../../src/brepnet/models/diffusion_denoiser.py)
- visibility cache 生成与审计：[2026-06-18 NOTE](../2026-06-18/NOTE.md)

### 4. 实验 0 的代码改动范围

**为什么这么做**

- 实验 0 必须保持 GT adjacency、数据、image condition、diffusion 架构和训练协议不变，只替换 attention prior 的数值参数化。

**具体实施**

- 修改 [diffusion_denoiser.py](../../src/brepnet/models/diffusion_denoiser.py)：
  - 为 topology soft bias 增加显式 `parameterization` 配置；
  - 保留旧 `additive` 行为以兼容已有实验；
  - 新增 `log_multiplier`：

```text
multiplier = 1 + (max_multiplier - 1) * adjacency_weight
bias = timestep_weight * log(multiplier)
```

  - oracle 与 learned topology 共用同一个 helper，防止两套公式再次漂移。
- 修改 [diffusion.yaml](../../configs/model/diffusion.yaml)：
  - 默认继续使用 `parameterization: additive`，避免旧配置静默改变语义；
  - 增加 `max_multiplier` 字段。
- 新增实验 0 配置：[train_diffusion_oracle_topology_log_multiplier.yaml](../../configs/train_diffusion_oracle_topology_log_multiplier.yaml)。
  - `source: gt_adjacency`
  - `mode: soft_bias`
  - `parameterization: log_multiplier`
  - `max_multiplier: 2.0`
  - real-photo 与 DeepCAD v7 24-view 数据协议保持不变。
- 新增 topology bias 单元测试：[test_topology_attention_bias.py](../../src/brepnet/test/test_topology_attention_bias.py)。
  - `P=0` 时 bias 为 0；
  - 最大 timestep、`P=1` 时 bias 为 `log(2)`；
  - 相同 base logits 下 attention odds ratio 为 2；
  - hard mask 行为不变；
  - learned probability 路径梯度非空且 finite。
- 不修改：
  - [diffusion.py](../../src/brepnet/models/diffusion.py) 的训练/推理主流程；
  - condition encoder、VAE、dataset、latent cache 和 post-process。

**结果**

- 已完成实现：
  - [diffusion_denoiser.py](../../src/brepnet/models/diffusion_denoiser.py) 新增共享 `build_soft_topology_attention_bias`，oracle / learned topology 共用。
  - 旧配置缺少 `parameterization` 时默认 `additive`，并保留原 `t/1000 * scale` 数值，兼容历史 checkpoint。
  - `log_multiplier` 使用 `log(1 + (max_multiplier-1)*P)`；`P=0/1` 在最大 timestep 对应 `1x/2x`。
  - [diffusion.yaml](../../configs/model/diffusion.yaml) 新增显式 `parameterization` 和 `max_multiplier`。
  - 训练命令记录在 [command.sh](./command.sh)。
- 验证：
  - Python compile：通过。
  - 五项公式 / hard-mask / backward 测试：全部通过。
  - Hydra compose：解析为 `parameterization=log_multiplier`、`max_multiplier=2.0`、real-photo ratio `1.0`。
  - 完整模型 build：得到 `OracleTopologySelfAttentionMask`，新配置字段正确。
  - 单个真实 DeepCAD v7 样本 forward：`face_adj=(1,30,30)`，`diffusion_loss=1.1495`，所有 loss finite。
  - 当前 `torch` 环境未安装 `pytest`，因此直接调用测试函数执行断言；测试代码本身保留为标准 pytest 文件。
- 实验 0 的决策标准：
  - **继续**：相对 no-topology baseline 提升 valid STEP，且不明显损害 image-condition geometry；
  - **停止该接口**：GT oracle 在正确的乘 2 参数化下仍无稳定下游收益；
  - **迭代强度**：方向有效但收益较弱时，后续才测试 `max_multiplier`，不与实验 0 同时调参。

**参考 / 关联**

- 当前实现：[diffusion_denoiser.py](../../src/brepnet/models/diffusion_denoiser.py)
- 基础模型配置：[diffusion.yaml](../../configs/model/diffusion.yaml)
- 实验配置：[train_diffusion_oracle_topology_log_multiplier.yaml](../../configs/train_diffusion_oracle_topology_log_multiplier.yaml)
- 单元测试：[test_topology_attention_bias.py](../../src/brepnet/test/test_topology_attention_bias.py)
- 训练命令：[command.sh](./command.sh)
- 当前 oracle 配置：[train_diffusion_oracle_topology.yaml](../../configs/train_diffusion_oracle_topology.yaml)
- no-topology baseline：[2026-06-26 NOTE](../2026-06-26/NOTE.md)

## 今日结论

- 当前 two-step 路线的问题不能继续简化为“CVAE topology 不够准”；topology bias 的 `+2` 实际对应约 `7.39x` attention mass，是必须优先修正的接口问题。
- 新路线将 topology 从独立离散中间结果改为 diffusion 内部的连续、可微 pairwise structural prior。
- exact topology loss 先作为受控监督验证接口；one-to-many 和 visibility weighting 延后到基本联合训练链路成立之后。
- 下一步只实施实验 0，不同时加入 predictor、联合梯度或 visibility weighting。

## 待办

- [x] 实现实验 0 的 `log_multiplier` topology bias。
- [x] 增加配置与单元测试，并完成 Hydra compose / forward / gradient smoke test。
- [x] 给出 8-GPU real-photo oracle 训练命令。
- [ ] 训练完成后按统一 legacy post-process / evaluator 协议评估。
- [ ] 实验 0 完成后，根据 valid STEP 与 condition consistency 决定是否进入实验 1。
