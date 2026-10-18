# UV boundary VAE：实现架构快照

本文件描述 `train_vae_uv_boundary` **已经实现的网络和数据合同**，用于阅读结构、核对实验变量；不把这些说明转换成每层重复执行的 shape/type 检查。

| 项目 | 本快照 |
|---|---|
| 架构实现基准 commit | `7b7d23a334c4b01fa10a7c89d958174aafc0367f` |
| 配置入口 | `configs/train_vae_uv_boundary.yaml` → `train_vae_intersection` → `model/vae_intersection` → `model/vae_light` |
| 模型 | `AutoEncoder_intersection`，位于 `src/brepnet/models/vae/intersection.py` |
| 启用 | CNN encoder、legacy intersection、一轮 refinement、最终 curve conditioning、辅助 UVPathHead |
| 关闭 | UV **面编码器**、GAT edge-value 增量、local intersection、curve point feedback |
| 更新范围 | encoder/posterior 冻结且保持 eval；其余 decoder 和新 UV head 训练 |
| 归档实验状态 | **已实现，尚未正式训练**；既有记录完成标签小样本、真实 checkpoint 的 FP32/BF16 前反向和无 GT 推理，无 optimizer step |

状态依据为 [README](README.md) 和 [远端实现验证](validation/remote_smoke.json)，不是对服务器当前进程的实时查询，也不是模型收益结论。包拆分的旧新等价性见 [结构验证](../vae_structure/validation/equivalence.json)。本轮新增 UV 辅助头，不是重启此前的 UV face encoder 实验。

阅读入口：[完整网络](#2-全模型主图) · [原 inter](#41-一个有向面对如何产生一条候选边) · [refinement](#51-intersectionrefiner先读取偏差再交换信息) · [新增 UV 层](#61-uvpathhead-的每层与输出职责) · [dataset 改动](#11-dataset如何读取和打包uv监督) · [UV loss](#63-两个新增-loss-的梯度路径)。

注意代码的两层命名：`AttnIntersection` 是原面对注意力 block，输入 `[P,2,32]`、输出 `[P,64]`，本次没有替换它；`IntersectionDecoder` 装配原 inter、几何头、refiner、conditioning 和新增 UVPathHead。最新 UV 改动要从后者的 `forward` 往下读。

## 1. 输入、符号与坐标

下面是 packed batch：不同对象的面直接拼接，不先补齐成 30 个 token。

| 符号 / 字段 | 形状 | 含义 |
|---|---|---|
| `B`、`face_counts` | `[B]` | 对象数；每个对象面数 `N_b` |
| `F = sum(N_b)` | — | batch 中全部面的数量 |
| `face_points` | `[F,16,16,6]` | 对象归一化坐标中的 XYZ + 面法向 |
| `edge_points` | `[E_gt,16,6]` | GT 有向边的 XYZ + 切向；第一维按 edge ID 索引 |
| `edge_face_connectivity` | `[R,3]` | 每行 `(edge_id, face_i, face_j)`；按连接行编号 |
| `face_norm` / `edge_norm` | `[F,16,16,6]` / `[E_gt,16,6]` | 各自 primitive 的局部归一化重建目标 |
| `face_bbox` / `edge_bbox` | `[F,4]` / `[E_gt,4]` | `center_xyz[3] + scale[1]`，不是 6 维包围盒 |
| `attn_mask` | `[F,F]` | 屏蔽跨对象的面注意力；同一对象内允许互相读取 |
| `P = sum(N_b*(N_b-1))` | — | decoder 自行枚举的全部对象内、有向、非 self 面对 |
| `edge_uv_targets.uv` | `[R,2,16,2]` | 连接行、两侧 owner、边采样序号、`(u,v)` |
| `valid` / `residual` / `method` | 各 `[R,2,16]` | 有效掩码、标签回代距离、提取方法 |

当前 accepted 数据合同要求每个 GT 边样本恰好被连接表覆盖一次，因此 `R=E_gt`；**数量相等不代表行号等于 edge ID**。每个有向 owner pair 最多一条 GT 边，且必须有互反边；self/seam pair 和同一对面间多条边不属于此合同。

loader 的几何张量为 float32，面/边 ID 与 counts 为整数，mask 为 bool；网络可在 bf16 autocast 下运行，下面形状图不表示所有中间值均为 float32。UV 标签的 `method` 为 0=无效、1=native pcurve、2=投影；有效性由 `valid` 明确控制。

`face_points` 的空间轴按 `[v_index,u_index]` 存储；UV 标签按 `(u,v)` 保存。标签是原 CAD 参数图范围归一化后的坐标，不是 XYZ，也不是中心/尺度。闭环两端可能分别在 0 和 1，不能无条件取模合并。数据刚体旋转同时作用于 XYZ 和方向，不改变 UV 标签。所有几何距离使用当前对象归一化坐标的单位，不直接等于毫米。

### 1.1 dataset如何读取和打包UV监督

没有另建 UV dataset，也没有把 UV 塞进原 `data.npz`。两个数据根目录按同一个原始 CAD ID 配对：

```text
dataset.raw_data_root/{id}/data.npz
  face/edge samples + connectivity[R,3]
                 |
                 +---- AutoEncoder_dataset3.__getitem__ ----+
                 |                                         |
dataset.edge_uv_root/{id}/edge_uv.npz                        |
  edge_uv[R,2,16,2], valid/residual/method[R,2,16]             |
                 |                                         |
                 v                                         |
  load_edge_uv_sidecar: source NPZ digest + row correspondence
                 |                                         |
  keep = connectivity[:,1] != connectivity[:,2]              |
  same keep mask on connectivity rows AND UV label rows      |
                 |                                         |
  XYZ / normals / tangents rotate; UV coordinates unchanged  |
                 |                                         |
  sample: original fields + uv_targets dictionary <---------+
                 |
  AutoEncoder_dataset3.collate_fn
    connectivity edge_id += preceding objects' edge count
    connectivity face_i/j += preceding objects' face count
    UV arrays: concatenate by object/row order, no ID offset
                 |
  batch['edge_uv_targets']
    uv       [sum_R,2,16,2]  float32
    valid    [sum_R,2,16]    bool
    residual [sum_R,2,16]    float32
    method   [sum_R,2,16]    uint8
                 |
  IntersectionLoss._uv_boundary
    predictions: gather P candidates with candidate_rows[r]
    UV target:   uv[r]               (connectivity row)
    XYZ target:  edge_points[edge_id] (explicit edge ID)
    -> uv L1 + fixed-UV geometry anchor
```

实际入口继承了 `dataset: vae_joint`，因此运行类名仍是 `AutoEncoder_joint_dataset(AutoEncoder_dataset3)`。它调用父类完成上面的 UV 读取/打包，再保留既有 `boundary_targets`；当前 `AutoEncoder_intersection` 的损失不消费该 shared-boundary 字段。这只是沿用既有数据路径，不表示切换到了 joint decoder。

具体修改在 `src/brepnet/dataset.py` 的 `AutoEncoder_dataset3.__init__`（保存可选根目录）、`__getitem__`（配对读取与同步过滤）、`collate_fn`（新增 batch 字段）。标签文件的读取和来源核对在 `src/brepnet/data/edge_uv.py:load_edge_uv_sidecar`；离线 OCC 提取在 `src/brepnet/data/extract_edge_uv.py:extract_edge_uv`。训练读取的是已有标签，不在每个 batch 中调用 OCC。

基础 `configs/dataset/vae.yaml` 的 `edge_uv_root: null` 保留旧数据行为。`train_vae_uv_boundary` 开启 `model.uv_boundary.enabled`，要求显式提供 `dataset.edge_uv_root`；142 上约定为 `/mnt/d/data/deepcad_v7_uv/v1`。指定目录后缺失标签文件会在读取前报出；文件中个别点的 `valid=false` 则仅屏蔽这些点的 UV/anchor loss，原几何监督仍参与训练。`residual` 和 `method` 用于标签质量/来源核对，不是额外网络输入或额外 loss。

## 2. 全模型主图

```text
VAE reconstruction inputs                         training-only targets
face_points [F,16,16,6]  edge_points [E_gt,16,6]    local/bbox/XYZ, connectivity,
          |                     |                 UV [R,2,16,2] + valid
          +----------+----------+                          |
                     v                                     |
  encoder: HolisticEncoder  [FROZEN / eval / no_grad]       |
  face CNN + edge CNN -> face attention -> GAT -> global   |
          | face_features [F,32]    | edge_features         |
          v                        | [E_gt,64]              |
  posterior: FacePosterior         |                       |
  [FROZEN / no_grad]               +----> shared edge head |
  mean/std -> z [F,32]                    (no conditioning) |
          |                               | teacher geom   |
          v                               +--------------->|
  decoder: IntersectionDecoder                             |
  +----------------------------------------------------+   |
  | face_features: 8-layer face Transformer -> F0[F,32] |   |
  |                    |                               |   |
  | all ordered face pairs [P,2]                       |   |
  |                    v                               |   |
  | inter: 8-layer AttnIntersection -> E0[P,64]         |   |
  |       |                         |                  |   |
  | shared geometry heads      classifier + symmetry   |   |
  | S0[F,16,16,3], C0[P,16,3]   logits0[P] -> p0[P]    |   |
  |       +--------------+----------+                  |   |
  |                      v                             |   |
  | refiner: geometry read + candidate message exchange|   |
  |          -> delta_F[F,32], delta_E[P,64]             |   |
  |                      |                             |   |
  | F1 = F0 + delta_F; E1 = E0 + delta_E                |   |
  |       |              |               |             |   |
  | face head       edge head +          classifier    |   |
  |                 curve conditioning   + symmetry    |   |
  | S1[F,16,16,3]   C1[P,16,3]           logits1[P]    |   |
  |       |              |                             |   |
  |       |         E1 + F1 owners                      |   |
  |       |              v                             |   |
  |       |       uv_boundary: UVPathHead               |   |
  |       |       U_pred[P,2,16,2]                      |   |
  +-------+--------------+-----------------------------+   |
          +--------------+--------------------------------->|
                                      criterion: IntersectionLoss
                                      -> scalar total_loss

Inference: supplied z -> same decoder -> probability > 0.5 -> export
           No GT edges, connectivity, UV, or loss calculation required.
```

`F0/F1` 是面特征，`E0/E1` 是有向候选边特征；它们还不是 XYZ 点。`S0/S1`、`C0/C1` 才是解码几何。模型没有额外的顶点、loop 或 coedge 拓扑 latent。`BaseVAE` 只定义接口，不注册新的网络层或权重。

## 3. Encoder 与 posterior 的内部层级

### 3.1 两个 CNN

`encoder.face_coords` 和 `encoder.edge_coords` 读取 **object-space 的 `*_points`**；`*_norm` 是 loss 目标，不是这两个 CNN 的输入。`RB` 表示一个 `ResBlockXD`，每阶段恰好一个。

```text
encoder.face_coords                         encoder.edge_coords
[F,6,16,16]                                 [E_gt,6,16]
 | Conv2d 6->96, k3/s1/p1                     | Conv1d 6->96, k3/s1/p1
 | LayerNorm + LeakyReLU                      | LayerNorm + LeakyReLU
[F,96,16,16]                                [E_gt,96,16]
 | RB 96->192; MaxPool2d(2)                   | RB 96->192; MaxPool1d(2)
[F,192,8,8]                                 [E_gt,192,8]
 | RB 192->384; MaxPool2d(2)                  | RB 192->384; MaxPool1d(2)
[F,384,4,4]                                 [E_gt,384,4]
 | RB 384->768; MaxPool2d(2)                  | RB 384->768; MaxPool1d(2)
[F,768,2,2]                                 [E_gt,768,2]
 | RB 768->768                               | RB 768->768
 | Conv2d 768->8, k1                          | Conv1d 768->32, k1
[F,8,2,2] -> flatten -> [F,32]               [E_gt,32,2] -> flatten -> [E_gt,64]

RB(x) = ReLU( LN(conv2(ReLU(LN(conv1(x))))) + skip(x) )
conv1/2: k3/s1/p1; skip: identity, or k1 projection + LN when width changes.
CNN LayerNorm covers [channels, spatial axes], not channels alone.
```

### 3.2 面间信息与图连接

```text
CNN faces [F,32]
  | face_attn_proj_in: Linear 32->768
  | face_attn: 8 x TransformerEncoderLayer
  |   pre-LN, self-attention 16 heads, FFN 768->2048->768, ReLU, dropout 0.1
  |   attention uses [F,F] object mask; final LayerNorm(768)
  | face_attn_proj_out: Linear 768->32
  + add CNN faces --------------------------------------------> [F,32]
  |
  | graph_face_edge: 5 x (GATv2Conv + LeakyReLU)
  |   GAT width 32->32, 1 head, edge_dim=64
  |   edge_index = connectivity[:,1:].T                    [2,R]
  |   edge_attr  = CNN edges[connectivity[:,0]]             [R,64]
  |   each update: h = LeakyReLU(h + GAT(h, edge_index, edge_attr))
  |   after all five: faces = input_faces + h
  |
  | per-object scatter_mean                               [B,32]
  | global_feature1: Linear 32->32, LeakyReLU, Linear 32->32
  | repeat context to its owner faces                     [F,32]
  | concat faces + context                                [F,64]
  | global_feature2: Linear 64->32, LeakyReLU, Linear 32->32
  + add faces -----------------------------------------------> [F,32]
                                                   holistic face_features
```

GAT 由真实连接表限制消息来源，边特征参与注意力权重；本配置没有 `EdgeValueGATv2Conv.value_net` 的额外边内容消息。GAT 使用其默认 self-loop 机制。全局聚合让每个面接收本对象的摘要。因而最终 32 维已经混合面、连接和对象上下文，不能再把它直接解释为四个独立象限。

### 3.3 FacePosterior 与采样

```text
face_features [F,32]
  -> posterior.projection: Linear 32->64 -> LeakyReLU -> Linear 64->64
  -> reshape [F,32,2] -> mean [F,32], log_variance [F,32]
  -> std = exp(0.5 * log_variance)
  -> z = mean + std * epsilon  OR  z = mean                 [F,32]
  -> cached statistics = concat(mean, std)                 [F,64]
```

`sample(..., v_is_test=True)` 才选择 mean；`model.eval()` 本身不切换此选项。本训练配置 `vae_validation_inference=false`，因此普通 training **和 validation loss** 都走随机采样；独立测试 `v_test=True` 用 mean。encoder/posterior 冻结不等于取消采样。posterior 返回的 KL 已乘 `1e-6`、采用 sum reduction；本冻结配置的总 loss 对 KL 权重为 0。

## 4. 原 intersection 与几何头

`decoder.face_features` 先执行一次 `Linear(32,768)` → 8 层 pre-LN TransformerEncoderLayer → `Linear(768,32)`，得到 `F0`。每层 16 heads、FFN2048、ReLU、dropout0.1；使用同一对象内的面注意力，没有 encoder 那个额外的末尾 LayerNorm。refinement 后不再运行这 8 层。

### 4.1 一个有向面对如何产生一条候选边

```text
F0[pairs.indices] [P,2,32]
  -> inter.attn_proj_in: Linear 32->512
  -> add inter.pos_encoding [1,2,512]
       | token 0                       | token 1
       v                               v
     tgt [P,1,512]                    mem [P,1,512] (fixed)
       |                               |
       +--- 8 x inter.layers ----------+
       | each TransformerDecoderLayer:
       |   pre-LN self-attention: Q/K/V = tgt
       |   pre-LN cross-attention: Q = tgt, K/V = mem
       |   pre-LN FFN 512->2048->512, ReLU
       |   residual at each sublayer; 8 heads; dropout 0.1
       v
  inter.attn_proj_out: Linear 512->64 -> squeeze -> E0 [P,64]
       |
  classifier: Linear 64->1 -> raw_logits [P]
       |
  logits[k] = (raw_logits[k] + raw_logits[reverse[k]]) / 2
       -> sigmoid -> p0 [P]
```

`(i,j)` 与 `(j,i)` 是两个独立候选，共用网络权重；单次 pair 运算输出 **一个** 64 维特征。只有 logits 双向平均，边特征和曲线没有强行平均。所有候选都进入 refiner，训练时不按 GT 连通性筛选，也不先按 0.5 阈值裁掉。

### 4.2 复用的 PrimitiveDecoder

下图中的两种 `points_decoder` 各有 4 个 RB。转置卷积均为 k2/s2；最终普通卷积为 k1。coarse/refined **共用同一套参数**。

```text
decoder.face_geometry.points_decoder:
[F,32] -> [F,8,2,2] -> RB 8->768
 -> ConvTranspose2d 768->384 -> [F,384,4,4] -> RB 384->384
 -> ConvTranspose2d 384->192 -> [F,192,8,8] -> RB 192->192
 -> ConvTranspose2d 192->96  -> [F,96,16,16] -> RB 96->96
 -> Conv2d 96->6 -> permute -> local_points [F,16,16,6]

decoder.edge_geometry.points_decoder:
[P,64] -> [P,32,2] -> RB 32->768
 -> ConvTranspose1d 768->384 -> [P,384,4] -> RB 384->384
 -> ConvTranspose1d 384->192 -> [P,192,8] -> RB 192->192
 -> ConvTranspose1d 192->96  -> [P,96,16] -> RB 96->96
 -> (refined-only conditioning) -> Conv1d 96->6 -> [P,16,6]

face_geometry.bbox_decoder / edge_geometry.bbox_decoder:
[F,32] or [P,64] -> RB(dim=0) input->768 -> 6 x RB 768->768
 -> (edge refined-only conditioning) -> Linear 768->4
 -> center_scale [F,4] or [P,4]

global XYZ = local_points[..., :3] * scale + center   (FP32 arithmetic)
```

`dim=0` 的 RB 将上面的两层卷积替换成 Linear，其余残差/LN/ReLU 结构相同。方向的后三个通道参与 local 重建；导出的 XYZ 使用前三个通道。边 teacher 路径也复用 `edge_geometry`，输入 encoder 的 `[E_gt,64]`，不启用 conditioning；冻结 encoder 后，此辅助路径仍训练共享几何头。

## 5. 一轮 refinement 与最终 curve conditioning

### 5.1 IntersectionRefiner：先读取偏差，再交换信息

```text
C0[P,16,3] + two owner S0 grids + pairs
  -> closest_surface_points (each 16x16 grid -> 450 triangles)
  -> observations [P,16,17]:
     C(3), nearest_i(3), nearest_j(3), C-nearest_i(3), C-nearest_j(3), d_i(1), d_j(1)
  -> geometry_read.point_encoder: Linear 17->128, LN, SiLU, Linear 128->64, SiLU
  -> [P,16,64] -> concat(mean, amax over 16) [P,128]
  -> geometry_read.pool_projection: Linear 128->64 -> G[P,64]

concat(E0, G) [P,128]
  -> message_encoder: Linear 128->128, LN, SiLU -> M[P,128]
  -> p0-weighted mean to BOTH owners -> A[F,128]

concat(E0, G, F0_i, F0_j, A_i, A_j) [P,448]
  -> relation_encoder: Linear 448->128, LN, SiLU, Linear 128->128, SiLU
  -> T[P,128] --- edge_delta: Linear 128->64 -----------> delta_E[P,64]
       |
       +-> p0-weighted mean to BOTH owners -> B_face[F,128]
       +-> concat(F0, B_face) [F,160]
           -> face_update: Linear 160->128, LN, SiLU
           -> face_delta: Linear 128->32 -------------> delta_F[F,32]

F1 = F0 + delta_F; E1 = E0 + delta_E
  -> shared geometry heads and shared classifier once more
```

最近点可位于三角片内部或边界；这是预测网格的近似几何读取，不是真实 B-spline 精确求交。读取、距离与加权累计使用 FP32。每条有向候选向两个 owner 发消息，互反候选仍各自贡献；分母是对应概率之和，无硬阈值。单面对象没有候选，返回零增量。一共一轮反馈，没有迭代到收敛的循环。

### 5.2 FacePairConditioning：只调制 refined edge 的最后一层之前

```text
concat(E1[P,64], F1_i[P,32], F1_j[P,32]) -> [P,128]
  -> edge_geometry.conditioning.projection: Linear 128->128, SiLU
       +-> point_affine: Linear 128->192 -> gamma_p, beta_p [P,96]
       |   h[P,96,16] = (1+gamma_p[:,:,None])*h + beta_p[:,:,None]
       |   -> original final Conv1d(96,6)
       +-> bbox_affine: Linear 128->1536 -> gamma_b, beta_b [P,768]
           h[P,768] = (1+gamma_b)*h + beta_b
           -> original final Linear(768,4)
```

refiner 的增量头和 conditioning 的 affine 头在**最初引入时**零初始化；本轮从已训练 conditioned checkpoint 保留这些权重，不能当作仍为零。本配置没有最终逐点 XYZ 增量的 `CurvePointFeedback`。

## 6. 本轮新增：UVPathHead 和固定对应监督

### 6.1 UVPathHead 的每层与输出职责

```text
E1[P,64] + owner F1[P,2,32]                   t = k/15, k=0..15 [16,1]
  -> concat [P,128]                            -> position:
  -> uv_boundary.context:                        Linear 1->32, SiLU,
     Linear 128->128, SiLU                        Linear 32->128
  -> [P,128]                                     [16,128]
       | broadcast over 16                         | broadcast over P
       +-------------------- add ------------------+
                              |
                          [P,16,128]
                              |
  uv_boundary.layers: 2 independently initialized TransformerEncoderLayer
    each: pre-LN self-attention (Q/K/V = 16 point tokens), 4 heads
          residual; pre-LN FFN 128->256->128, GELU; residual; dropout 0
                              |
  uv_boundary.output: LayerNorm(128) -> Linear 128->4
                          [P,16,4]
                              |
  reshape [P,16,2,2] -> permute -> U_pred[P,2,16,2]
      four channels per point: (u_i, v_i, u_j, v_j)
```

这里的 `t` 是边采样序号的位置编码；两个 UV 输出对应两张 owner 面，不是起点/终点。输出不加 sigmoid/clamp。UV head 读取 refined features，不将预测 UV 回灌 XYZ 几何头；它通过训练梯度约束这些共享特征。新增 **286,532 参数、34 个 state keys**，初始化保留原模型 **736 个权重项**。

### 6.2 P 个候选与 R 行 GT 的映射

```text
GT row r: connectivity[r] = (edge_id=e, owner_i=i, owner_j=j)
                    |                          |
                    |                          +-> find candidate k with pairs[k]=(i,j)
                    |                              candidate_rows[r] = k
                    v
  XYZ/local/bbox target = edge_points/edge_norm/edge_bbox[e]
  UV target            = edge_uv_targets.uv[r]       (NOT uv[e])
  prediction           = C1[k], U_pred[k]
  classification       = positive at k; remaining candidates negative
  twin                 = reverse[k], matched to its explicit GT edge ID
```

候选生成只看 `face_counts`，不读取 GT adjacency。GT 映射只在监督中使用。全部 P 个候选接受 balanced BCE；边几何、边特征和 UV 监督使用映射后的 GT 正例；面几何监督使用全部 F 个面。

### 6.3 两个新增 loss 的梯度路径

令 `I(S,U)` 为对 `[v,u]` 网格在 `(u,v)` 位置的双线性读取：`grid_sample` 使用 `align_corners=True`、`padding_mode=border`，把归一化 UV 映射到 `[-1,1]`。

```text
U_pred[k] -----------------------------> L_uv = masked L1(U_pred, U_gt)
    | gradient to UV head, F1/E1, shared decoder

S1[owners] -- I(., U_gt) --+                 (fixed GT UV, detached)
C1[k] -------------------+-> C1 - I(S1,U_gt) - D_gt -> masked L1 -> L_anchor
                         |                         |
GT S, GT C -- no_grad ----+                         +-> gradient to XYZ heads,
  D_gt = C_gt - I(S_gt,U_gt)                            F1/E1 and shared decoder
```

anchor 使用 **GT UV**，不是预测 UV，因而不能通过滑动预测 UV 来逃避边面对应约束。`D_gt` 是由 GT XYZ 和同样的 16×16 插值计算的三维残差，扣除网格表示本身的插值误差；它不是 sidecar 中记录质量的标量 `residual`。anchor 不直接训练 UV head，UV L1 才直接训练它。只有 `valid` 位置参与这两项；无效位置在采样前置零并屏蔽，原有几何监督照常保留。`uv_valid_fraction` 只作覆盖率统计。

### 6.4 当前总损失

| 项 | 本配置实际乘数 | 监督内容 |
|---|---:|---|
| coarse face/edge local、bbox | 各 0.25 | local 的 6 通道、center/scale 的 4 通道 L1 |
| refined face/edge local、bbox | 各 1 | 同上 |
| coarse / refined classification | 0.025 / 0.1 | 正负类分别均值后平衡的 BCE |
| edge_feature | 1 | coarse E 对冻结 encoder edge feature，GT edge ID 对应 |
| encoder_edge_local / bbox | 各 1 | teacher feature 经共享边头后的重建 |
| face_xyz / edge_xyz | 各 1 | refined object-space XYZ 的 L1 |
| incidence | 0.1 | 边到两张预测网格距离，扣除 GT 同类网格距离及 `1e-5` 后取 ReLU |
| twin | 0.1 | 用 GT 折线确定互反边采样对应，扣除 GT 残差及 `1e-5` 后取 ReLU |
| uv / uv_anchor | 0.01 / 0.1 | 上述有效 UV 与固定位置的几何对应 |
| KL | 0 | 本轮冻结 encoder/posterior；仍可记录原始 KL |

`total_loss` 为这些加权项之和；precision/recall、增量大小、UV 覆盖率不加入 loss。加入新项后的 total loss 与旧训练数值口径不同，不能据此直接判断模型进步。

## 7. 训练、推理与运行追溯

训练通过统一 `src/brepnet/train.py` 调用 `encode → sample → decode → criterion`。新 run 使用 `init_uv_boundary_checkpoint`，严格保留已采用的 conditioned CNN VAE，仅初始化新 UV head；全模型加载和 resume 仍严格匹配。参考初始化权重为 [README](README.md) 指定的 epoch157 / step57354 checkpoint。新配置预算为 8 卡×每卡16、bf16-mixed、Adam、学习率 `2e-5`、seed42、100000 steps；这是配置预算，不表示已经执行。

`decode(z, face_counts)` / `decode_state(...)` 返回可微的所有候选状态；省略 counts 表示一个对象。`decode_latents(z)` 和基类 `inference(z)` 导出单对象预测，不要求 GT UV 或 GT 连接表，也不在这里追加 face 去重。选中候选数记作 `K`，逐对象导出 `pred_face[N,16,16,3]`、`pred_edge[K,16,3]`、`pred_edge_face_connectivity[K,3]`、`pred_edge_uv[K,2,16,2]` 及概率，表示标签为 `legacy_half_edges`。原后处理不消费 `pred_edge_uv`；仍从 XYZ 走拟合、裁剪、缝合与 solid 检查。UV 辅助监督是否提高合法率，必须后续评估。

当前候选分块为256，几何投影分块为128，UV head 分块为256，`checkpoint_candidates=false`；分块管理内存，不增加网络层或减少候选。VAE 本身不强制30面：单对象若有30面则 `P=870`，不是900；diffusion 的固定槽位/padding 是另一个接口。

**本文件的 hash 是架构设计快照，不替代真实运行记录。** 正式运行必须补齐 `dataset.edge_uv_root`、`model.init_uv_boundary_checkpoint` 及实际数据/split。每次调用由 `run_management.py` 保存：

```text
checkpoints/uv_boundary/candidate/<run UTC>/
  metadata/run.json
  metadata/invocations/<invocation UTC>/
    config.yaml       # resolved config：最终消费的结构、路径和参数
    git.json          # 实际执行 commit、branch、工作区状态
    git.diff          # 相对执行 commit 的未提交差异
    command.json      # argv、seed、resume 来源、调用身份
  weights/            # checkpoint；是否恢复 optimizer/step 由 resume 决定
  logs/  wandb/  raw_predictions/  visualizations/
```

继续训练以该次 invocation 的 resolved config、Git 信息、数据版本和 checkpoint 为准，不能仅凭本快照或文件名推定一致。`init` 是新 run；`resume` 恢复原 run 的训练状态。

源码定位：装配与导出在 `models/vae/intersection.py`；encoder/posterior 在 `encoders.py`；原 CNN 工厂在 `legacy.py`；注意力、几何头、refiner 分别在 `decoders/intersection.py`、`primitives.py`、`refinement.py`；投影在 `geometry.py`；监督在 `losses/intersection.py`；这些路径均相对 `src/brepnet/`。
