# Conditional Intersection：架构快照

日期：2026-10-29（北京时间）。网络实现Git hash：`74c7ecb4`，preflight路径引用修正`9c6b5908`。配置入口`configs/train_vae_conditional_intersection.yaml`。状态：本轮8项检查及54项旧decoder／VAE／shared-curve回归共62项通过，142真实FP32/BF16前后向及最大尺寸batch8检查通过，两卡短运行完成2次更新；resume验收进行中，未正式训练。本快照不代表训练完成或有效性已验证。

## 1. 数据流与保留范围

F：batch总面数；每个对象有n个面；P=sum(n*(n-1))是有向面对候选数，K=P/2。单个30面对象有P=870；P不是16。面格点数16×16、每条曲线采样16点是不同的轴。所有跨对象attention／候选仍按原face_counts隔离。

```text
训练数据：face_norm[F,16,16,6], edge_norm[E_gt,16,6], bbox, connectivity
          edge_uv_targets[R_gt,2,16,2]及valid（仅loss使用）
    |
    v
原 HolisticEncoder（CNN + GAT + 全局交互）              [本轮可训练]
    | [F,32]
原 FacePosterior（mean/std；训练sample，验证mean）       [本轮可训练]
    | Z[F,32] <-- 纯解码推理入口；不读取GT连接或GT UV
    v
原 FaceFeatureDecoder（8层768维，16 heads）
    | F0[F,32]
    +--------------------------------------------+
    |                                            |
    |                            原FaceGeometry.forward_with_local_features
    |                              | S0[F,16,16,3]
    |                              | H0[F,96,16,16]，末个Conv2d之前
    |                              | CNN只运行一次；H0不detach
    +---------+--------------------+
              v
新增 ConditionalIntersection(F0,H0,pairs[P,2])
              | h0[P,64]，每个有向面对一个query
              +--> 原CurveGeometry --> coarse C0[P,16,3]
              +--> 原classifier/symmetrize --> p0[P]
              |
原 IntersectionRefiner(h0,F0,C0,S0,p0,pairs)
              | h1[P,64], F1[F,32]
              +--> 原FaceGeometry --> S1[F,16,16,3]
              +--> 原classifier/symmetrize --> final logits[P]
              |
原 SharedCurveFusion(h_ij,h_ji) --> g[K,64]
              |
原 CurveGeometry + FacePairConditioning
              | 一条共享XYZ C[K,16,3]及local[K,16,6],bbox[K,4]
              +--> 反向半边仅反转采样轴 --> XYZ[P,16,3]
              +--> 原CurveConditionedUVPathHead --> auxiliary UV[P,2,16,2]
              |
原 IntersectionLoss（权重不变；解冻后KL参与优化）
```

输入及半边输出字段与来源相同；输出仍为`legacy_half_edges`，没有可微B-Spline或UV生成XYZ路径。保留原refinement，不额外堆叠correction。原模型／loss细节见[共享曲线快照](../shared_curve_uv/ARCHITECTURE.md)。

## 2. 空间memory与单个参考UV

主要实现：`src/brepnet/models/vae/decoders/conditional_intersection.py`。小module分别承担空间memory、参考点、位置编码、单层attention和8层装配，不将完整模型塞进一个class。

```text
F0[F,32] ------------------------------ H0[F,96,16,16]
  | 原Linear(32,512)                      | flatten HW + transpose
  |                                      | [F,256,96]
  |                                      | FaceSpatialMemory:
  |                                      | Linear(96,512), weight/bias=0
  | [F,512]                              | [F,256,512]
  +---------- broadcast add -------------+
                    | +原memory-side PE[512]
                    v
                 M[F,256,512]

pairs[P,2]=(i,j)
  F0[i][P,32] --------- F0[j][P,32]
          +----- cat ----+
                 | [P,64]
            ReferenceUVHead
                 | Linear(64,128) -> ReLU
                 | Linear(128,2) -> sigmoid
                 v
              r_ij[P,1,2] = (u_ref,v_ref)，位于面j参数网格
              同一次forward只预测一次，8层共用；无直接GT参考点loss

  原Linear(32,512)(F0[i]) + 原query-side PE[512]
                 | q0[P,1,512]
                 v
  8 x ConditionalIntersectionLayer
    query q[P,1,512]读取M[j][P,256,512]
                 |
          原Linear(512,64)
                 v
              h0[P,64]
```

反方向j→i有自己的有向query与r_ji，并读取M[i]。没有先把两个面拼成512-token memory；没有16个曲线点query。memory和各层K/V先按面计算，再按candidate分块gather，避免重复计算同一面的投影。新Intersection和原refinement的candidate chunk size均为64并开启activation checkpoint；原refinement几何分块保持128。重算用于节省全量训练显存，不改变层、参数、loss或全候选集合。

## 3. 每层：保留self-attention与FFN，仅替换cross-attention

以下为8层独立参数。内容Q/K/V及输出投影从来源`MultiheadAttention`迁移；旧Pre-LN、self-attention、FFN及dropout保留。

```text
q[P,1,512]
  | LN1 -> 原SelfAttention(512,8heads,dropout=.1)
  | dropout + residual
  v
q[P,1,512]
  | LN2 ---------------------+--------------------------+
  |                          |                          |
  | 原Wq:512->512            | g(q):                     | r[P,1,2]
  | q_content[P,1,512]        | 首层g=1                  | UVPositionEncoding
  |                          | 后续Linear512->512       | -> [P,1,512]
  |                          | ReLU,Linear512->512       |     |
  |                          +------- multiply ---------+
  |                                      | 新W_qpos:512->512
  |                                      | q_position[P,1,512]
  +-------- split 8 heads -------+--------+
             | q_cat[P,8,1,128] = cat(内容64,位置64)
             |
M[F,256,512] |                固定grid UV[256,2]
  |          |                  | UVPositionEncoding -> [256,512]
  | 原Wk/Wv:512->512            | 新W_kpos:512->512
  | K_content/V[F,256,512]       | K_position[256,512]
  | 按j gather及split 8heads    |
  +------------------------------+
             | k_cat[P,8,256,128] = cat(内容64,位置64)
             | value[P,8,256,64]，只传内容
             v
  scores = q_cat @ k_cat.transpose(-2,-1) / sqrt(128)
             | [P,8,1,256]
  softmax(256位置) -> attention dropout(.1) -> @ value
             | [P,8,1,64] -> merge heads[P,1,512]
  原output projection512->512 -> dropout + residual
             |
  LN3 -> 原Linear512->2048 -> ReLU -> dropout
      -> 原Linear2048->512 -> dropout + residual
             |
           q_next[P,1,512]
```

score等价于`(q_content·k_content + q_position·k_position)/sqrt(128)`。首层不用query MLP调制位置；后续层参考点不变但当前query改变，故空间打分可变。各head可关注不同位置，但没有显式沿边分工监督。所有256个位置始终参与softmax，参考UV不构成硬窗口。

## 4. 坐标、索引与监督边界

- 面CNN张量轴`[F,C,H,W]`中H=v、W=u；格点UV为`(u_idx/15,v_idx/15)`。flatten按row-major，p=16*v_idx+u_idx，与原几何／UV anchor的`[v,u]`网格一致。
- ReferenceUVHead输出在[0,1]内，是特征网格位置。它不是OCC原始参数值、曲线中心、曲线某采样点，也不是原辅助UV头输出；本轮不经它生成几何。
- 两侧参考位置未被约束为同一个XYZ。共享最终曲线依赖既有SharedCurveFusion与单次XYZ解码；新attention本身不硬保证边面相合。
- 训练GT连接只用于候选分类／监督映射，推理候选由每对象全部有向面组合独立生成。参考UV及memory均来自预测特征，无GT UV进入decoder。
- 原UV sidecar及有效点语义不变；本轮loss完整权重见[README](README.md#loss沿用来源权重完整列出)。

## 5. 初始化、梯度与运行边界

来源是32维shared_curve_uv best42471，完整SHA及路径见README。模型总参数190,236,065，全部可训练；新增7,937,922个参数，778个来源state项严格保留。新增局部投影零初始化，使M每个位置初始等于旧单memory token。eval时所有value相同，softmax加权结果在精确算术下保留原value；实际浮点计算不保证完整模型逐位一致，训练dropout在256位置上的随机行为也不等于旧单key。

142实测在两个30面CAD重复构成的batch8中：coarse边特征最大差1.34e-5、coarse边XYZ最大差3.89e-6；已有refinement对数值变化敏感，final全部候选XYZ最大差1.55e-4，出现在未选中候选。实际选中曲线XYZ最大差2.03e-6，分类选择分歧0，互反共享XYZ差0。原2件7＋20面检查的coarse边XYZ最大差6.14e-6。完整数值见[验收表](README.md#实现及研究状态)及[PREFLIGHT_LARGEST_BATCH8.json](PREFLIGHT_LARGEST_BATCH8.json)。这些是初始化与实现检查，不是完整validation合法率结果。

关闭attention dropout时，零局部投影使所有value相同，改变attention权重不改变加权value，因此参考点及位置分支首步梯度为零。正式训练保留0.1 attention dropout；随机mask作用后，权重之和不再固定，参考UV／位置分支首步可能产生非零梯度。这种梯度来自随机mask，不是位置间的空间内容差异。

CNN局部特征输入经零投影的首步梯度仍为零，但局部投影本身可以得到梯度。投影更新后，不同位置的value产生内容差异，局部内容及其位置选择路径才真正打开；面CNN的原几何监督等其他梯度路径仍存在。实现验收必须覆盖无dropout的理论启动、正式dropout条件及投影更新后的空间梯度，不把首步位置梯度非零误报为空间定位已学习。

`freeze_encoder=false`、`trainable_scope=all`，无新冻结层；KL sum系数1e-6恢复训练作用。feature matching权重1.0保留，teacher target仍detach。源checkpoint的encoder冻结历史与本次全解冻不同，正式效果不作为严格单变量attention消融。新run不继承Adam状态；恢复本实验必须使用同结构完整checkpoint。

142环境PyTorch2.8/cu128，最大尺寸batch8全模型BF16训练单卡峰值7.91GiB；正式计划为物理GPU1–7、microbatch8、累积2，即112 CAD/update，替代初始batch2／累积8方案但保持总批量。来源历史128 CAD/update。GPU0保留给现有ComfyUI。配置中的7张可见卡在进程内编号0–6，不与物理编号混用；正式运行metadata记录最终实测配置。

论文依据：[Conditional DETR](https://arxiv.org/abs/2108.06152)及[作者Transformer实现](https://github.com/Atten4Vis/ConditionalDETR/blob/main/models/transformer.py)。这是保留旧B-Rep接口的适配，非原论文完整架构复现。
