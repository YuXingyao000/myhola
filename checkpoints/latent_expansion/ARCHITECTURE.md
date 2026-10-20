# Latent64：整体架构快照

日期：2026-10-20。实现提交：`4f7b2df78f5ad22313dca0ea9e732db53ba4344e`。
本文件的提交hash/验收状态在实现后的文档提交中补齐，网络代码未变。
代码基准：`85f6fb33da269fd75d0f7352164ce1af3663be27`。
配置：`configs/train_vae_latent_expansion.yaml`；
32维解冻对照：`configs/train_vae_latent_expansion_control.yaml`。
状态：实现及本机/142验证完成，未正式训练，不代表几何或合法率改善。

## 输入、输出与模块

F=batch中真实面总数；R=GT有向边数；P=sum(n*(n-1))有向候选；
K=P/2共享曲线候选；D=64，旧D=32。训练不使用random repeat。
数据与归一化不变，face_points包含XYZ和法向，edge_points包含XYZ和D1方向，
归一化local几何和center/scale另用于loss。最终XYZ保持归一化CAD坐标。

```text
训练数据
 face_points[F,16,16,6]      edge_points[R,16,6]
 connectivity[R,3]         face_counts[B] + object attention mask
 local geometry / bbox     GT UV[R,2,16,2] + valid[R,2,16] (仅监督)
             |                         |
             v                         v
  HolisticEncoder.face_coords    HolisticEncoder.edge_coords
  Conv2d6->96 + LN + LeakyReLU   Conv1d6->96 + LN + LeakyReLU
  ResBlock96->192 @16x16        ResBlock96->192 @16
  MaxPool2d -> 8x8              MaxPool1d -> 8
  ResBlock192->384              ResBlock192->384
  MaxPool2d -> 4x4              MaxPool1d -> 4
  ResBlock384->768              ResBlock384->768
  MaxPool2d -> 2x2              MaxPool1d -> 2
  ResBlock768->768              ResBlock768->768
  Conv2d768->16 (旧8)           Conv1d768->64 (旧32)
  Flatten C,H,W                 Flatten C,L
             | [F,64]                 | [R,128]
             v                        | edge teacher -> 训练用原曲线头/loss
  Linear64->768                       |
  8 x TransformerEncoderLayer        |
    768 /16heads /FFN2048             |
    pre-LN /ReLU /dropout.1           |
  final LN768 -> Linear768->64        |
  + 原face features                   |
             |                        |
             v                        v
  5 x GATv2Conv(64,64, heads1, edge_dim128)
    connectivity给图的连接；edge只影响attention分数
    原残差 + LeakyReLU + 外层face残差
             | [F,64]
  scatter_mean按对象 -> global_feature1: Linear64->64 -> LeakyReLU -> Linear64->64
    全局context广播到所属face
  cat(face,context)[F,128] -> global_feature2: Linear128->64 -> LeakyReLU -> Linear64->64
  + face
             | [F,64]
             v
  FacePosterior
  Linear64->128 -> LeakyReLU -> Linear128->128
  reshape [F,64,2] -> mean/logvar
    train: z = mean + exp(.5*logvar)*randn -> [F,64]
    posterior_mean / v_test=True: z = mean -> [F,64]
    cache: cat(mean,std)                  -> [F,128]
             |
             +<------- 纯解码接口只输入z及可选face_counts
             v
  FaceFeatureDecoder
  Linear64->768 -> 8 x TransformerEncoderLayer(768,16heads,FFN2048)
  Linear768->64
             | F0 [F,64]
             +-------------------------------+
             | gather pairs [P,2,64]         | FaceGeometry
             v                               v
  AttnIntersection                         S0 [F,16,16,3]
  Linear64->512 + learned side PE[1,2,512]
    Q [P,1,512] / memory [P,1,512]
    8 x TransformerDecoderLayer(512,8heads,FFN2048)
    pre-LN /ReLU /dropout.1
  Linear512->128 -> squeeze
             | h0 [P,128]
             +-> Linear128->1 -> reciprocal logits均值 -> sigmoid -> p0[P]
             +-> 原CurveGeometry -> C0[P,16,3]
             |
             v
  IntersectionRefiner (下节展开)
    (F0,h0,S0,C0,p0,pairs) -> dF[F,64], dh[P,128]
             |
    F1=F0+dF | h1=h0+dh
      |      +-> Linear128->1 + 双向均值 -> final logits[P]
      |      |
      |      +-> canonical i<j -> (h_ij,h_ji)[K,128]各一份
      |           cat[K,256] -> Linear256->128 -> SiLU -> Linear128->128
      |           + h_ij -> g[K,128]  (SharedCurveFusion)
      |                                |
      +-> FaceGeometry -> S1           +-> CurveGeometry + conditioning
                                         C[K,16,3]
                                            |
                     canonical/reciprocal flip(t) -> C1[P,16,3]
                                            |
  owner UV context=cat(h1,g_expanded,F1_owner)[P,320]
    Linear320->128 -> SiLU -> [P,1,128]
    + PE(t): Linear1->32 -> SiLU -> Linear32->128
    + XYZ embedding: Linear3->128 -> SiLU -> Linear128->128
    -> tokens[P,16,128]
    -> 2 x TransformerEncoderLayer(128,4heads,FFN256,pre-LN,GELU,dropout0)
    -> LN128 -> Linear128->2 -> owner_uv[P,16,2]
    -> stack(本侧, 互反侧.flip(t)) -> UV[P,2,16,2]

完整训练：Encoder / posterior / Decoder全部解冻
导出：face XYZ、legacy_half_edges XYZ/连接/概率、UV；字段/后处理不变
```

## Refiner、几何头和loss

```text
GeometryRead: C0和两侧S0 -> 每点17维观察[P,16,17]
 Linear17->128 -> LN -> SiLU -> Linear128->64 -> SiLU
 mean/max pool cat128 -> Linear128->64 -> G[P,64]

message: cat(h0,G)[P,192] -> Linear192->128 -> LN -> SiLU
 概率加权汇聚到两侧face -> context[F,128]
relation: cat(h0,G,F_i,F_j,context_i,context_j)[P,576]
 -> Linear576->128 -> LN -> SiLU -> Linear128->128 -> SiLU
 -> Linear128->128 -> dh[P,128]
face: 概率加权relation汇聚；cat(F0,汇聚)[F,192]
 -> Linear192->128 -> LN -> SiLU -> Linear128->64 -> dF[F,64]

FaceGeometry: F1[F,64] -> reshape[F,16,2,2] -> ResBlock16->768
 ConvTranspose2d768->384 -> RB384 (4x4)
 ConvTranspose2d384->192 -> RB192 (8x8)
 ConvTranspose2d192->96  -> RB96  (16x16)
 Conv2d96->6 -> local[F,16,16,6]
 bbox: ResBlock64->768 + 6 x ResBlock768 -> Linear768->4
 XYZ = local_xyz * scale + center

CurveGeometry: g[K,128] -> reshape[K,64,2] -> ResBlock64->768
 ConvTranspose1d768->384 -> RB384 (length4)
 ConvTranspose1d384->192 -> RB192 (length8)
 ConvTranspose1d192->96  -> RB96  (length16)
 conditioning affine -> Conv1d96->6 -> local[K,16,6]
 bbox: ResBlock128->768 + 6 x ResBlock768
 conditioning affine -> Linear768->4
 XYZ = local_xyz * scale + center
 conditioning: cat(g,F_i,F_j)[K,256] -> Linear256->128 -> SiLU
   -> Linear128->192  (CNN末层前的gamma/beta,96维)
   -> Linear128->1536 (bbox末层前的gamma/beta,768维)

loss:
 coarse/final local XYZ+方向（face法向、edge D1）、center/scale、classifier
 GT edge CNN feature teacher (detach) 与训练用teacher geometry
 final global XYZ、incidence、twin
 UV L1 + 固定GT UV位置上的edge/surface anchor
 KL sum: D64用5e-7；D32对照用1e-6
```

ResBlock内部沿用既有`ResBlockXD`：conv/linear、LayerNorm、激活、投影残差；
本轮不改内部算子。具体模块分别在 `models/vae/encoders.py`、
`decoders/{intersection,refinement,primitives}.py`、`legacy.py` 与
`losses/intersection.py`；主装配仍为 `AutoEncoder_intersection`。

## 权重迁移与边界

迁移工具位于已有 `models/vae/checkpoints.py`，不参与forward。
81个tensor按维度和语义段扩展，697个tensor直接继承。
参数182,298,143→182,982,439，增加684,296（约0.375%）。
这主要扩大表示瓶颈，不是将整个网络参数翻倍。

输出复制，输入读取按0.49/0.51分摊；两者和为1，同时避免复制通道
始终得到相同梯度。拼接向量必须分别展开E、F_i、F_j等片段。
GAT attention向量复制后除2；posterior按交错mean/logvar序列复制。
mean推理初始近等价，随机采样分布不等价，详见PLAN。

旧32维latent cache和diffusion不能直接用于本模型。新checkpoint严格按
64维配置加载；resume恢复本实验optimizer/step并清空组件初始化源。
候选结构、半边方向、UV有效点筛选、GT只用于训练监督的约定不变。
不保证边面严格相交或闭环，也不把容量扩大称为已解决trimming。
