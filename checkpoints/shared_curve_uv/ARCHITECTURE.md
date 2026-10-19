# Shared curve + owner UV：架构快照

日期：2026-10-19。代码基准：`580bec89b378796c144476771cdce6c396cc1215`；本轮实现提交见本目录README。配置：`configs/train_vae_shared_curve_uv.yaml`。

这是已实现、未正式训练的实验。初始化采用CNN encoder的UV-boundary best，非UV attention encoder。保持半边输出，不把半边拓扑删除。改动只发生在refinement之后的最终曲线和UV路径。

## 1. 完整调用边界

F = batch中的总面数；每个对象有n个面；P = sum(n*(n-1))有向候选；K=P/2共享曲线候选。GT连接用于监督，不用于枚举推理候选。当前数据已经将同一面对的边界合并，且不包括seam；不能将此约定推广为任意B-Rep都有K条真实边。

```text
训练输入（原dataset / sidecar不变）
  face_norm [F,16,16,6]、edge_norm [E_gt,16,6]、bbox、connectivity
  edge_uv_targets.uv [R_gt,2,16,2]、valid/method/residual（仅loss读取）
       |
       v
  HolisticEncoder(CNN + 原GAT + attention/global)  [冻结]
       | face_features [F,32]
  FacePosterior / sample                         [冻结]
       | Z [F,32]    <------ 纯推理直接输入Z，不读取GT/UV标签
       v
  FaceFeatureDecoder（原8层768维TransformerEncoder）
       | F0 [F,32]
       +----------------------------------------------+
       | gather pairs [P,2,32]                       | 原FaceGeometry
       v                                              v
  AttnIntersection：Linear32->512 + side PE          S0 [F,16,16,3]
    tgt [P,1,512], memory [P,1,512]
    8 x TransformerDecoderLayer(512,8heads,FFN2048,
        norm_first=True, ReLU, dropout=.1)
    Linear512->64 -> squeeze
       | h0 [P,64]
       +-> 原CurveGeometry -> coarse C0 [P,16,3]
       +-> 原Linear classifier -> symmetrize -> p0 [P]
       |
       v
  原IntersectionRefiner(h0,F0,C0,S0,p0,pairs)
    几何读取：C0到两面三角网格的最近点
    point MLP -> mean/max -> relation MLP -> owner聚合
    delta_h [P,64]、delta_F [F,32]
       |
       +-> h1 = h0 + delta_h [P,64]   [有向特征保留]
       +-> F1 = F0 + delta_F [F,32]
       |       +-> 原FaceGeometry -> S1 [F,16,16,3]
       +-> 原classifier + symmetrize -> final logits [P]
       |
       v
  [新增] SharedCurveFusion -> g [K,64]
       |
  原CurveGeometry + 原FacePairConditioning，每个K只运行一次
       | local [K,16,6]、center/scale [K,4]、C [K,16,3]
       |
       +-> 两个反向视图 -> final half-edge XYZ [P,16,3]
       |                        |
       |                        +---------> 曲线重建/几何监督
       |                        |
       v                        v
  [调整] CurveConditionedUVPathHead(h1,g,F1,owner,C)
       | owner_uv [P,16,2]
       +-> stack(owner_uv, reciprocal_owner_uv.flip(t))
       | edge_uv [P,2,16,2]
       v
  UV loss + 固定GT-UV anchor loss（权重不变）
       | UV梯度经过C回到CurveGeometry、SharedCurveFusion、refiner

导出继续是 legacy_half_edges；面、半边、存在概率、两侧UV的字段不变。
```

未改动层的完整细节见[原UV实验架构](../uv_boundary/ARCHITECTURE.md)，其中encoder、posterior、原inter、refiner和原曲线head保持原参数层级。

## 2. 新增融合层：保留方向，不平均特征

```text
pairs[P,2] -> select i<j -> canonical[K] / reciprocal[K]
                              |                |
h1[P,64] -----------------> h_ij[K,64]      h_ji[K,64]
                              |                |
                              +---- cat -------+
                              |  [K,128]
                              |      |
                              |  nn.Linear(128,128)
                              |  nn.SiLU
                              |  nn.Linear(128,64)  [weight/bias初始为0]
                              |      | delta_g[K,64]
                              +------+
                                     | add
                                     v
                                  g[K,64]
```

公式：`g_ij = h_ij + MLP([h_ij,h_ji])`。初始g等于代表方向的已训练h，因此能沿用该方向的曲线预测；残差学起来后，两侧特征共同决定曲线。第一次反向时末层先获得梯度，之前层的梯度初始为零，这是零残差初始化的预期行为。

共享的是最终曲线，不是宣称h_ij=h_ji。代表方向按face索引i<j选择；任意重排face可能选中另一侧旧预测，**本版不保证face任意重排后的曲线严格相同**。两侧在同一次解码内共享几何则严格成立。直接平均有向特征的尝试已在初始化检查中排除，见PLAN。

## 3. 原曲线头：现在batch轴为K

```text
g[K,64]
  +-> reshape [K,32,2] -> ResnetBlock(32,768)
  |   ConvTranspose1d(768,384,k2,s2) -> [K,384,4] -> RB384
  |   ConvTranspose1d(384,192,k2,s2) -> [K,192,8] -> RB192
  |   ConvTranspose1d(192,96,k2,s2)  -> [K,96,16] -> RB96
  |   原condition affine -> Conv1d(96,6,k1) -> [K,16,6]
  |
  +-> 原bbox MLP：RB64->768、6 x RB768
  |   原condition affine -> Linear768->4 -> center/scale [K,4]
  |
  +-> cat(g,F1_i,F1_j) [K,128]
      原FacePairConditioning：Linear128->128,SiLU
        +-> Linear128->192  -> gamma/beta [K,96] -> 上述曲线CNN
        +-> Linear128->1536 -> gamma/beta [K,768] -> 上述bbox MLP

local[..., :3] * scale + center -> C[K,16,3]（FP32）
  +-> C_ij(t) = C(t)
  +-> C_ji(t) = C(1-t) [flip采样轴，不重新解码]
```

local的后三个通道是数据中的intrinsic D1方向，反序时只反转采样轴，不额外取负；它们不是另定义的coedge有向切线。bbox两侧复用。coarse和encoder-edge teacher仍走原有P/E_gt条解码路径，与final共享同一套CNN/bbox参数。

## 4. UV头：每个点实际读取预测XYZ

下面P中每行代表一个有向半边、owner=该行pairs的第一列。g通过inverse索引从K展开到P；C也按照该侧点序展开。

```text
h1[P,64]   g[P,64]   F1[owner][P,32]
    +----------+------------+
               | cat [P,160]
        nn.Linear(160,128) -> nn.SiLU
               | [P,128] -> unsqueeze -> [P,1,128]
               |
               |         t[16,1]                 C[P,16,3]
               |         Linear1->32             Linear3->128 [新增]
               |         SiLU                    SiLU
               |         Linear32->128           Linear128->128 [新增]
               |           | [16,128]                | [P,16,128]
               +-----------+------------------------+
                           | broadcast add
                           v
                      T0[P,16,128]
                           |
          +---------------------------------------+
          | 2 x nn.TransformerEncoderLayer       |  [原层，独立参数]
          | d_model128, nhead4, FFN256            |
          | norm_first=True, dropout0, GELU      |
          |                                       |
          | x -> LN128 -> SelfAttention Q/K/V128  |
          |          4 heads x 32 -> residual add |
          |   -> LN128 -> Linear128->256 -> GELU |
          |          -> Linear256->128 -> add     |
          +---------------------------------------+
                           | [P,16,128]
                      nn.LayerNorm(128)
                      nn.Linear(128,2)
                           | owner_uv[P,16,2]
                           v
    stack(owner_uv, owner_uv[reverse].flip(sample_axis), dim=1)
                           | edge_uv[P,2,16,2]
```

`T(t)=MLP_context([h,g,F_owner])+PE(t)+MLP_xyz(C(t))`。C来自模型预测，没有GT输入、没有detach。UV路径仍可在两面上不同；结构保证相同XYZ和同一采样对应，**并不硬保证**`S_i(U_i(t))=S_j(U_j(t))=C(t)`。当前仍用监督学习这些关系，未引入新的可微求交器/投影循环。

## 5. 标签、loss与权重迁移

- 数据加载器和sidecar均不变。仅final loss内部对每个正面对选i<j的GT连接行作为代表，反向行的XYZ/local反序；UV、valid、residual、method同时交换owner并反序采样。coarse/encoder teacher用原标签。
- 这避免闭曲线两个方向独立采样、起点相位不同时产生相互矛盾的逐点标签。保留代表行完整首尾分支，不滚动UV、不取模、不平均闭环；没有相位匹配搜索。
- 训练的UV anchor仍使用原GT-UV固定对应及原残差补偿，不是直接将预测UV采样所得位置强制设为C。
- 所有loss权重从旧配置继承，`uv_weight=.01`、`uv_anchor_weight=.1`。共享几何会改变twin项以及final目标，不能单凭总loss下降判定研究收益。
- 原UV模型770个state条目中767个逐位保留；3项映射：context权重从`[h,F_i,F_j]`变为`[h,g,F_owner]`，保留h/第一owner列，g列零初始化；输出Linear的前两个通道迁移到单owner输出。旧第二owner的context列和输出通道不迁移。原position、Transformer、LN均保留。
- 新增8个参数tensor：融合MLP4项、XYZ embedding4项；净增45,630参数（182,252,513 -> 182,298,143）。UV输出不是旧头的恒等迁移，训练需要适应；曲线代表方向的初始化被真实批量验证保留。
- `shared_curve.enabled=false`时保持原结构、旧state和RNG行为。新run用`init_shared_curve_checkpoint`；新模型的full load/resume必须启用相同结构，resume不重复旧权重迁移。

## 6. 解释边界

本实验解决互反半边独立XYZ预测可能产生的分歧，同时让UV读取共享曲线的实际位置。它不新增vertex/loop拓扑监督，不改classifier，不保证闭合solid，不直接解决每对面有多条独立边或seam的通用表示问题。收益必须用固定validation下游协议衡量，test不参与设计或挑选。
