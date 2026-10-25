# UV Intersection Decoder：架构讨论稿 v0.1

**当前训练协议：** [直接双侧 UV 第一版](TRAINING.md)，对应用户确认后的可读性实现及显式64D权重迁移。下文保留历史讨论。

**后续实现：** [单文件架构 v0.2](REVIEW_IMPLEMENTATION.md) 已完成前后向检查，未训练、未接 OCC。下文保留 v0.1 讨论历史，不作为最新实现说明。

日期：2026-10-25（北京时间）。状态：**神经网络信息流建议稿，未实现、未训练；连续曲面重建与共享3D边构造的数值规则尚未冻结。**

本稿回应用户对面去重、全局信息交换、双侧UV和共享曲线依赖关系的讨论。它不是已批准的训练配置，也不以草案追认旧实验。

**2026-10-25 后续讨论：本版暂不作为可冻结实现的架构。** 用户指出 `Fusion(h_ij,h_ji)` 再读出双侧UV只共享了隐藏特征，没有保证共享几何。这一质疑成立：旧单条XYZ曲线保证两个coedge引用同一组点，双侧UV则保证两条轨迹分别位于各自面上，两者并不等价。以下v0.1保留为讨论快照，不能把其中`JointUVHead`视为已接受方案。

本次重新讨论的方向是每个无向边候选维护沿共同`t`展开的点状态，直接读取两面Decoder的空间特征、预测XYZ及参数位置，再读出两侧UV。它改变信息读取方式，但仍不保证`S_i(U_i)=S_j(U_j)`；唯一共享几何的构造继续未决。层数、宽度、训练范围和是否需要后续跨边交换均未冻结。

**随后确认的读取方式：** 用户选择两面空间token沿序列维拼接：各`[256,d]`，加入UV位置及所属面编码后形成`M[512,d]`；共同曲线位置query为`Q[16,d]`，经TransformerDecoder读取M，输出`H[16,d]`，由`Linear(d,4)`读出`[16,4]`，再按两侧组织为`[2,16,2]`。这取代上次讨论草图中的双路分别cross-attention及消息合并。宽度和层数未定，未实现。面的采样点重建监督保留；连续曲面求值仅用于取得预测UV处的XYZ，不要求训练时调用OCC或生成密集曲面。固定B-Spline只是求值候选，耗时尚未测量。

历史[local intersection评估](../intersection_local/evaluation_20261013/NOTE.md)已经检验过“有向面对单query读取两面512个空间token→单个边特征→独立XYZ头”：完整8层版合法solid为1705/2417，对照2113/2417。该历史test结果只用于恢复已有证据，不据此开发新规则；它并非等预算、仅改变局部读取的因果消融。不能把重访空间读取称为首次尝试，也不能声称新的逐点状态一定有效或更快。

“固定可微B-Spline”在本稿中只指待设计的确定性曲面转换：固定采样参数、次数、结点及控制点布局后，采样XYZ到控制点可用固定线性映射表达；曲面随预测XYZ变化。它统一训练与导出的几何，不自动保证边面一致；周期规则、参数域、拟合精度仍需论证。当前后处理的自适应OCC拟合不是已经具备这套训练内接口。

## 来源与范围

- 核查工作树：`/mnt/d/python_curve_eval`，源码HEAD `158bf91b317e9bf61aa4163220258386ec224c96`，工作树含此前未提交工作。
- 当前事实参考：`configs/train_vae_latent_expansion_no_edge_feature.yaml`及`checkpoints/latent_expansion/no_edge_feature_20261023/resolved_config.yaml`。
- 参考已训练模型：64维、30k步、关闭edge_feature；checkpoint SHA256 `4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4`。本稿没有重新加载或修改该权重。
- 新实现Git hash / config入口：**尚无**。下面的`JointUVHead`和`SurfaceReconstructor`是拟议职责名称，不表示源码中已经存在。
- 本轮只改解码表示与信息流；保留现有面/边/连接的encoder输入和每面latent合同，不新增loop/vertex预测头。
- 为固定latent坐标系，首版建议冻结同源64维encoder与posterior，训练decoder。该训练范围是本稿建议，尚未启动运行。

## 张量与索引

单个对象记号：`N`为输入面latent数，`D=64`，`P=N(N-1)`为有向非self候选数，`Q=P/2`为无序候选数，`E<=Q`为推理分类后保留的物理边数。批量实现保持packed face和对象mask；不能跨对象配对。

每个物理候选对应一个`g[q]`和两侧UV。`i<j`用于复用现有canonical配对索引，不表示输入GT顺序。共同曲线参数`t`具有相同空间方向；导出反向coedge视图时才反转`t`。闭合曲线的起点及周期接缝分支需和监督约定一致。

`i<j`不是网络严格对称的证明：现有共享融合有方向性，新UV头的面输入及四个输出也有顺序。首版沿用该约定，不承诺任意面排列下结果严格仅交换UV侧或反转`t`；这也不是已经证明的训练失败原因。

继续沿用当前受支持数据范围：每对不同面至多一条物理公共边；不含self-face/seam和同一面对的多重边。不能将此模型称为支持任意B-Rep拓扑。

## 总体信息流

```text
VAE reconstruction                         Generated latents (separate task)
CAD faces + edges + connectivity            DDPM [30,D]
    |                                         |
same encoder + posterior (freeze proposed)   latent duplicate removal
    |                                         |  D must match VAE
    +------------ z [N,64] -------------------+
                         |
                 Linear 64 -> 768
                         |
              8 x TransformerEncoderLayer
              width 768, heads 16, FFN 2048
              pre-LN, ReLU, dropout 0.1
              same-object self-attention
                         |
                 Linear 768 -> 64
                         |
                      F [N,64]
                    /          \
                   /            \
     retained FaceGeometry       enumerate directed face pairs
      CNN + center/scale           [P,2,64]
               |                     |
      XYZ [N,16,16,3]          retained AttnIntersection
               |                     |
     SurfaceReconstructor         h [P,128]
     (deterministic,                /     \
      differentiable)       classifier   SharedCurveFusion
               |              |              |
           surfaces S_f      p [Q]        g [Q,128]
               |              |              |
               |              |       JointUVHead <- F_i,F_j,t
               |              |              |
               |              |       UV [Q,2,16,2]
               |              |              |
               +--------------|------> geometry state: (S, UV, owners)
                              |              |          |
                              |              |          +--> evaluate at 16 UV samples
                              |              |               C_i,C_j [Q,2,16,3]
                              |              |                 -> discrepancy -> loss
                              +------> select state for predicted edges
                                             |
                              state's UV samples -> continuous 2D curves
                              (deterministic rule not yet frozen)
                                             |
                              state's S evaluated along those curves
                                             |
                              shared 3D curve construction
                              (geometry algorithm not yet frozen)
                                             |
                              same S + pcurves + one 3D curve
                                             |
                              OCC loops/trimming/sewing/checks
```

示意中UV用`Q`标维度便于理解：训练对GT正候选计算UV/几何监督；推理可在分类后仅为预测正候选展开UV。分类始终覆盖所有候选。GT不是推理输入。

`paired discrepancy`是空间曲线两侧读数的训练分支，后续构造仍读取面、UV及其曲线，不是只读取一个残差向量。16点求值与连续路径求值分开；训练如使用更密的`t`，也必须经过同一个连续UV构造器。

最终3D边由曲面与UV派生。禁止在该节点恢复一个独立预测XYZ/bbox的神经头。两侧轨迹平均可以作为几何初始化候选，但**未被选定为无条件最终输出**。

## 保留的全局模块

源码：`src/brepnet/models/vae/decoders/primitives.py:FaceFeatureDecoder`，旧权重定义在`models/vae/legacy.py`。

```text
z [N,64]
 -> Linear(64,768)                                 [N,768]
 -> 8 x TransformerEncoderLayer                   [N,768]
      LayerNorm -> SelfAttention(16 heads) -> residual
      LayerNorm -> Linear768/2048 -> ReLU
                -> Linear2048/768 -> residual
 -> Linear(768,64)                                [N,64]
```

Transformer内部有残差，模块外没有额外`z+F`。它在同一对象所有面之间交换信息；不混合不同对象。没有新全局attention层，也不是只保留一层。

## 保留的面几何头

源码：`models/vae/legacy.py`和`decoders/primitives.py:PrimitiveDecoder`。

```text
F [N,64]
 |
 +-> reshape                                  [N,16,2,2]
 |   ResBlock2D 16->768                        [N,768,2,2]
 |   ConvTranspose2D 768->384, k2,s2            [N,384,4,4]
 |   ResBlock2D 384->384                       [N,384,4,4]
 |   ConvTranspose2D 384->192, k2,s2            [N,192,8,8]
 |   ResBlock2D 192->192                       [N,192,8,8]
 |   ConvTranspose2D 192->96, k2,s2             [N,96,16,16]
 |   ResBlock2D 96->96                         [N,96,16,16]
 |   Conv2D 96->6, k1                          [N,6,16,16]
 |   permute                                  [N,16,16,6]
 |                                             local XYZ + normal
 |
 +-> ResLinearBlock 64->768                    [N,768]
     6 x ResLinearBlock 768->768               [N,768]
     Linear768->4                             [N,4]
                                               center XYZ + scalar scale

global XYZ = local XYZ * scale + center        [N,16,16,3]
```

每个残差块含两次Conv/Linear及LayerNorm，中间ReLU；identity或投影skip相加后ReLU。保留6通道兼容现有面头；normal不会作为空间位置传给曲面拟合。面bbox仍是7个残差全连接块，不是两层小MLP。

### 连续曲面接口（待冻结数值规则）

输入仍是规则采样点，不将其直接冒充控制点。`SurfaceReconstructor`从这些点确定一套连续B-Spline及参数域；训练求值和导出OCC使用同一套参数。可采用固定基函数下的可微插值/拟合，不能在训练用双线性网格、导出再独立调用另一种自适应拟合。

尚需决定：次数、结点/控制点预算、正则项、周期处理、UV轴顺序、原UV标签到新参数域的对应。这些没有敲定前不能声称完整几何架构已可实现或旧UV标签必然可直接用。

## UV Intersection Module

### 面对特征提取：保留原Inter

```text
F [N,64]
 -> gather ordered i!=j                        [P,2,64]
 -> Linear64->512 + learned side embedding     [P,2,512]
 -> split: target=first face                   [P,1,512]
           memory=second face                 [P,1,512]
 -> 8 x TransformerDecoderLayer
      width512, heads8, FFN2048, pre-LN
      ReLU, dropout0.1
 -> Linear512->128, squeeze                    h [P,128]
```

这里每个candidate的target和memory都只有一个token；不能画成对256个面采样点做cross-attention，也不在不同candidate之间做attention。

### 分类与共享特征：两个并行输出

```text
h_ij,h_ji
  |
  +-> Linear128->1 on each directed h
  |   mean(logit_ij,logit_ji) -> sigmoid        p [Q]
  |
  +-> concat(h_ij,h_ji)                        [Q,256]
      Linear256->128 -> SiLU -> Linear128->128 [Q,128]
      residual + h_ij                         g [Q,128]
```

分类复用已有classifier，shared融合复用`SharedCurveFusion`。共享的是边特征/身份，不是已经一致的XYZ曲线。

### 新联合双侧UV头

```text
concat(g128,F_i64,F_j64)                       [Q,256]
 -> Linear256->128 -> SiLU                    [Q,128]
 -> broadcast to 16 parameter positions       [Q,16,128]
                                                    +
t=linspace(0,1,16) [16,1]
 -> Linear1->32 -> SiLU -> Linear32->128       [16,128]
                                                    |
 -> 2 x TransformerEncoderLayer               [Q,16,128]
      width128, heads4, FFN256, pre-LN
      GELU, dropout0
 -> LayerNorm128 -> Linear128->4              [Q,16,4]
 -> reshape/permute                           [Q,2,16,2]
                                               (u_i,v_i), (u_j,v_j)
```

此处self-attention沿16个曲线参数位置运行，两个UV轨迹由同一组token联合输出。当前`CurveConditionedUVPathHead`输入含已预测XYZ；新头明确去掉这一依赖，否则删除3D边CNN后前向图不成立。

旧UV位置MLP和两层Transformer的形状可复用；context与输出层形状/语义改变，需要显式初始化或迁移，不能整头strict加载。原`UVPathHead`的双侧输出设计可参考，但不代表本新配置已经实现。

移除refiner会改变保留模块收到的特征。旧权重在此是初始化复用，不保证新decoder初始输出等于旧decoder输出。

## 几何构造及监督

```text
Ui(t) -- evaluate S_i --> Ci(t) --+
                                +--> paired discrepancy Ci(t)-Cj(t)
Uj(t) -- evaluate S_j --> Cj(t) --+
                                |
                                +--> deterministic shared-curve construction
```

`Ci`严格位于模型定义的`S_i`上，`Cj`位于`S_j`上（数值求值精度内）；这不保证`Ci=Cj`，也不保证GT保真。共享特征和共同`t`不会自动消除形状差异、周期分支歧义或闭合起点问题。

上式的连续`Ui(t)`还需要从16个网络UV样本构造。其插值/拟合次数、结点、参数域及周期处理尚未冻结；不能将离散UV数组直接当作连续pcurve。中间段越界和高曲率区偏离要由这一接口处理和检查，不能靠16点残差小推断整段成立。

损失职责建议（权重尚未冻结）：

- face：约束保留的面输出及最终连续曲面几何；需要区分原始采样点误差和转换后的几何误差。
- classification：对所有candidate监督有边/无边，不把GT邻接作为推理输入。
- UV：仅GT正pair；在已核对的曲面chart中监督双侧UV。
- curve fidelity：分别约束`Ci`、`Cj`与同一GT物理边，不能只监督两者平均，避免两边误差相互抵消。
- paired consistency：同一`t`下两侧XYZ差；不在不同chart中直接相减UV。
- 评估整条曲线和未参与训练的参数位置；16点残差不能充当连续误差上界。

不沿用已删除输出对应的teacher-edge reconstruction、coarse edge XYZ/bbox、edge feature matching、旧refiner incidence路径。是否保留面local/normal/bbox辅助项及其权重留到损失方案明确时冻结；冻结encoder的KL只能作为监控，不更新参数。

最终3D曲线构造可考虑对两侧轨迹作有误差控制的共同拟合；若轨迹差异过大，应显式处理不一致，不能靠平均、无限放宽容差或另一个自由XYZ网络隐藏问题。OCC环路/裁剪仍有独立失败机制，本稿不承诺合法率。

## 去重与数量

```text
Reconstruction:
known CAD N faces -> N face latents -> global attention -> N predicted faces
                    no duplicate padding / no face pruning

Generation (current random-padding contract):
DDPM [30,D] -> face mask -> latent greedy dedup -> [N,D] -> VAE decoder
                           mean L1 < 1e-2
```

当前`AutoEncoder_intersection`的`decode_latents`不再额外做面几何去重。旧`AutoEncoder_light.decode_latents`曾在解码面之后用对应采样点平均欧氏距离`<1e-3`做几何去重；不能把旧流程当作当前重建路径。

重建不去重是因为输入已有面数/逐面对应，不是模型数学上不会产生重复面。不同裁剪区域可以共享同一承载曲面，几何相似不充分证明它们是重复B-Rep face。

当前主diffusion默认32维，本草案64维；不得直接声称已有32维diffusion或cache可接入此decoder。即便维度相同，也必须核对encoder/posterior权重、输入归一化和cache同源。只改decoder不改变冻结encoder的latent数值，但会改变这些latent解码出的几何，需要重新评估。

## 明确移除的旧路径

- 独立3D边CNN和边center/scale头；旧curve conditioning随其退出。
- 初稿边几何解码、旧几何最近三角搜索refinement及第二轮几何解码。
- 当前依赖XYZ的逐侧UV头，换为上述共同`t`的联合UV头。

不是只设置`refinement.enabled=false`：当前shared_curve分支即使关闭refiner仍执行coarse与shared两次几何；新实现应在装配时移除不用的路径。

## 下一次敲定范围

本稿可用于确认神经网络信息流与复用范围。尚未冻结的三项关键接口是：

1. 16x16采样点到统一连续曲面的具体可微规则及UV标签映射。
2. 16个UV样本到连续2D参数曲线的构造、域约束和周期处理。
3. 双侧面上轨迹到唯一3D曲线的构造、容差和失败处理。

没有新增训练run、没有改动生产后处理，也没有因本稿创建实验结果。
