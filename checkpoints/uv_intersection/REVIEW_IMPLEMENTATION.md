# UV Intersection：单文件实现 v0.2

## 当前可读性修订（2026-10-25）

用户审阅后指出 query 后的网络被候选分块循环遮住。当前实现将 `FaceSpatialTokens`
提升到总 decoder，候选分块移到总 decoder 的 `_decode_candidates`，
`UVIntersection.forward` 只保留六行网络计算。
新增 `CurveAttentionStack` 命名并容纳原来的八个 `CurveAttentionBlock`，没有增加可训练层。
`DualUVHead` 是一套 `LayerNorm + Linear(512,4) + sigmoid`，四个数为 `(u_i,v_i,u_j,v_j)`。

```text
FaceSpatialTokens (outside candidate chunks)
    spatial + XYZ -> face_tokens[N,256,512]
                              |
UVIntersection.forward        v
    PairMemory(face_tokens, owners) -> memory[p,512,512] ----+
                                                           |
    CurveQueries(F, owners) -> edge_query[p,16,512]          |
                                  |                        |
                                  v                        v
                         CurveAttentionStack(query, memory)
                           8 x CurveAttentionBlock
                                  |
                         edge_features[p,16,512]
                              /           \
                        DualUVHead    EdgeExistenceHead
                       [p,2,16,2]         [p]
```

新代码 SHA256：`1fddc31b42dea56daf586ed798d176bdbf929cfea146d068c4a2ef8abb0140cb`。
当前为 24 个 `nn.Module` 加 1 个配置 dataclass，最长 class 41 行。
参数名称随可读性整理改变；此版本尚无已训练 checkpoint，未添加静默兼容加载。
历史完整尺寸检查保留在 [VALIDATION.json](VALIDATION.json)，本次等价性证据另记在其
`readability_revision` 字段；不把历史源码 hash 改写为本次版本。

以下保留初版结构快照；其中 `FaceSpatialTokens` 的容器归属、属性名和 class 数以本节修订为准。

## v0.2 初版快照

日期：2026-10-25 UTC。状态：**实现及前后向检查完成，未训练、未接 OCC。**

用户要求先直接审阅代码。因此所有自定义网络层、曲面求值和 loss 都放在
`src/brepnet/models/vae/uv_intersection.py`，只依赖 PyTorch 和标准库。
总入口 `UVIntersectionDecoder.forward(latents, face_counts)` 位于文件开头；下方逐个展开小 module。
原有 encoder、trainer 和后处理未改。v0.1 留作讨论历史，本文件描述这次实际实现。

## 版本与默认配置

- 工作树：`/mnt/d/python_curve_eval`，分支 `codex/vae-uv-boundary`。
- 基础 Git HEAD：`158bf91b317e9bf61aa4163220258386ec224c96`。新文件尚未提交；不是该 commit 已包含的实现。
- 实现 SHA256：`d17ce9b048f8d57c9b235b77de22e0b9e2dd6588ead7ad7af264759b01d11c9f`。
- 配置入口是同文件 `UVDecoderConfig`，尚无 Hydra / trainer 注册。
- face：latent 64、width 768、8 层、16 heads、FFN 2048。
- curve：width 512、8 层、8 heads、FFN 2048；保留历史 intersection 的容量作为审阅默认值。
- dropout 0.1；每次处理 16 个面对。分块不会释放训练中已保留的各块反向计算图。
- 23 个 `nn.Module` 加 1 个配置 dataclass；最长 class 40 行；默认可训练参数 96,774,703。
- 首次检查全部 decoder 参数随机初始化，无 optimizer 更新。正式冻结范围、训练预算及 loss 权重仍需制定；没有把讨论中的 decoder-only 建议写成已开始的训练。

## 输入与总结构

`N`：batch 内输入面 token 总数；`P=sum_b n_b(n_b-1)/2`：物体内无序候选数。
`face_counts[B]` 仅描述输入 token 分组，不是推理额外获取 GT 面数的接口。
重建时输入来自 encoder；生成时需要上游先处理 padding / latent 去重。本 decoder 不做面几何去重。

```text
latents [N,64]                    face_counts [B]
       |                               |
       |                         object_ids [N]
       |                               |
       +--> GlobalFaceAttention <------+ same-object attention mask [N,N]
              Linear 64 -> 768
              8 x TransformerEncoderLayer
                pre-LN self-attention, 16 heads
                pre-LN FFN 768 -> 2048 -> 768, ReLU
              Linear 768 -> 64
                       |
                    F [N,64]
                       |
         +-------------+--------------------------+
         |                                        |
      FaceDecoder                           CurveQueries
         |                                (see local path below)
         +--> local XYZ/normal [N,16,16,6]         |
         +--> center/scale [N,4]                   |
         +--> spatial CNN [N,96,16,16]             |
         +--> global XYZ X [N,16,16,3]             |
                  |                |               |
                  |                +------> UVIntersection
                  |                           ^    |
                  |    object_ids -> i<j pairs[P,2] |
                  |                                +--> existence logits [P]
                  |                                +--> UV [P,2,16,2]
                  v                                          |
       FixedBSplineSurface.fit                               |
       poles [N,16,16,3]                                     |
                  |                                          |
                  +--> gather owner poles [P,2,16,16,3] ------+
                                     |
                           surface.evaluate(poles, UV)
                                     |
                          two trajectories [P,2,16,3]
```

## 面分支与局部读取

面分支复制旧 Light 的全网格 LayerNorm 残差块及 center/scale MLP。
global attention 无新增末尾 LayerNorm 或外部残差；center/scale 保留旧标量定义，没有改成 softplus。

```text
FacePointDecoder:
F[N,64] -> reshape[N,16,2,2]
        -> ResidualGrid 16 -> 768, 2x2
        -> ConvTranspose2d 768 -> 384, k2/s2 -> ResidualGrid, 4x4
        -> ConvTranspose2d 384 -> 192, k2/s2 -> ResidualGrid, 8x8
        -> ConvTranspose2d 192 -> 96,  k2/s2 -> ResidualGrid, 16x16
                                                    |
                         spatial[N,96,16,16] <------+
                                                    |
                                       Conv2d 96 -> 6, k1
                                       permute -> local[N,16,16,6]

FacePlacementHead:
F[N,64] -> ResidualLinear 64 -> 768
        -> 6 x ResidualLinear 768 -> 768 -> Linear 768 -> 4
        -> center[N,3], scale[N,1]
global XYZ = local XYZ * scale + center

FaceSpatialTokens:
spatial[N,96,16,16] -> flatten -> [N,256,96] --+
XYZ[N,16,16,3]     -> flatten -> [N,256,3] ---+ concat channels [N,256,99]
                                            |
                                  Linear 99 -> 512 + LN
                                            |
UV grid[256,2] -> MLP 2 -> 128 -> 512 ------+ add
                                            |
                                      tokens[N,256,512]
                                            |
PairMemory: gather pairs -> [P,2,256,512] + side embedding[2,1,512]
                         -> flatten face/token axes -> M[P,512,512]

CurveQueries:
Fi,Fj [P,64] -> concat(Fi+Fj, abs(Fi-Fj)) [P,128]
             -> MLP 128 -> 512 -> 512 -> seed[P,512]
t[16,1]      -> MLP 1 -> 32 -> 512 ----------------+ broadcast/add seed
                                                  |
                                              Q[P,16,512]
                                                  |
8 x CurveAttentionBlock:                          |
    LN(Q) -> self-attention(Q,Q,Q) -> residual ----+
    LN(Q) -> cross-attention(Q,M,M) -> residual ---+
    LN(Q) -> Linear512/2048 -> ReLU/dropout -> Linear2048/512 -> residual
                                                  |
                                       H[P,16,512]
                                          /            \
       DualUVHead: LN + Linear512/4 + sigmoid        mean over 16
                      [P,16,4]                      LN + Linear512/1
                 (u_i,v_i,u_j,v_j)                   logits[P]
                      |
             reshape/permute -> UV[P,2,16,2]
```

没有分别生成 `h_ij/h_ji`；一组沿共同 t 的 query 读两个面。
对称 seed 不等于整体排列等变：side embedding 与四通道输出仍区分 i/j。
候选是每个对象所有 i<j 面对，GT 不筛选 decoder 输入；一个面对最多一个物理边，不支持 self/seam 边和多重公共边。

## 可微曲面与 loss

网格存储顺序是 `[v,u]`，UV 向量是 `(u,v)`。
`FixedBSplineSurface` 用固定非周期三次、open-uniform clamped knots、16×16 控制点，
对归一化均匀 UV 处的 16×16 XYZ 网格作插值。预计算 `A=B^-1` 后，每面做一次 `P=A X A^T`；
再仅计算边请求的 UV 位置。这里没有 OCC、迭代优化或稠密曲面采样。

几何误差可以通过求值同时回传 UV 和面采样点。BF16 autocast 下 UV 输出与曲面运算保留 FP32。
插值不同于旧 OCC 自适应拟合，可能在采样点之间过冲；不能据此声称精度更好或训练更快。
不保证周期接缝连续、边闭合、两侧轨迹完全相同。sigmoid 输出域为 [0,1] 内部，尚未加入周期展开规则。

监督 target 显式提供 `face_local[N,16,16,6]`、`face_center_scale[N,4]`、
`face_xyz[N,16,16,3]`、唯一 `edge_pairs[R,2]` 及共同点序 `edge_xyz[R,16,3]`。

| loss | 当前实现 |
|---|---|
| face_local | 预测与 GT 局部 XYZ/normal 的 L1 |
| face_center_scale | 预测与 GT center/scale 的 L1 |
| face_xyz | 预测与 GT 全局面 XYZ 的 L1 |
| edge_existence | 全部候选 BCE，正负组分别平均再等权平均 |
| curve_xyz | 仅 GT 真边：两侧面上 XYZ 分别对 GT 的 MSE；两侧共同选择正序或反序 |
| curve_incidence | 仅 GT 真边：两侧同 t 的 XYZ MSE |

审阅默认总和为三项 face + 0.1 existence + curve_xyz + 0.1 incidence。
没有先平均两侧轨迹再监督，防止相反误差抵消；没有独立为两侧挑 GT 方向。
没有复用旧 UV 标签，新插值曲面参数域不能假定与旧拟合曲面完全相同。
这不是完整 VAE loss：encoder/posterior 与 KL 尚未接入。

## 已验证与未完成

证据：[VALIDATION.json](VALIDATION.json)。15 项测试通过，包括 CUDA BF16、
非对称平面轴顺序、端点导数、SciPy 数值对照、双精度 gradcheck、物体隔离、
正负候选梯度、共同反向、相反误差不可抵消、空边和非法 owner 回归。

完整默认网络用归档 posterior mean `[7,64]` 与同源 GT 做过一次前向/loss/反向：
21 个候选、10 条 GT 真边，UV `[21,2,16,2]`、XYZ `[21,2,16,3]`；所有参数梯度有限，各模块梯度非零。
输入来自 `00000178` 的既有归档，原始数据 SHA256 已核对；检查只确认接口与梯度，不分析个例质量、不据此调规则。
随机 decoder 未加载旧 checkpoint、未执行 optimizer step。没有训练收益或 OCC 合法率结论。
源代码内 residual Linear/Conv2D 与旧实现四种同/异维配置的随机权重数值对照最大差为 0。

后续实际接训练需要显式映射旧全局/面分支权重，新增模块单独初始化；当前参数名不同，不能直接 strict load 旧 decoder。
不使用 permissive loading 掩盖缺失。encoder 输入形状保留不代表任意旧 cache 都兼容，仍须同源 encoder/posterior。
唯一共享 3D 边的构造、同曲面参数域的 OCC 导出、闭合/周期路径和正式训练协议仍未实现。
