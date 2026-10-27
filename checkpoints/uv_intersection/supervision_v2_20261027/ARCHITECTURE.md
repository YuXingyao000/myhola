# 64D UV VAE：监督第二阶段快照

配置：`configs/train_vae_uv_intersection_supervision_v2.yaml`。网络实现：`src/brepnet/models/vae/uv_intersection.py`；训练装配：`uv_intersection_training.py`。实现commit由启动记录填写；本文件不是训练效果证明。

参数量170,699,647，全部可训练；与30k源模型相同，新增loss无参数。Encoder输入仍为面/半边六通道采样与连接；面后3通道为法向，边后3通道为旧切向。新切向监督不使用旧切向符号，直接从GT XYZ顺序求差分。

```text
GT faces [N,16,16,6] + half-edges [D,16,6] + connectivity [D,3]
    -> unchanged CNN/GAT/attention encoder
    -> FacePosterior mean/logvar [N,64] -> sample z [N,64]
    -> Linear 64->768 -> 8 x self-attention(16 heads, FFN2048)
    -> Linear 768->64 -> F [N,64]
           |
           +-> FaceDecoder
           |     CNN feature A [N,96,16,16]
           |     local XYZ+normal [N,16,16,6]; center/scale [N,4]
           |     global XYZ [N,16,16,3]
           |           |
           |           +-> FixedBSplineSurface -> poles [N,16,16,3]
           |           |                                |
           |           +-> concat(A[96],XYZ[3])->Linear99->512
           |                 + grid UV MLP -> tokens [N,256,512]
           |                                      |
           |                    gather owners i<j, add side embedding
           |                    concat token sequences -> M [P,512,512]
           |                                      |
           +-> concat(F_i+F_j,abs(F_i-F_j)) [P,128] |
                 -> MLP128->512->512               |
                 + t MLP1->32->512 -> Q [P,16,512] |
                                      |           |
                          8 x CurveAttentionBlock |
                          self-attention Q        |
                          cross-attention Q <- K,V=M
                          FFN512->2048->512
                                      |
                                H [P,16,512]
                         /                          \
             mean over16 -> LN -> Linear1     LN -> Linear512->4 -> sigmoid
                   logits [P]                       UV [P,2,16,2]
                                                        |
             predicted poles + owners -----------------+
                                                        v
                                             XYZ [P,2,16,3]
```

`P`为每件内部的全部无向面对候选总数；不跨CAD配对。两侧共享query，分别输出UV；仍未保证两条XYZ轨迹严格重合。网络中无独立XYZ曲线头，无新增法向头。GT只用于encoder重建输入和监督匹配，不筛选decoder候选。

```text
GT faces -> same fixed spline -> offline project GT curve XYZ
                                    -> GT UV [R,2,16,2] + valid [R,2,16]
GT pairs [R,2] -> match prediction rows
pred XYZ vs GT XYZ -> jointly choose forward/reverse for each edge
                          |
                          +-> same choice for XYZ / UV / valid / tangent
                          |
                          +-> absolute XYZ L1             x1
                          +-> GT-scale relative XYZ L1    x0.05
                          +-> valid UV L1                 x0.1
                          +-> XYZ difference cosine loss  x0.01
pred XYZ side0 vs side1 ------> incidence MSE              x0.1
all candidate logits vs pairs -> balanced existence BCE   x0.1
face local6 / center-scale4 / globalXYZ3 L1                each x1
posterior KL sum                                              x5e-7
```

相对项逐边除以`max(max_k||GT[k]-mean(GT)||,0.01)`，所以其附加XYZ系数至多5；不使用预测尺度。切向差分在GT长度>1e-6时有效；预测零差分仍被惩罚，归一化eps=1e-6。闭曲线本轮只沿用共同正反匹配，不新增循环起点匹配。

所有训练项同时记录raw和weighted值。`curve_xyz`保持旧min-MSE诊断，`curve_xyz_l1`是新目标；`face_normal`诊断不重复计入。`kl_loss`历史命名已含5e-7，新增`kl_unweighted`显示未乘权值，`weighted_kl_loss`只加入总目标一次。

推理接口仍为`dual_uv_surface_v1`：预测面点/控制点、两侧UV、两侧XYZ、owner及存在性。旧OCC流程不能直接接该表示。本轮未修改表示，已有checkpoint严格加载；全量续训改变encoder后，旧latent cache/diffusion不能默认与新checkpoint兼容。
