# UV Intersection 64D：直接双侧 UV 第一版

2026-10-25。用户已确认架构并授权在 142 启动训练。当前记录为启动前协议；实际启动和检查结果随后追加。

## 研究问题与模型

检验“共享的逐点曲线特征直接读出双侧 UV，再用连续曲面计算三维几何 loss”能否学到一致的边界。
首版保留一套 `LayerNorm + Linear(512,4) + sigmoid`；不额外添加 owner-conditioned UV 输出头。
旧 refinement、conditioning、独立三维曲线 CNN/bbox、encoder-edge teacher 和 edge_feature 匹配不参与新模型。
本轮不把初步训练收敛当成 OCC 合法率收益。

```text
CAD face/edge/connectivity
    |
retained HolisticEncoder + FacePosterior [FROZEN / eval]
    | training: sample; validation: posterior mean
    z [N,64]
    |
GlobalFaceAttention [TRAINABLE, inherited weights]
    Linear64/768 -> 8 x TransformerEncoder(16 heads, FFN2048) -> Linear768/64
    |
    F [N,64] --------------------------------------+
    |                                              |
FaceDecoder [TRAINABLE, inherited weights]      CurveQueries
    | local XYZ/normal + center/scale              | 16 query positions
    +-> spatial [N,96,16,16]                       | [p,16,512]
    +-> XYZ [N,16,16,3]                            |
          |         |                             |
          |    FaceSpatialTokens -> PairMemory ---+
          |         [N,256,512]     [p,512,512]    |
          |                            CurveAttentionStack
          |                            8 x CurveAttentionBlock
          |                               /             \
          |                         DualUVHead      ExistenceHead
          |                          [p,2,16,2]        [p] logits
          v                              |
FixedBSplineSurface.fit -> poles ---------+
                    surface.evaluate -> [p,2,16,3]
                                        |
              GT curve XYZ loss + dual-side incidence loss
```

新的网络定义仍全部位于 `src/brepnet/models/vae/uv_intersection.py`，保留清晰的六行
`UVIntersection.forward`；SHA256 为 `11f02b4106a9be9a33ca5d228148db71bb584777f0d7fcfa222f757d4b1a7fa8`。
`uv_intersection_training.py` 仅负责旧 encoder 装配、权重迁移、数据/loss 适配和导出，不定义新神经网络层。
运行代码使用独立工作树 `/mnt/d/python_uv_train`、分支 `codex/uv-intersection-training`；
保留原工作区的其他未提交修改，正式源码提交后再部署。实际 Git commit 由运行 metadata 记录。

## 初始化与数据

- 源权重：142 `/mnt/d/python/checkpoints/latent_expansion/d64_no_edge_feature/20261023T034042Z/weights/last.ckpt`。
- SHA256：`4262f4998fe047e19c3ca21a835d74c183b7430f7a656c709ec8ed045262ada4`，source step 30000、epoch 82。
- 严格保留 encoder/posterior/global attention/face CNN/center-scale 的全部对应 tensor；仅新曲线模块初始化。
- 新 optimizer；不是旧训练 resume。完整恢复必须使用 `trainer.resume_from_checkpoint`，不再次加载初始化来源。
- DeepCAD v7：`/mnt/d/data/deepcad_v7`；保持既有严格 shared-boundary accepted split。
- train 46,480 CAD：`checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt`，SHA256 `14a00585500f1c7996411caa1b3fba796a1c39c2b935a1153ec98e30c69dc097`。
- validation 3,495 CAD：`checkpoints/joint_boundary/validation/val_contract_audit/accepted.txt`，SHA256 `ae607bfe5499e919d1e68a5110c0c211daf1158aedbe57a4756eda99f6f3f0ed`。
- 使用既有 cube24 训练增广；validation 不增广。GT 连接仅供旧 encoder 与训练监督，不筛 decoder 候选。
- 不读取旧 UV sidecar；`edge_uv_root=null`。不变更既有 accepted/rejected 名单，不查看 test 个例来调整规则。

## 训练配置

- 入口：`python -m src.brepnet.train --config-name train_vae_uv_intersection`。
- 部署计划：物理 GPU 1–7，每卡 2 个 CAD，梯度累积 8 次，nominal batch 112；epoch 尾部可能不足一次完整累积。
- GPU 0 已被其他进程占用，本次不使用或中断该进程。
- seed 42，Adam（weight_decay=0，与现有 trainer 一致），LR `2e-5`，30,000 optimizer steps；所有 decoder 参数训练，encoder/posterior 冻结。
- BF16 mixed；UV 与可微曲面保持 FP32，`float32_matmul_precision=highest`，clip norm 1。
- 每次验证使用 posterior mean，记录全量 validation loss；不生成逐对象导出。旧 checkpoint 的总 loss 不与本轮新 loss 直接比较。
- 新 loss：face local/bbox/global 三项 L1 + curve XYZ MSE + 0.1 incidence MSE + 0.1 balanced existence BCE。
- 几何监督仅作用于 GT 真边；分类作用于所有候选。冻结 encoder 的 KL 仅记录，不加入总 loss。
- 数据、resolved config、Git commit/diff、命令、seed、初始化报告、日志和权重分别保存在独立 run 对应目录。

## 启动前验证与边界

已有单文件网络 15 项行为检查；新 trainer 注册、恢复、均值验证、梯度累积和精度配置已有定向测试。
合成 checkpoint 的迁移验证通过；正式启动前还需在 142 对实际64D源做严格加载、真实 batch、冻结/梯度与导出检查。
142 上默认 decoder BF16 + AdamW 显存预检：30 面/435候选峰值7.58GiB，两个30面对象/870候选峰值15.22GiB。
显存预检使用 AdamW；实际 trainer 和正式源预检使用 Adam（weight_decay=0）。这只是资源检查，不是训练吞吐或质量结论。

本轮导出表示为 `dual_uv_surface_v1`，保留双侧UV、双侧XYZ轨迹、曲面控制点/结点/次数。
不把两侧平均后冒充唯一共享边；旧 OCC 构造入口会拒绝这一新表示。
非周期样条、闭合路径、同一连续曲面的裁剪与最终共享边构造仍是后续几何导出工作。
