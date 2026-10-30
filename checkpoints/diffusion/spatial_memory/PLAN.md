# 照片 → 空间 memory → B-Rep latent diffusion

2026-10-28，用户确认实现方向 A。分支 `codex/photo-spatial-memory-20261028`。

## 研究问题

在 VAE、配对照片及目标 latent 不变时，将图像条件组织为带三维坐标的空间 query，是否有助于从二维观测生成三维 B-Rep？原照片 baseline 仅向每个 face slot 广播同一个 DINO CLS 条件，局部图像证据没有直接的逐层读取接口。

这是条件表示的研究假设。固定 XYZ 地址提供参数共享和位置区分，并不自动赋予 token 正确的占据、深度或局部曲面语义。单图隐藏部分仍由条件概率模型补全，不要求单张图唯一决定完整 CAD。

## 本轮实施合同

- 冻结原 SVR checkpoint 内的 DINOv2 ViT-L/14-reg 和完整 VAE，与 photo310k baseline 同源。保留 30×32 latent、随机复制 padding、epsilon MSE、1000 步 cosine DDPM；不用 DPO 或新增几何 loss。
- 照片只读 `real_photo.npz[flux]`，512→224 整图缩放、ImageNet normalize。没有 GT 相机、GT topology、GT 面数或 GT 面顺序的推理输入。
- 固定 canonical 网格的 8³ 个格心，extent=1，XYZ + 6 频带 Fourier 编码后用 39→512→512 MLP 构造 query。4 层完整 self-attention / image cross-attention / FFN，输出 `[B,512,512]`。坐标系取原 identity latent 的对象坐标约定，不为每个样本读取 GT bbox。
- 保留 256 个 RGB patch token 旁路；两类 memory 沿 token 轴拼成 `[B,768,512]`。空间 lifting 使用 CLS+patch，RGB 旁路不额外复制 CLS。没有显式深度、表面监督或相机投影 bias。
- 新去噪器保留 24 层、1024 宽、16 heads、FFN2048；每层为 self-attention → cross-attention → FFN，时间 adaLN 调制 self-attention 和 FFN。与已有 `TransformerEncoderLayer→cross-attention` 的执行顺序不同，使用独立注册名称/配置，不修改旧网络默认行为。
- face slots 始终是无序集合，不给第 i 个 face slot 指定第 i 个空间位置。attention `[B,16,30,768]` 只产生 30 个输出。
- 空间编码器和去噪器从头训练；旧照片 baseline/DPO checkpoint 不能 resume。只有冻结 DINO/VAE 来自原 checkpoint。完整新 checkpoint 自含这些冻结参数，恢复不需再次读取原 SVR 权重；仍需要固定历史源码 Git 对象及本地 DINO 源码。

## 数据与验证

复用 `photo_preference_data/20261026T055801Z` 中原 VAE 的 GT mean/std cache 和逐图输入摘要，不读取偏好对、winner/loser 或生成候选 latent。清单是完整 46,627 train / 3,505 validation，rotation=0。配置绑定生成协议 SHA256；逐样本检查 latent 内容、photo 摘要与 GT 源摘要。该数据是 FLUX 合成真实感图像，不能据此声称实拍能力。

实现验收包括 face permutation 等变、坐标进入计算、首步空间分支梯度、全部 trainable 参数有限梯度、冻结模块无梯度、采样每图只编码一次、真实小批量训练、原 VAE 解码接口、多卡保存及严格恢复。短采样只检查接口，不能说明 STEP validity 或图像一致性提高。

研究评价使用冻结的完整 validation 清单、单次输出 K=1、DDPM1000、相同 seed/FP32/后处理版本。失败保留在分母，分别报告生成覆盖率、合法 solid、几何误差、可见区域一致性和时延。实际效果仍待正式训练及下游评估。

必须区分两类对照：photo310k 是已有实用基线，但它还继承了 SVR 预训练，不能把与本轮从头训练的差异全归因于空间条件。归因需要相同新 backbone/初始化/训练曝光预算的 RGB-only 对照；进一步比较无 XYZ 的 query memory 才能判断三维位置编码的贡献。RGB 旁路可能使模型绕过空间路径，需要这些下游对照判定，而不是仅看 attention 图或 loss。

## 参考与边界

- [LRM](https://arxiv.org/abs/2311.04400)：image-to-spatial-query 的参考；其相机和 triplane 解码合同不能直接套入当前照片/固定 B-Rep VAE。
- [Transformer](https://arxiv.org/abs/1706.03762)：cross-attention 中 Q 与 K/V 长度可不同。
- [HoLa](https://arxiv.org/abs/2504.14257)：本轮保持原 holistic surface latent；没有将其误解释为局部空间 voxel。

本轮不实现 decoded-geometry refinement、联合点云 diffusion 或 3D teacher。相机未知时采用学习的 lifting；不能把原灰模相机当作 FLUX 照片的可靠标定。
