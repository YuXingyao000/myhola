# 对比资产审计（2026-10-11）

本审计支持 [PLAN.md](PLAN.md) 中的固定 checkpoint 解码替换比较，不把两次训练视为同预算架构消融。以下权重、覆盖和交集数字来自实际读取；坐标约定由同期日志与源码合同共同支持。机器上的完整记录位于 `metadata/inference/{posterior_mean,archived_latents}/protocol.json` 和逐分片 JSONL，运行资产不提交 Git。

## 权重与共享 latent 合同

| 项目 | Legacy | Joint |
|---|---|---|
| checkpoint | `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` | `/mnt/d/python/checkpoints/joint_boundary/frozen/20261010T084310.423740Z/weights/epoch=0538-step=00097559-val_loss=0.028098.ckpt` |
| checkpoint 元数据 | epoch 8,499；step 1,045,500；历史类名 `AutoEncoder_1119_light` | epoch 538；step 97,559；`AutoEncoder_joint` |
| 当前严格加载类 | `AutoEncoder_light` | `AutoEncoder_joint` |
| 全模型 state tensors | 704，全部匹配 | 338，全部匹配 |
| SHA256 | `8b714943d24c3f3f18f3a1e64222b1f3577323ddbf42e4b90d603186ab7e917a` | `9c57c51ca7918e11e7f400e3d78845bcccca7430fc38157d61daa544deae198a` |

两模型 encoder 与 posterior 的 **243 个 tensor 逐一 `torch.equal`，无差异**；输入通道、latent 维度、归一化类型、encoder 层数、Gaussian 参数化等配置同时核对。共同合同为全局 XYZ／normal 六通道面和旧半边输入、每面 32 维 latent；缓存统计是 `[mean(32), std(32)]`。`sample(..., v_is_test=True)` 返回 mean，见 [vae.py:475](../../../src/brepnet/models/vae.py#L475)。这支持当前固定 encoder 下复用 latent；不支持今后解冻 encoder 后沿用旧 cache 而不再验证。

使用 `weights_only=True, mmap=True`；旧文件需要显式允许已知 OmegaConf 数据容器，不使用任意 pickle fallback。模型按完整 state dict `strict=True` 加载，并拒绝键冲突、shape 不匹配与 encoder 差异。实现见 [安全加载与严格比较](../../../tools/vae_compare_inference.py#L57)。FP32 eval，autocast／TF32／cuDNN benchmark 关闭，seed 42；未开启 PyTorch 全局 deterministic algorithms，因此不宣称跨设备／库版本逐 bit 可复现。

主组每件只 encode 一次，把同一 posterior mean 交给两个 Decoder。Encoder 使用 GT 面／半边及连接是 VAE 重建输入；**Decoder 不读取 GT 拓扑**。Legacy 明确调用 `decode({"face_latents": z}, v_data=None, v_deduplicated=False)`，由全部有序 face pairs 自主预测连接；不调用默认隐式几何删面的 `decode_latents()`。见 [推理适配](../../../tools/vae_compare_inference.py#L217)、[旧自主拓扑分支](../../../src/brepnet/models/vae.py#L578)。Joint 保留 existence `>=0.5` 的全部 query，包括重复 pair／同面分配；旧阈值为 `>0.5`。原始表示分别标记 `legacy_half_edges` 和 `shared_boundary_v1`，不能直接把两者 edge count 当作同一实体数量。

## 历史 diffusion 归档的来源与坐标

归档根：`/mnt/d/python/experiments/baseline/diffusion_baseline_real_v7_cube24/test_outputs/`，共 **2,424** 个对象；本次 accepted test **2,417／2,417** 均存在 finite float32 `face_latents[F,32]`，`F` 范围 **5–28**。这些是生成后的 latent，不是 GT posterior cache。

同期测试 [hparams.yaml](../../../experiments/baseline/diffusion_baseline_real_v7_cube24/tb_logs/version_0/hparams.yaml) 记录 checkpoint `experiments/baseline/ckpt/realphoto_baseline.ckpt`，其实际元数据为 epoch **9**／step **182,130**。该 checkpoint 的 `model.autoencoder.*` 与 `model.latent_provider.autoencoder.*` 两份注册权重各含 **704** tensors，分别与上述 1119 Legacy checkpoint 全部逐 tensor 相等，无缺失／多余／差异。它是无 topology bias 的 baseline；配置中的 `source: gt_adjacency` 不生效，因为 `topology_bias.enabled=false`，且 `dataset.load_topology=false`。

该 diffusion checkpoint 的 SHA256 为 `8a4e368e4a76ec2661cb15a8f2de4ed45e34840b22f69135243d1364dd5529bc`；本次只读取它核验来源，重放不重新运行 diffusion。

归档的 `face_latents` 是 Decoder 的原始输入，不是内部 Transformer 变换后的 feature：[vae.py:625](../../../src/brepnet/models/vae.py#L625) 和 `decode_latents()` 的导出都保留原输入。此前 diffusion 已执行 random-padding 的 latent 去重，再调用 AE，见 [diffusion.py:171](../../../src/brepnet/models/diffusion.py#L171)。本次重放**不再次去重、不按 GT 面数筛选、不重新采样或选择候选**，两个 Decoder 接收相同归档数组。

Identity 坐标约定的证据：

1. `experiments/baseline/diffusion_baseline_real_v7_cube24/train.log:330` 是 **2026-06-21 07:13:11 UTC** 的实际测试 resolved config；第 365 行 `is_aug:false`，第 374 行 `real_photo_ratio:1.0`。同目录 hparams 第 35／44 行一致，图像 `augment_probability:0.0` 见第 136 行。示例 `00141434/data.npz` 修改时间为当天 **07:21:09 UTC**，与该调用相符。
2. [dataset.py:692](../../../src/brepnet/dataset.py#L692) 在 `is_aug=false` 时固定 `rotation_id=0`，读取 `{id}_0/features.npy` 与同编号条件图。[rotations.py:15](../../../src/brepnet/data/rotations.py#L15) 定义 view 0 为单位矩阵。**2026-05-31 已提交版本 `4346c7a2`** 的同两文件已有相同分支与 identity-first 约定，不是本次审计新加的解释。
3. [2026-06-19 NOTE:27](../../../experiments/2026-06-19/NOTE.md#L27) 明确 baseline／two-step 比较使用 real-photo view 0。配置里残留的 `dataset.augmentation.random_rotate/random_scale=true` 未被该 diffusion loader 消费；不能据此推断测试随机旋转。

因此可以按**同期测试协议为 identity-view 的固定归档**，与 identity GT 做不额外配准的几何比较。但归档未逐件保存 rotation／相机／源码 hash，旧 `test_post/_metadata/post_eval.json` 还记录当时工作树含未提交改动；本审计不声称逐文件证明了完整照片—CAD 配对。`real_photo.npz` 的 FLUX 图仍属合成真实感图片，不能把这组结果称为实拍泛化表现。

## 数据划分与结论边界

固定 accepted test：`checkpoints/joint_boundary/validation/test_contract_audit/accepted.txt`，**2,417** ID，SHA256 `2864849e8240134c6033cc6cd64efa19337f67fea94b89a493690ae84355b7b3`。原 test 为 **2,424**；7 个原契约不支持对象保留于总体覆盖率分母，不能当作成功预测。

| checkpoint 记录的训练清单路径 | 当前文件 ID 数 | 与全部 2,417 个 intended test ID 的交集 | 当前文件 SHA256 |
|---|---:|---:|---|
| `src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt` | 46,627 | 0 | `bfd0f90428a40600c6f9f72658544c018e75f5576ad6f0af00dbe29224ac9417` |
| `checkpoints/joint_boundary/validation/train_contract_audit/accepted.txt` | 46,480 | 0 | `14a00585500f1c7996411caa1b3fba796a1c39c2b935a1153ec98e30c69dc097` |

这是**当前文件的 ID 交集检查**，不是历史训练清单内容证明，也不是跨 ID 几何去重证明。旧 AE 元数据的数据根为 `/mnt/d/img2brep/deepcad_v6`，joint 为 `/mnt/d/data/deepcad_v7`；不能把旧 AE 描述成已验证在同一 v7 数据、相同预算训练。当前推理协议明确记录 `historical_list_contents_verified=false`。

本次两种模式各 2,417 件、每件两模型的 raw 导出均已完成，逐件状态无失败；这是执行完整性，不是几何／拓扑／solid 优劣结论。Posterior 重建与固定 diffusion 归档上的替换效果分别解释；二者都不能直接证明新的 diffusion 训练方案或真实照片重建能力。
