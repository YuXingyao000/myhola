# BRepNet Source Guidelines

研究目标和协作原则见根目录 [AGENTS.md](../../AGENTS.md)。本文件描述当前实现与接口约束；允许有理由的架构重设计，不把已有实验方向设为固定路线。

## 入口与配置

- `train.py`：`python -m src.brepnet.train`，`UnifiedTrainer` 根据 `model.stage` 分派 VAE / diffusion，数据类通过 `dataset.name` 查找。
- 多卡由 `trainer.devices` 启用，W&B 仅记录；`batch_size`／`num_workers` 按卡计。joint-boundary frozen 默认 8 卡 × batch 32，joint 每卡 batch 4；改变全局 batch 后要重新核对样本曝光预算。正式测试／NPZ 导出暂要求 `trainer.devices=1`，避免 distributed sampler 补齐产生重复输出。变长 packed geometry 配置关闭 `trainer.cudnn_benchmark`，其他配置保留原行为。
- 所有训练入口默认使用 `checkpoints/<exp_name>/<UTC 时间戳>`。`trainer.run_dir` 指定完整路径；续训／测试显式提供 `trainer.resume_from_checkpoint`，恢复原 run。`trainer.output_dir` 始终表示输出根，不被 Hydra 改写成 run。运行目录和逐对象 NPZ 合同分别由 `run_management.py`、`prediction_export.py` 管理。
- `models/__init__.py`：`build_model`；主模型在 `models/vae/` 与 `diffusion/model.py`。旧 `models/diffusion*.py` 保留兼容导出。
- `inference.py`：独立推理工具，手工配置已切到逐层 image cross-attention，单图输入为 `[B,3,224,224]`。权重通过 `load_diffusion_state_dict` 加载，保留独立指定的 VAE，严格检查其余 key；仍需核对训练时的完整配置，不能认为手写默认值完整复现任意 run。
- `eval/run.py`：`python -m src.brepnet.eval.run`；`post/` 完成 OCC B-Rep 构造。
- 行为配置归属 `configs/model/`、`dataset/`、`condition/`、`trainer/`、`datagen/`；不要复活已移除的 `model.py`、`diffusion_model.py`、`train_diffusion.py` 入口。

## VAE：表示与解码

2026-10-18起使用 `models/vae/` 包，具体结构见 [说明](../../checkpoints/vae_structure/README.md)。`BaseVAE`约定编码、采样、解码与forward接口，但允许各版本不同的参数和返回结构；主类在legacy/intersection/joint三个文件。组件在encoders/decoders、监督在losses、权重迁移在checkpoints、投影在geometry。旧模块导入仍兼容，旧日期记录中的 `vae.py`／`blocks.py` 为历史位置，不据此继续往旧文件堆代码。

模型按根指南的“实验代码与架构记录”约定编写：不在各层重复 shape/type/device/finite 检查，不另建 validator 隐藏这些检查。`forward` 用简短形状注释展示真实计算；实际 mask、空候选、精度转换照常保留。实验整体的层级结构、张量职责、配置和实现 hash 放在主题 `ARCHITECTURE.md` 的 ASCII 图中，与代码执行解耦。当前 UV 实验见 [架构快照](../../checkpoints/uv_boundary/ARCHITECTURE.md)。

`models/vae/legacy.py` 的 `AutoEncoder` / `AutoEncoder_light` 编码面 UV 采样点、半边采样点与 edge-face connectivity；图网络、注意力与全局聚合把几何／拓扑融合到每面 latent。常用 latent 维度是 `8 × 2 × 2 = 32`。这是包含全局信息的 holistic 表示，不能直接当作独立局部 surface code。

`sample()` 内部参数化 mean/logvar，但返回的缓存统计量是拼接的 **mean/std**。解码器恢复面几何，并通过有序 face pair 的 neural intersection 预测连接和有向半边几何。每个 pair 只产生一个候选半边；多重公共边、复杂裁剪和全局闭合不能由 adjacency 分类保证。最终合法 solid 仍依赖 `post/` 的拟合、优化、裁剪与缝合。

改变 VAE 或归一化时说明旧 latent cache、AE checkpoint 与 diffusion checkpoint 是否兼容；不能复用名称相同但定义不同的缓存。正式 diffusion run 确认冻结 AE 加载了正确 checkpoint，并与 cache 同源。

`vae/joint.py:AutoEncoder_joint` 是独立的 shared-boundary decoder 变体，解码 block 在 `vae/decoders/joint.py:JointBoundaryDecoder`，匹配与损失在 `vae/losses/joint.py:joint_boundary_loss`。训练合同在 [PLAN](../../checkpoints/joint_boundary/PLAN.md)。使用 `AutoEncoder_joint_dataset`，encoder 保持旧 face／half-edge 输入；shared-boundary target 只用于匹配与监督，不作为 decoder 输入。`freeze_encoder` 由模型同时控制梯度与 train/eval 模式；trainer 的 `trainable_scope` 保持 `all`。

`AutoEncoder_intersection` 是原 light VAE 的一轮反馈变体，协议与命令见 [README](../../checkpoints/intersection_refinement/README.md)。使用相同严格 dataset/accepted split 校验互反半边，输出仍为 `legacy_half_edges`。`vae/encoders.py:HolisticEncoder/FacePosterior` 保留原 encoder/posterior，`vae/decoders/intersection.py:IntersectionDecoder` 枚举全部对象内有向非 self 面对，`vae/decoders/refinement.py:IntersectionRefiner` 从预测几何与预测概率生成 ΔF/ΔE，原几何头与线性 classifier 复用。只对 logits 双向平均。GT 不筛 decoder 候选；`vae/losses/intersection.py:IntersectionLoss` 集中处理原目标、global XYZ、连续网格贴合与 GT 对应 twin 一致性；投影计算与 tensor 状态结构分别位于 `vae/geometry.py` 和 `vae/state.py`。`model.init_legacy_checkpoint` 严格迁移整个旧模型；`model.checkpoint` 严格加载完整模型；resume 跳过所有组件初始化。默认冻结 encoder/posterior；latent cache 兼容还要求实际 encoder checkpoint、输入与 posterior 同源。没有额外 decode face 去重，也没有显式 loop/vertex 预测。保留原 `inter`/classifier，独立拓扑 head 不属于本轮范围。

`train_vae_intersection_local` 是可选的局部注意力 inter 实验，见 [协议](../../checkpoints/intersection_local/README.md)。`model.intersection.kind=local_attention` 用面 pair 的单 Query 读取两面 CNN 局部特征与预测 XYZ；保留有向 E、classifier、refiner 和原损失，不引入 H/P。local 分支的 `model.init_refinement_checkpoint` 仅替换旧 `decoder.inter.*`，严格保留全部其他权重，创建新 run／optimizer；完整 local checkpoint 仍严格加载／resume。新分支的 candidate 分块独立于 refiner 分块，不能混用。旧配置默认 legacy，不改变旧 checkpoint 架构。

2026-10-12 local容量修正后，主配置使用 `block_type=transformer_decoder`，恢复原inter的8层512维／8heads／FFN2048／dropout0.1和完整self-attention＋cross-attention＋FFN，输出写 `intersection_local/full_capacity`。此前2层128维版本仅作为历史证据保留；复现／续训用 `train_vae_intersection_local_small_legacy`。缺省block_type按历史cross_attention解释，防止旧checkpoint语义漂移。两种架构不能混用权重；不得仅凭遗留顶层 `intersection_layers: 8` 推断历史local实际容量。

2026-10-13 将 `holistic_encoder.py`、`local_intersection.py`、六个 `intersection_*` 与三个 `joint_*` 文件归并至 `vae.py`／`blocks.py` 后移除原文件。原模型注册名称、模块属性和已有 state key 保持不变；移动类的源码位置不改变旧 checkpoint 的层级。local 网络 block 为 `blocks.py:LocalPairIntersection`，模型注册仍统一指向 `vae.py`，不新增 trainer 分支。

可选最终边几何头调制见 [curve_conditioning README](../../checkpoints/intersection_refinement/curve_conditioning/README.md)。`model.curve_conditioning` 默认 `{enabled: false, hidden_dim: 128}`；仅支持 `intersection.kind=legacy` 且 refinement 开启。`blocks.py:FacePairConditioning` 从 `[E1,F1_i,F1_j]` 产生缩放／偏移，仅施加于 refined edge CNN 和 center/scale MLP 的最终输出层之前，coarse 与 encoder-edge teacher 路径不调制。末层零初始化；原 inter 的层数、宽度与原损失不变。此分支的 `model.init_refinement_checkpoint` 严格保留原 refinement 全部旧参数，只允许新增 `decoder.edge_geometry.conditioning.*`，拒绝缺少旧参数、额外参数、形状不符及已 conditioned/local 来源。完整 conditioned checkpoint 使用同配置的 `model.checkpoint` 或 resume 严格加载；这不是 local 分支的 inter 替换迁移。

对 `AutoEncoder_joint`，旧 encoder 权重只能通过 `model.init_encoder_checkpoint` 做显式组件迁移；完整新模型权重初始化使用 `model.checkpoint` 严格加载；恢复优化器／步数使用 `trainer.resume_from_checkpoint`。新模型禁止旧 trainer permissive `init_from_checkpoint`；resume 时不重复迁移旧 encoder，也不能在无权重来源时冻结随机 encoder。新 VAE `v_test=True` 返回 `{"items": list[dict]}`，每个对象保留本地实体编号与表示标签；旧 VAE 测试保持 batch=1，不能把整个 batch 重复写入各样本。

`train_vae_intersection_point_feedback` 在已训练 conditioning 上新增最终曲线逐点反馈，见 [协议](../../checkpoints/intersection_refinement/point_feedback/PLAN.md)。`blocks.py:CurvePointFeedback` 用每点17维预测几何观察加E/两个owner F，145→128→64→3末层零初始化，新增27139参数。所有几何读取与新增MLP/坐标加法使用FP32，原主干仍遵循既有AMP。无点序号/空间池化/GT筛选；方向分量、bbox、face输出不直接修改；local XYZ和global XYZ同时修正。`init_curve_feedback_checkpoint` 仅允许从完整已conditioned、无feedback的legacy/refinement来源新增这6个tensor，full/resume保持严格加载；通用trainer的resume清除此初始化来源。`trainer.max_time` 默认null，本轮配置8小时；Timer时间计入checkpoint，续训须设置足够的累计预算。

2026-10-15 新增可选 [Encoder边内容消息实验](../../checkpoints/encoder_edge_value/PLAN.md)。`blocks.py:EdgeValueGATv2Conv` 保留原GATv2 attention/value/self-loop/bias，增加原始 `[target F,source F,E]` 的 `128→64→32` 消息增量，末层零初始化。`vae.py:HolisticEncoder` 按 `model.encoder_edge_value.enabled` 替换五层GAT，旧key不变，仅新增51,680参数，默认关闭。`init_encoder_edge_value_checkpoint` 只从完整conditioned、无point-feedback/value分支的模型迁移；完整加载/resume仍严格匹配。`train_vae_encoder_edge_value` 两组均解冻encoder/posterior/decoder，恢复既有KL项；原GAT解冻对照与候选使用相同预算。此实验改变latent语义，旧cache/diffusion不可直接认定兼容。

2026-10-17 新增可选 [UV面编码器](../../checkpoints/uv_face_encoder/README.md)。`blocks.py:UVFaceEncoder` 接收`[M,6,16,16]`，几何MLP与归一化UV索引MLP相加，六层512维／8heads／FFN2048的PreNorm attention保留全部256 tokens，单query汇聚后输出`[M,32]`，再进入原面间attention／GAT／global／posterior。`model.face_encoder.kind`默认`cnn`，`uv_attention`只替换`encoder.face_coords`；旧默认参数、state与RNG行为不变。新模块22,219,040参数，比旧CNN增加171,000；不增加最终latent维度。`init_face_encoder_checkpoint`严格验证完整conditioned来源（包括被弃用的旧CNN），仅初始化UV、保留其余689项；新run解冻全模型，旧loss及KL不变。完整UV加载须同配置并清空组件初始化，resume自动清空；strict state加载不替代对nhead等无shape变化配置的核对。入口`train_vae_uv_encoder`，不混用local-inter、point-feedback或edge-value。仅实现与CPU合同验证已完成，未正式训练、未验证OCC收益；旧cache／diffusion不能默认兼容。

## Diffusion：实际数据流

1. `dataset.py:Diffusion_dataset` 读取 cache 并完成 padding；`diffusion/latents.py:LatentProvider` 取 mean，或用 std 采样。非 cache 分支调用冻结 VAE 再由 padder 打包。
2. `diffusion/model.py` 对 `[B, F, 32]` 加噪；默认 DDPM 1000 steps、epsilon prediction、L2 loss；损失汇总与采样分别在 `losses.py` / `sampling.py`。
3. `diffusion/conditioning.py` 选择条件编码器。此次研究与实质验证以 single image 为范围；sketch、point cloud、无条件保留接口，不据此宣称已验证同等能力。显式拒绝 `multi_img` / `text`；有 YAML 或论文支持不代表当前路径可用。
4. `diffusion/image_encoder.py` 默认冻结 DINOv2 ViT-L，224 输入得到 257 个 token（CLS + 256 patches），再训练 projection。Depth Anything V2 选项只使用其 ViT 特征，不等于显式深度预测／监督。
5. `diffusion/denoiser.py` 先将 noisy latent 投影为 `[B,30,768]`，在主干入口加 timestep embedding `[B,1,768]`。保留 24 个完整 TransformerEncoderLayer（pre-LN self-attention + residual、pre-LN FFN + residual），每层后由 `fusion.py` 的独立 image cross-attention 再查询共享图像 memory；各层 Q=768，K/V 从 `condition.output_dim`（默认 1024）投影到 768，12 heads。
6. 每个 DDPM timestep 都运行 24 组 encoder + cross-attention；推理图像编码一次。首个 image query 已含 t，但没有逐层另加 t 或 adaLN。24 组后执行最终 LN 与输出投影。旧前端 4 层 fuser、全局 alignment 及未用的 camera 参数已移除。
7. 采样后筛选／去重 face latent，交给冻结 VAE 解码。当前 diffusion loss 不直接优化 OCC solid validity，也没有直接渲染重投影损失。

当前配置为 `model.denoiser.conditioning=layerwise_cross_attention`，条件维度直接来自 `condition.output_dim`。
当前先用 `train_diffusion_layerwise_svr.yaml`：训练/验证 photo ratio 均为 0，只读 `imgs.npz["svr_imgs"]`，不要求或读取 `real_photo.npz`；保持 `is_aug=false` 的视角 0 与对应 cache。`train_diffusion_layerwise_real.yaml` 保留生成图条件入口；`baseline` 文件名不保证旧模型。
SVR 吞吐配置与本机验证见 [说明](../../checkpoints/diffusion/throughput/README.md)：首视图局部解压、省略未使用原图、每进程 CPU 线程4、DDP unused 检测关闭；逐步日志为 rank-0 local loss，全局 epoch/validation loss 仍聚合。`trainer.diffusion_validation_inference` 默认关闭且永远跳过 sanity；正常 validation/选模不变。显式暂存工具仅复制 train/validation 的灰模图与 rotation-0 latent，切换 photo/augmentation/test 前必须更换数据根目录。
旧 `condition_fuser` 配置显式报错。历史 baseline tag `archive/2026-10-16/codex/diffusion-modular-baseline`（`b67c5d8e`）保留旧实现；旧 checkpoint 不能直接用于本架构 resume，不能用 `strict=False` 跳过新增层或旧 fuser 权重。
`load_diffusion_state_dict` 只允许省略独立 VAE 权重，严格检查其余参数；不会用 diffusion checkpoint 内的 AE 覆盖单独加载的 AE。
`checkpoints/diffusion/VALIDATION.json` 是旧模块拆分的等价性报告，不是新架构的验证或效果报告；新接口验证见同目录 `LAYERWISE_VALIDATION.json`。

## Face padding 与 topology 的共同契约

- 默认 random padding 复制真实面并随机排列到 30 slots；mask 全为 true。`RandomFacePadder.postprocess()` 用 latent 距离阈值去重，面数由此间接产生。
- zero padding 有 validity head；当前训练输入 denoiser prediction，推理输入最终 latent。epsilon prediction 下两种输入分布不同，切换 padding 不能假设等价。
- Oracle adjacency 在 dataset 中通过同一 padding index 重排。新增 face-level 标签也必须使用相同映射。
- 当前 `LearnedTopologyPredictor` 从图像 CLS 预测固定 slots 的 adjacency；默认 learned 配置仍用随机 latent 排列，没有显式 slot matching。先解决对应关系，再讨论预测精度或 attention 强度。
- `experiments/2026-06-19/cvae_diffusion_two_step.py` 读取 GT adjacency 计算 canonical ordering，并使用 GT 面数恢复预测；只能作为带 GT 信息的诊断路径，不能当作部署时纯照片推理。
- `source: gt_adjacency` 是 oracle；`source: learned` 读 raw `conditions["imgs"]`。存在预测缓存或 `freeze=false` 不表示已经实现 topology BCE 联合目标；当前 `Diffusion.forward()` 没有自动加入这些辅助损失。
- additive bias `+2` 将相同 base logit 下的未归一化 attention mass 乘 `exp(2)`；`log_multiplier` 使用 `log(1 + (max_multiplier-1)*P)`。这不保证 feature 输出被相同比率约束。旧数值解释错误不等于已经证明 two-step 退化的原因。
- `dataset.topology_visibility_weighted` 当前用 `min(v_i,v_j)` 修改 oracle adjacency；它不是完整的 visible/hidden 分解模型，也不同于实验脚本中的 visibility-weighted BCE。

## 已知配置漂移（2026-10-09 源码审计）

| 区域 | 使用前要处理的问题 |
|---|---|
| `configs/train_diffusion_real.yaml` | 继承 `is_overfit=true`、`validation_real_photo_ratio=0.0`；额外写的 `scale_factor` 不被 diffusion loader 读取 |
| `configs/train_diffusion_baseline_real.yaml` | 显式修正 overfit 与 real-photo validation，但有历史机器路径和 8-GPU 设置，需要 overrides |
| `AutoEncoder_dataset3` | 支持规范 `raw_data_root`／`epoch_scale_factor`／`condition_name`／`condition_data_root` 及旧别名；双定义冲突报错 |
| cached condition | 当前图像分支读取 `imgs`，不能只打开 cached flag 就假定特征缓存可替代图像 |

Diffusion loader 字段为 `raw_data_root`、`cached_latent_root`、`condition_data_root`、`train_dataset`、`val_dataset`、`test_dataset`、`epoch_scale_factor`。记录实际解析配置和实际消费字段；配置解析成功不代表数据语义正确。

## 修改与验证

Diffusion 实现在 `diffusion/`：cache 在 `latents.py`，padding 在 `padding.py`，编码选择在 `conditioning.py`，图像编码在 `image_encoder.py`，单层 image cross-attention 在 `fusion.py`，24 组路由及去噪主干在 `denoiser.py`，可选拓扑在 `topology.py`，装配／loss／采样分别在 `model.py` / `losses.py` / `sampling.py`。旧 `models/` 路径保留当前对象导出，不恢复已删除的 fuser / alignment。说明与验证见 `checkpoints/diffusion/`；后续新增完整架构可建新模块与显式 config。

重设计交代训练／推理各自可得的信息、face correspondence、mask、相机约定、输出表示、checkpoint 迁移及最终评估。联合 geometry/topology 训练可以作为新方向，需明确损失语义和权重，不沿用旧指南中限制联合训练的默认立场。

相关检查包括导入、配置消费、真实小批量前向／反向、有限梯度、排列一致性与训练／推理接口。只运行与改动有关的测试；研究有效性由固定协议的下游结果判断。文档整理不需要启动训练或下载模型。
