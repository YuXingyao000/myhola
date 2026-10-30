# 单图输入证据增强：候选与架构讨论

2026-10-25 后续进展：用户已授权并启动原版 HoLa 照片 RGB baseline，完整缓存校验、训练协议与运行状态见 [photo_baseline](../photo_baseline/README.md)。下文保留当时的候选讨论，新增条件架构仍未启用。

更新：2026-10-25。状态：源码核对与文献调研；以下新增架构尚未实现、训练或评估。

## 当前优先顺序：先建立旧合成照片的 RGB 基线

用户希望先用已有、质量不够理想但可用的合成照片尝试训练，并先建立 baseline；当前请求重点是直观解释输入处理，尚未确认前文新增条件结构。以下复杂空间条件仍是候选，不作为本轮默认训练架构。

本次只读检查见 [输入检查契约](PHOTO_BASELINE_INPUT_SPEC.md) 和 [实际报告](photo_asset_check.json)：当前源清单 train 46,627、validation 3,505，二者没有 ID 交集；这些对象均存在 `real_photo.npz`。固定 seed=42 抽取的 4 个 validation 对象均能解压 `flux`，为 uint8 `[512,512,3]`，并含 `blender`、`flux_masked` 字段。抽样照片主体较大且带有场景背景，不能由四件推断整库主体位置或配对合格率。

原版 AE 重新编码的 `preference_data/20261020T134930Z/gt_cache` 覆盖全部上述 train 和 100 个 validation。全量 validation 的同源 cache 尚需补齐或显式指定已核验的其他来源。这里只检查路径存在，未逐件重新读数组或校验来源。

建议先固定以下基线：

- 模型：原版 1024 维、24 层 concat HoLa，从原始 SVR checkpoint 初始化，使用其同源 AE/cache；新 run 仅用照片条件做 GT DDPM 微调。先不引入 DPO、输出几何 loss、深度／法线／分割条件。
- 图片：读取 `real_photo.npz["flux"]` 的完整 RGB。对于已是正方形的照片直接等比缩放到 224×224，再按原 DINO 约定归一化；原 512 图保留。将来非正方形输入采用等比缩放与 padding，避免形变，所有比较固定同一变换。
- 第一版不做自动紧裁剪和额外分割；这批抽样主体已较大，先避免为基线引入新的分割误差。后续若研究小主体／复杂背景，再比较由输入估计的带边距裁剪或软 mask。不能把参考 CAD／灰模轮廓生成的 `flux_masked` 当作纯照片可用输入。
- 当前单张 `flux` 对应 identity 配对：不随机切换到其他 rotation latent，也不单独旋转／翻转图片而维持目标不变。初版固定预处理；轻度外观扰动作为后续单独变量。
- 沿用 CAD 级 split；训练和验证均为照片。第一版冻结数据版本，不根据“看起来不够真”全库清洗；损坏、缺字段和已知配错需记录，而风格不理想与几何配错分开处理。
- 评价以照片输入下的重建行为为目标。对合成编辑引入的相机／几何偏移，不直接沿用旧 SVR 固定相机的照片重投影指标并宣称可见一致性；该部分评价协议需单独核对。

现有入口不能照名字复用：`train_diffusion_real.yaml` 继承的 dataset 默认有 `is_overfit=true`、validation photo ratio=0；`train_diffusion_baseline_real.yaml` 修正了这两个字段，但实际装配 768 维逐层 cross-attention 模型，不是这里的原版 concat baseline。当前 `PreferenceDataset._image()` 固定读 `imgs.npz["svr_imgs"][0]`，其训练数据还强制要求偏好对；仅改路径／权重不是照片 DDPM baseline。正式实施需要显式照片入口、严格权重迁移与实际 batch 验证，本轮尚未修改或启动训练。

## 用户确认的目标域：照片

用户补充：灰模是当前训练阶段，后续输入虽然来自合成流程，但在模型设计上应按一张真实照片处理。后续讨论以照片域为目标：不依赖纯色背景、无纹理材质、固定照明或固定 SVR 相机。数据来源仍记录为合成真实感图像；其表现不能直接代替独立实拍评估。

相机／视角是否变化、是否有遮挡不是用户已确认的数据事实；架构上不默认这些信息已知，只有实际推理可获得的元数据才能进入条件。若后续明确保留固定相机或其他约束，可显式利用。目标对象暂按一张图有明确待重建主体设计，多对象场景的选择接口需另行定义。

相应修正：

1. 目标区域由图像分割／定位提供，保留原 RGB 与软前景信息，不用灰模阈值分割作为最终接口，也不把预测 mask 外的所有证据不可逆删除。裁剪、缩放及 padding 对 RGB、mask、深度、法线使用同一像素映射并记录变换；几何预测器需要的上下文可从原图保留，输出再映射到目标区域。
2. 深度／法线是有误差的辅助表示，使用相对／归一化几何，不把估计的米制尺度当作已知尺寸。法线先保持一致的相机坐标约定；未知姿态时不能直接将其解释成 CAD 全局法线。
3. RGB 空间特征保持主通路；深度、法线与前景作为辅助，原始图像边缘降为弱线索。材质纹理、阴影、反光都可能产生与 CAD 边无关的响应。是否保留 edge 通道需在照片域决定，7 通道只是初稿。
4. 二维 memory 的 cross-attention 不要求先知道准确相机。若增加点图的三维对应或重投影约束，则必须另行定义相机估计／可用元数据，以及 CAD 输出的规范化、朝向、对称性歧义；不偷偷沿用旧 view0 的 45° 相机。
5. 灰模用于初始化、接口验证与解释已有结果；照片域的 validation 决定预测器和架构选择。灰模模型不能通过加一个深度支路就被认为已具备照片能力。照片训练应覆盖目标域外观；只有几何保持一致的外观变化才能共享同一 CAD 监督。

保留 concat + 新增空间条件仍是候选，理由是复用已训练的 B-Rep 生成先验。冻结旧主干仅作为开始学习新条件的阶段；照片域是否需要解冻图像投影／主干由实际适配需求决定，不能预设永远冻结足够。新增残差零初始化的等价性仅针对**相同输入与旧前处理**；更换照片或裁剪会改变条件分布，不存在跨域质量不变保证。

## 研究判断

目标是让生成器更好地利用可见孔槽、轮廓、深度层次与表面朝向。预测的深度／法线是同一张图的派生表示，加上外部模型学到的先验，不是新的传感器观测，不能恢复本来不可辨识的背面。

此前几何辅助 loss 约束的是生成结果经冻结 VAE 解码后的几何；这里讨论输入侧表示。两者的作用机制不同，前者没有收益不能推出后者无效。输入增强也不能单独保证可裁剪、流形、闭合的 solid。

建议保留原图，先考虑“RGB 局部特征 + 一个几何预测器 + 前景／有效性／边缘”。不要同时堆叠多个大型深度模型，也不要把深度伪彩色图片作为新的 RGB 去替换原图。

## 当前源码事实

- 本轮研究对应 `src/brepnet/diffusion/preference.py:LegacySVRDenoiser`，是原版 HoLa 单图 concat 架构。它复用 `legacy.py` 指定的历史源码 `c2660c00c6a9c8f9f82bd5b9f4d3c774fa38d361`。
- DINOv2 ViT-L/14 register 版本被冻结；224×224 图像的全局 CLS 为 1024 维，`img_fc` 输出 256 维，复制给 30 个 latent slots。
- `p_embed` 为 Linear(32,768) → LN → SiLU → Linear(768,768)。与图像 256 维 concat 后主干宽度为 **1024**，不是 768。
- 时间编码为 sin/cos(1024) → Linear(1024,1024) → LN → SiLU → Linear(1024,1024)，广播相加。
- 主干 24 个 pre-LN TransformerEncoderLayer：16 heads、每 head 64 维、FFN 1024→2048→1024、ReLU、dropout=0.1；DPO 的确定性比较路径关闭 dropout。末尾有 LN。
- 输出头 Linear(1024,1024) → LN → SiLU → Linear(1024,32)，预测 epsilon；采样后去重，再由原版冻结 VAE 解码、OCC 构造。
- 当前 DPO 不读取 DINO patch tokens。若用 `forward_features` 提取，224 输入有 16×16=256 个 patch tokens，每个 1024 维；CLS 和 register tokens 单独处理。
- `diffusion/image_encoder.py:62` 的 Depth Anything V2 选项只保留 `_da_v2.pretrained`，没有运行其深度预测头。该文件属于另一条模块化路径，不能称为当前 DPO 已经使用了深度图。
- `eval/metrics/observation.py:legacy_svr_camera` 记录原 SVR view0 的透视相机、45° FOV。这里的坐标 y 向上，而 MoGe 输出 OpenCV 相机坐标 y 向下；接入时必须显式转换。不能把另一套 Blender 渲染配置自动当作这些旧图的相机来源。

## 方法收集：官方资料截至本次查询

下表“对 CAD 的用途／风险”为研究判断，不是这些论文在本项目上的实验结果。

| 信息 | 可用候选与官方来源 | 对本任务的意义与限制 |
|---|---|---|
| 相对深度 | [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2)、[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) | 表面前后关系、台阶／凹槽；普通 V2 是相对深度模型，DA3 需明确 Mono、Metric、Any-view、Nested 的区别。相对输出不能直接解释成 CAD 尺寸。 |
| 联合几何 | [MoGe 官方仓库](https://github.com/microsoft/MoGe)、[MoGe-3 方法](https://qft-333.github.io/moge3page/) | point map、depth、normal、valid mask、相机；适合用一个预测器提供多种一致表示。MoGe-2 应选择带 `-normal` 的权重；MoGe-3 已有公开权重。 |
| 表面法线 | [DSINE](https://github.com/baegwangbin/DSINE) | 平面朝向、曲面变化；假定透视相机内参，缺省采用 60° FOV，不能原样用于本项目 45° 的旧 SVR。法线仍可能受光照、材质和先验影响。 |
| 边界清晰的深度 | [Depth Pro](https://github.com/apple-aiml-research/ml-depth-pro) | 适合关注轮廓附近的深度断层；输出 metric depth 与焦距估计。官方发布权重为重新训练的参考实现，不能把论文数字当成本机表现。 |
| 前景／轮廓 | 图像定位／分割、[SAM 2](https://github.com/facebookresearch/sam2) | 照片域使用从输入估计的目标区域与软前景；灰模背景阈值只用于当前诊断。SAM mask 不是 CAD face 分割，不保证保留孔；提示与候选选择只能使用推理可得的信息。 |
| 图像边缘 | Canny、[TEED](https://github.com/xavysp/TEED) | 为孔槽、轮廓和局部突变保留直接图像线索；阴影、反光、纹理边缘不等于 B-Rep 边，不应据此硬造 topology。 |
| 局部图像特征 | 当前 DINOv2 的 patch tokens、[DINOv3](https://github.com/facebookresearch/dinov3) | 避免只用全局 CLS 的信息压缩。先复用现有 DINOv2；更换骨干与加入几何是两个变量。特征不是物理坐标。 |

MoGe 官方仓库记载 2026-08-18 发布 MoGe-3；提供 ViT-L（370M）及 ViT-G（1.25B），有 normal 与 metric scale。其 sparse 3D refinement 针对细节与深度断层混合，可作为当前候选；这里没有复现其论文，也没有证据证明它在本项目的零件照片上优于 MoGe-2 或 DA3。

具体候选优先级：MoGe-3 ViT-L 作为联合几何候选，MoGe-2 ViT-L-normal 作为清晰的替代；深度路线保留 DA3 Mono-Large 与 DA V2。Depth Pro 用于比较边界细节，DSINE 用于检查独立法线估计。首轮不把这些模型同时输入 diffusion。DA3 的 Any-view / Nested 部分旧权重已被官方标为 deprecated，若选择这些系列须固定刷新后的具体权重与 revision。

## 架构 A：全局几何融合

保留原 RGB 全局条件 `c_rgb [B,1,256]`。几何图编码后池化为 256 维，通过末层零初始化的 MLP 得到 `delta_c`，使用 `c_rgb + delta_c`。其余 24 层完全不变。

优点是旧权重可完整保留，改动与训练成本小。限制是所有局部证据最终仍压成一个 256 维向量，不能指望它可靠表达每个孔槽的位置与关系。它是保守候选，不是推荐的主要结构方向。

## 架构 B：保留原 concat，增加空间条件（推荐讨论方案）

保留旧模型的全局 CLS → 256 → concat 路径、1024 主干、24 层和 VAE。新增空间 memory，让无序 face tokens 学习读取图像各位置。

### 条件编码器的具体初稿

1. 同一张照片：估计目标区域并记录像素变换；DINO 分支初稿使用 224×224 目标图，几何预测器按自身预处理读取有上下文的原图，输出再映射到相同的目标网格。旧灰模等价性检查单独使用原前处理。224 是复用旧权重的初稿，不是照片细节分辨率的最终上限。
2. RGB：冻结 DINOv2 的 patch tokens `[B,256,1024]` → Linear(1024,256) → LN；加二维位置与模态 embedding，得到 `[B,256,256]`。
3. 几何初稿：归一化 depth 1 + camera-space normal 3 + soft foreground mask 1 + geometry-valid mask 1 + 弱 edge 1，共 `[B,7,224,224]`。前景与预测器有效性是两个不同通道；有效性也不是校准后的置信概率。照片域是否保留原始 edge 通道待验证，不将其当作结构边标签。
4. 几何 patch embedding：Conv2d(7,256,kernel=14,stride=14) → `[B,256,16,16]` → flatten/transpose → `[B,256,256]`，加二维位置和模态 embedding。
5. 几何编码：2 个 pre-LN TransformerEncoderLayer，width=256、8 heads、FFN=1024、GELU，最后 LN；输出 `[B,256,256]`。这些层数与维度是可审核的初稿，不是已验证的最优值。
6. 沿 token 轴拼接两种 memory：`[B,512,256]`。没有把它们池化成全局向量。背景、无效区域和缺失模态应有显式 mask／占位策略，禁止全 masked attention 产生 NaN；预测有效性不能直接硬删所有疑似细节。

不把 depth、point map、normal 全部当作独立证据重复堆叠。已知内参时，depth 与 point map 强相关；初稿使用 depth + normal，point map 留作后续替代表示。

### 去噪层

原输入：`z_t [B,30,32] → p_embed → [B,30,768]`，concat 广播后的 `c_rgb [B,30,256]`，加时间编码，成为 `[B,30,1024]`。

初稿在第 6、12、18、24 个完整原 EncoderLayer 后各放一个独立残差条件模块，共四处：

- `Q = Linear(1024,256)(LN(h))`，形状 `[B,30,256]`；
- `K,V = Linear(256,256)(LN(memory))`，各 `[B,512,256]`；
- 8 heads，每 head 32；attention 权重 `[B,8,30,512]`；
- attention 输出 `[B,30,256]` → **zero-initialized Linear(256,1024)** → 加回 h；
- h 的形状仍是 `[B,30,1024]`。原末尾 LN 与 epsilon 头不变。

四处注入是控制新增成本的起点，不是“越多越好”的结论。最终方案需要判断是否出现更强局部条件响应与过度跟随错误几何之间的取舍。

图像 memory 每图计算一次，复用于所有 DDPM timesteps；训练时冻结的几何结果可离线缓存。每层 K/V 可在推理时预计算；训练带可训练投影时不能跨参数更新复用。预测器的实际开销尚未测量。

face slots 无天然顺序，不给它们加入固定二维图像位置，也不把第 i 个 patch 当成第 i 个 face。cross-attention 学习软对应，二维位置只属于条件 memory；这不保证注意力权重能解释成真实 face 可见区域。

这种连接借鉴 [IP-Adapter](https://arxiv.org/abs/2308.06721) 的独立条件 attention 和 [ControlNet](https://arxiv.org/abs/2302.05543) 的零初始化残差思路。它不是直接复制 ControlNet：B-Rep latent 是集合，不是与输入图像像素对齐的二维 U-Net 特征图。这些已有连接方法本身也不是新的研究贡献。

## 权重、训练与推理合同

- 从本次无几何 loss 的已训练 concat DPO 初始化；旧模块和旧 VAE/cache 保持语义、形状及全部权重。新增模块使用显式迁移白名单；不能用 `strict=False` 宽松跳过旧 key。
- 初始新增残差输出为零，在 eval 模式、相同旧输入下应与原模型输出相同。浮点与执行路径可能造成数值误差，必须检验，不能只根据图宣称迁移等价。
- 初期冻结旧模型，学习新增支路；之后是否联合微调旧 policy 由研究结果决定。几何预测器、DINO 和 VAE 冻结。迁移是新 run 初始化，不是旧 optimizer resume。
- 训练与推理均使用**预测**几何图。GT depth/normal 可用于训练集上的几何预测器适配，或作为单独标注的 oracle 诊断，不能在普通重建评估时混入条件。
- 首先让新增条件通过原 GT 去噪目标学习与 B-Rep latent 的关系；保持原 DPO 目标作为后续偏好微调选项。不再同时重新加入本次没有收益的输出几何 loss，以免混淆因素。
- 若保留 DPO，冻结 reference 继续使用原 RGB 条件；policy 可读取同一图像的派生表示。明确这属于输入表示扩展，离线 pairs 的输入 identity、来源与 reference 权重仍须核对。
- 训练时随机移除几何模态，并保留 RGB 通路，避免模型必须依赖某个预测器。无效像素用显式有效性通道处理，不把缺失深度零值解释成近表面。
- MoGe/DSINE 法线必须统一坐标与方向；只在有意义的像素归一化。相对深度、逆深度、metric depth 分开定义，不能直接互换。规范化只能依赖图像／预测结果或可用相机信息，不能用 GT CAD 尺度／对齐。
- 不能只随机旋转／翻转图像而保持配对 CAD、相机和方向图不变；普通 photometric augmentation 与几何变换不是同一件事。

## 研究问题与去留依据

主要问题：空间 RGB 已经提供的局部证据之外，显式预测几何是否进一步改善可见孔槽、深度层次和表面朝向，并保持生成有效性？

必要的三个可比条件：当前 CLS 模型、增加 RGB patch memory、同一空间接口再增加预测几何。否则无法判断收益来自新的读取通道、额外容量，还是深度／法线。若要严格分离容量，还需给 RGB-only 对照匹配新增编码容量，而不是只匹配训练步数。

validation 用于预测器选择、表示检查与结构取舍；test 在方案冻结后一次性比较，不根据既有 test 失败样本挑规则。固定相同对象、视角、旧初始化权重、样本曝光与采样／重试预算、VAE、后处理和随机种子。比较单次输出的可见一致性、孔槽边界、完整几何、STEP/solid 有效率及端到端成本；失败留在预定分母，成功交集的几何指标单独报告。

判断标准不是深度图“好看”、attention 可视化“像对齐”，或训练 loss 更低。若 RGB patches 已得到同等收益，显式几何应退出默认方案；若几何带来可见一致性提升但损害 solid 有效性，应追踪 VAE／生成分布／后处理之间的矛盾，不能仅凭单项分数决定保留。

本方向对专用模型的预期价值，是使结构化生成器显式利用可见几何并控制隐藏补全的分布；是否比通用 CAD agent 更准确或更省成本仍需相同输入和修复预算的独立比较。
