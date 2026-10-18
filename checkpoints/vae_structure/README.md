# VAE 代码组织（2026-10-18）

本次只整理代码结构。基线为 `8a68f1f6`，包括上一轮已实现的 UV boundary 分支；不增加网络层、不改宽度、loss、采样或后处理，不启动训练。

原 `models/vae.py` 有2,298行，同时包含三个模型系列、encoder/posterior、loss、checkpoint迁移和导出；`models/blocks.py` 的1,790行又混合了通用层与完整decoder。现将 VAE 实现统一放到 `src/brepnet/models/vae/`。

## 从哪里读

```text
models/
├── blocks.py                  通用残差层、位置编码；旧VAE名称仅兼容导出
└── vae/
    ├── __init__.py             保留模型名称、工厂和旧导入路径
    ├── base.py                 BaseVAE 抽象接口
    ├── legacy.py               AutoEncoder / Light / Light-exp 主类
    ├── intersection.py         AutoEncoder_intersection 主类
    ├── joint.py                AutoEncoder_joint 主类
    ├── encoders.py             局部编码器、GAT融合、HolisticEncoder、FacePosterior
    ├── decoders/
    │   ├── intersection.py     pair候选、原/local inter、IntersectionDecoder
    │   ├── refinement.py       几何读取、ΔF/ΔE、逐点反馈
    │   ├── primitives.py       面/边几何头、conditioning、UVPathHead
    │   └── joint.py            shared-boundary decoder
    ├── losses/
    │   ├── intersection.py     有向边对应、原几何loss、贴合/twin/UV监督
    │   └── joint.py            匹配与共享边界loss
    ├── checkpoints.py          权重读取与各实验的严格迁移规则
    ├── geometry.py             最近曲线/曲面点与数值投影
    ├── state.py                decoder张量结构
    └── profiling.py            旧计时工具的归属
```

只看某一代模型的装配与训练过程，打开对应主文件即可。新一代完整VAE可增加一个主文件并注册；复用的几何计算、网络组件或loss应放到各自目录，只有该类使用的计算则作为私有方法。

## 基类负责什么

[`BaseVAE`](../../src/brepnet/models/vae/base.py) 规定 `encode`、`sample`、`decode`、`forward` 和 `decode_latents`。`sample` 沿用既有重参数采样名称，统计量仍为 mean/std。

基类不持有额外网络层，不统一各版本的数据字典，也不接管loss和导出。具体 `forward` 仍留在每个模型主类中，直接展示编码、采样、解码、监督的过程。因此后续版本可以有不同输入输出和forward参数，不需要绕过一个越来越复杂的通用forward。

- legacy 的 `decode` 接收 encoding字典，可使用训练数据；`decode_latents` 保留原预测去重流程。
- intersection 新增明确的 `decode(latents, face_counts)`，原 `decode_state` 转发到它，输出 `RefinementOutput`；原训练、推理入口保留。
- joint 保持 `decode(encoding_dict, ...)` 和 `decode_state(latents, face_counts)` 的各自合同，输出joint state；没有强行改成与intersection同一输入格式。

旧legacy/joint仍有一些历史顶层参数名；本次不把它们重新包进 `self.encoder` 或 `self.decoder`，这样不会给state_dict强加新前缀。joint通过legacy复用编码和采样逻辑，intersection直接继承BaseVAE并装配独立encoder/posterior/decoder。

## 工具的边界

模型主文件不再堆放全局loss和迁移函数。两类loss各自保留模块内部的计算helper，没有无实际复用价值的 `common.py` 或万能 `utils.py`。

checkpoint迁移由普通 `IntersectionCheckpointLoader`、`JointCheckpointLoader` 类承担，不注册为网络子模块。它们记录初始化来源和替换的参数集合；缺失/额外key、加载张量shape由 `load_state_dict(strict=True)` 处理，不再重复写Python类型和shape检查。简单且仅由legacy使用的加载方法仍为该类的私有方法。

内部代码显式从定义模块导入。`models.vae` 和 `models.blocks` 的旧名称通过明确的兼容表解析，供现有trainer、diffusion、测试及旧pickle查找使用；不是第二套实现。包的 `__all__` 只公布基类、模型主类和工厂。

## 2026-10-18：研究代码可读性约定

拆包之后按用户要求进一步删去模型内部重复的张量shape、dtype、device、有限值检查及历史实验组合限制。`forward` 直接读取既有pipeline的数据，显示编码、采样、解码、监督；空候选、实际mask、AMP精度转换和几何稳定计算保留。删除的validator及其旧private导出不再保留空壳。

整体模型用 [UV boundary架构快照](../uv_boundary/ARCHITECTURE.md) 记录模块路径、配置、Git基准和ASCII张量图。它是独立设计文档，不是代码读取的协议文件；历史运行仍由各run的metadata追溯。

下面是**第一次拆包**的验证，清理检查后的验证另记在 `validation/readability_*.json`，不覆盖原结果。

本次清理后：相关测试222项通过，1项因缺少外部diffusion cache样本而跳过；8组模型的初始化、输出、loss及2,310组参数梯度与独立旧实现逐位一致。VAE包内手写 `raise` 从211处减至30处，剩余集中在架构选项、权重来源/迁移和匹配容量；没有张量shape/type/finite validator。见 [测试记录](validation/readability_tests.json)、[数值对照](validation/readability_equivalence.json)。本次不改变网络结构或训练目标。

## 第一次拆包的验证与状态

实现与本机验证完成：

- 14个相关测试文件分组运行，去重后287项通过，1项因未指定独立diffusion真实cache样本而跳过。[测试记录](validation/unit_tests.json)。
- 从Git基线独立加载旧vae与旧blocks，8组小尺寸配置逐一比较：Full/Light/Exp、joint、原refinement、conditioned、UV面encoder和本次UV边界辅助头。初始化state key/权重/RNG、encode/sample/decode、loss、导出及2,310个参数梯度均完全相同（rtol=atol=0）。还将原本零初始化的反馈/conditioning末层设为相同非零值再比较；旧完整模型pickle可加载并保持相同推理。[等价性记录](validation/equivalence.json)。小尺寸检查不代表各模型默认容量的性能测试。
- 完整已采用的conditioned checkpoint成功初始化UV模型，严格保留736个旧state项；两件真实validation对象通过FP32/BF16前反向，527个梯度张量有限，UV/refiner获得梯度，encoder/posterior保持冻结；无optimizer step。[真实权重检查](validation/real_checkpoint_smoke.json)。
- 本次未修改config、数据生成代码或标签格式，因此当前142上的标签生成与已生成sidecar合同不受这次整理影响。研究效果沿用原实验记录，不因代码拆包宣称提升。
