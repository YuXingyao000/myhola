# 每面 latent 32→64：容量扩展

2026-10-20。代码基准 `85f6fb33da269fd75d0f7352164ce1af3663be27`。
状态：实现与部署验证中，尚未启动正式训练。实现提交与验证见 README。

## 研究问题

在已经采用 refinement、curve conditioning、UV/shared-curve 的 VAE 中，
扩大早期几何特征和最终 face latent，是否改善边的几何精度和可裁剪性？
本轮改变容量，不增加 attention 层，不替换交互方式，不修改数据和后处理。

## 固定起点与对照

- 来源：shared-curve validation best，epoch116 / step42471；
  SHA256 `ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff`。
- 64维候选：`train_vae_latent_expansion`，`latent_channels=16`。
- 32维对照：`train_vae_latent_expansion_control`，`latent_channels=8`；
  直接载入相同完整来源，Encoder/posterior/Decoder 同样全部解冻。
- 两者均新建 run/optimizer，seed42、8卡×每卡16、lr2e-5、bf16、
  100000步上限、is_aug=1、相同 train/validation 清单。
- 用户自行启动前台训练。本次只执行实现检查，不后台启动正式实验。

## 改动边界

64维候选将面 CNN 最终通道8→16，展平后F32→64；edge CNN输出通道
32→64，长度2展平后E64→128。posterior及所有读取F/E的接口随之扩容。
CNN中间通道、16×16/16采样及池化顺序保持。面间encoder/decoder
attention各8层768维，inter8层512维，refiner128维，UV2层128维均保持。

迁移按实际向量拼接段复制输出通道；重复输入权重分成0.49/0.51，
初始重复输入的读出和旧模型相同，但两份通道可获得不同梯度。
GAT attention向量复制并除2，保持原有打分；不增加edge-value消息。
posterior最终行按交错的mean/logvar成对复制，不按mean/logvar两半处理。

这只保持posterior-mean计算的初始函数（允许浮点误差）。
训练时新增latent使用独立高斯噪声，重复坐标初始读出的有效噪声方差
在单层线性读出下约为原来的0.49²+0.51²=0.5002。
因此不能声称随机训练分布或训练loss初始化完全等价。

所有重建、classifier、incidence、twin、UV权重沿用。
KL保持sum：32维系数1e-6，64维5e-7，保持每个对象beta×D标度；
从此前冻结状态恢复KL梯度是两组共同变化。这个实验同时改变容量与
扩容初始化，不能用一次候选运行单独证明理论信息容量极限。

## 数据与判定

DeepCAD accepted：train46480，validation3495，test2417。
沿用UV sidecar v1及有效点mask，不重新生成标签。
规则分析、失败归因和训练选模使用validation；test不用于本轮调参。
训练完按预先固定的validation最佳与最后完整checkpoint导出，
由各模型自己的Encoder得到posterior mean。**禁止继续复用旧32维latent缓存。**

比较全部3495件的raw face/edge CD²、对应边面贴合、漏边/额外边、
固定后处理下的合法solid数、救回/退步；缺失和超时留在分母。
CD²与连续几何公差分开。原始几何与后处理后的几何分开；后者共同可测
集合和缺测数量单列。不按test或下游指标回头选checkpoint。

## 实现验收

1. 迁移检查完整key与维度，保留大主干权重；检查有学习后的非零refiner、
   conditioning、shared-fusion参数时仍能保持初始mean几何、logits与UV。
2. 真实validation小批量FP32/BF16前后向；Encoder/posterior/Decoder均有
   有限非零梯度，新增CNN通道与旧通道获得不同梯度。
3. shared半边XYZ/UV反序、纯latent推理、完整保存加载和resume清除初始化源。
4. 验证142上的实际环境、W&B认证、路径、部署hash和每卡16的显存。

源码仍用已有VAE主类、checkpoint加载模块和统一trainer；不新增网络主类，
不在forward里加入shape/type检查。完整层级结构保存在ARCHITECTURE.md。
