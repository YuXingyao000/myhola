# 恢复32维显式共享曲线基线

## 决定

用户在UV监督v2的51,460步评估后要求回到改变曲线输出表示之前的32维版本。选定 **shared_curve_uv best42471**，作为后续研究起点。当前UV训练保持停止；保留全部实验与未提交工作，不回滚Git，不启动新训练。

本次结果支持暂停当前双侧UV生成XYZ的路线；尚不能证明UV表示本身不可行，也不能将退化归因于64维latent。历史64维显式XYZ模型合法率90.96%，与本32维90.99%基本持平，原始几何误差还更低。选择32维是回到已有完整评估的研究基线，不是32维优于64维的因果结论。

## 固定身份

- 模型：`AutoEncoder_intersection`，每面latent32维；配置`latent_channels=8`对应实际32维，不是8维。
- 权重：`/mnt/d/python/checkpoints/shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt`（142）。
- 本地另一份同SHA资产：`/mnt/d/python_curve_eval/checkpoints/shared_curve_uv/candidate/20261019T064906.049992Z/weights/epoch=0116-step=00042471-val_loss=0.036517.ckpt`。
- SHA256：`ce749416c1635dda1563161b971bd4260d064cfdb0c31f1fa9d06ad0e70961ff`。
- epoch116，global_step42471，验证loss选择的历史best。
- 保留legacy Intersection、refinement、curve conditioning、shared-curve与辅助UV头；不使用新的UV Intersection decoder。
- 一对面只输出一条XYZ曲线，反向半边反转采样顺序引用它；UV是辅助输出。导出表示为`legacy_half_edges`，沿用已有OCC后处理接口。

历史训练冻结encoder/posterior，`feature_weight=1.0`。这描述权重来源，不是未来训练安排。后续若做全量训练、关闭feature loss或加入新监督，应显式建新配置并记录；不能改完后仍称完全恢复原训练。

不要混用后来的32D全解冻15k对照（SHA256 `61ac9804d3d4456363bc79ef9638348c2548ec93fbf66268b50c2e09224800e5`）；它尚未完成同协议OCC验证。

## 已有评估与本次核验

固定3495件validation，原始面CD² `2.633865e-5`、边CD² `5.969158e-5`，原生合法solid **3180/3495（90.99%）**。这些为历史正式评估结果，本次没有重新跑全量OCC。详细协议与失败分母见 [32/64对照报告](../../latent_expansion/no_edge_feature_20261023/occ_evaluation_20261024/REPORT.md)。

本次重新核对142与本地权重摘要；142以当前代码严格加载，并读取validation前两件真实CAD进行posterior-mean推理。核对32维latent、有限几何输出和反向半边共享XYZ；不执行优化器更新，不读取GT UV辅助标签。结果见 [VERIFICATION.json](VERIFICATION.json)。两件检查仅确认旧模型可恢复使用，不是新的研究效果评估。

加载时使用该checkpoint中的完整`hyper_parameters.model`，清除历史`init_*_checkpoint`后严格加载完整state dict；不得把64维模型仅改latent数字，也不得用UV模型的权重宽松加载。后续完整续训需明确选择是否继承Adam和冻结策略，本次未作该选择。
