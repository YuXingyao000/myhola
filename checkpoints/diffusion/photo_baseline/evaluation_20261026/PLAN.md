# 照片 baseline 停训与固定验证比较

## Task
按用户要求停止训练、运行重建评价并展示数据。训练已在约 312,535 steps 收到 SIGINT；最近完整 checkpoint 为 310,000。不为等待下次保存继续训练，不假装保留未落盘尾部。

## Assumptions
这是开发阶段的 validation100 比较，不动保留 test。已有 FLUX 为合成真实感照片；单张图不唯一确定隐藏几何。源码事实：训练使用原版 concat 架构、原 AE、完整 RGB；目前不改变架构或后处理。默认三组：原 SVR、按训练期间最低 validation loss 选择的早期 step4553、固定最后完整 step310000。比较前冻结，不按重建成绩再挑 checkpoint。

## Inputs
`svr_comparison_20261017/validation100.txt`；`deepcad_v7_cond/{id}/real_photo.npz[flux]`；GT `deepcad_v7/{id}/{data.npz,normalized_shape.step}`；原 checkpoint 与本目录 `weights/{photo_val_best,photo_latest}.ckpt`。核对 train 不相交、照片与训练缓存 manifest 来源一致。

## Outputs
冻结 protocol / 输入和权重 SHA256、逐对象 raw latent/geometry、STEP 与终态记录、固定分母 CSV/JSON、中文数值表与 loss 曲线、运行笔记。

## Processing Protocol
每组每图只生成 1 个候选，seed42 按 CAD identity 派生，原始 1000-step DDPM、FP32、无 TF32、原 resize224/ImageNet 归一化。先严格核对所有新 checkpoint state keys/形状、冻结 DINO 与原始权重相同，只替换原网络五个去噪模块；VAE 仍来自原始 checkpoint。复用原 sampler / decoder，不读取 GT 指导生成或修复。GT 仅供评价。8 GPU 分片独立图片，CPU32 workers；三组各100，不丢失败。

## Metrics
复用现有 `eval.preference.evaluate_candidate`：raw symmetric squared CD（双向均值之和、未开方）、face F1（Hungarian、同一阈值0.1）、face-count error；`recover_v1` 后处理 drop_num0、几何优化200次、每对象整体180秒、独立 native solid validity；有效 solid 的 STEP 几何3次采样取均值。报告 raw 以及成功子集 post 几何、各自测量数和共同成功集合；成功/失败/超时以100为分母。照片不使用已知灰模相机投影分数冒充可见一致性。

## Decision Rule
固定样本、输入、seed、权重及后处理协议后判断照片适配是否同时改善几何与有效性。结果不自动触发新训练。早期 best 在相同 validation 集上被挑选，必须标注 selection bias，不能称独立 test 最优。不能仅凭随机 timestep 的 validation loss 认定重建退化。

## Edge Cases
空集、重复 ID、split 重叠、来源不符、缺字段、非有限输入或权重直接拒绝启动。逐件采样/解码失败保留100分母；OCC失败/timeout单列，不补采。无可计算几何时值为 null；仅成功子集分数不可当全体结果。

## Validation / Sanity Checks
检查训练所有进程退出/GPU释放。新权重映射验证缺key/冻结DINO改变拒绝，真实 checkpoint 前向与训练 wrapper一致；CPU后处理先做一个真实对象检查接口。正式结果每组100终态，seed/图片hash配对一致，重复运行不得覆盖冻结输入。

## Implementation Constraints
只新增照片评估编排与严格权重适配；保留已有其他任务未提交修改。复用原采样、解码和计量实现，不偷用此前灰模结果。不改后处理规则、阈值或数据split。记录源代码/运行环境。

## Deliverables
已停止的训练、完整固定验证结果与图表、明确下一步讨论所需证据和局限。若批量耗时，持续推进至完成，不只给启动命令。
