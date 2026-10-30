# Spatial release：执行规格（2026-10-27）

## Task

在现有 Krea + Depth 的一次采样中，分别控制背景与 CAD 表面的编辑起点。检验能否减少目标附近新增底座，同时让 CAD 表面获得外观变化；本规格在实跑前冻结。

## Assumptions

事实：旧 512 matte guard 的强背景重画有部分照片收益，但 `00495463` 新增棱条、`00182999` 新增底座；原 CAD RGB 回贴使材质始终不变。

假设：扩大并固定目标附近支撑面、孔隙和接触区域，可减少邻接结构幻觉；仅在低噪阶段释放 CAD 面内部，可能允许材质变化而减少轮廓改动。这是研究假设，不是几何保证。仍只用当前同一办公室的固定六件开发样本。

## Inputs

- 同尺寸 RGB 和二值目标 mask，本轮为 512×512；相机、深度、seed 及原 matte 配置不变。
- `halo_fraction=0.4`、`halo_min_px=24`、`feather_px=16`。
- CAD 内缩 `boundary_px=4`，内部 RGB Canny 边缘阈值固定为 40/100、保护半径 `cad_edge_px=3`，内部合成羽化 4 px。
- `interior_release_sigma=0.3`：使用调度器的真实 flow sigma，不将其称为 img2img strength。

## Outputs

`build_spatial_plan(rgb, mask, config)` 返回 PIL masks、摘要和内部 latent 网格。`make_callback(pipe, plan)` 返回回调及逐步日志。外围保存三种图：模型 raw、全 CAD/周边原像素合成的 strict、仅开放 CAD 面内部的 material 候选。

## Processing Protocol

1. 求目标凸包，保护其内部全部孔隙。外扩 `max(24, ceil(0.4 × bbox最长边))`，覆盖目标周边支撑面与接触区域。
2. 初始仅背景可编辑。CAD 可释放区域取轮廓内缩区域，排除内部强 RGB 边缘附近。RGB 边缘只是保守代理，不能等同真实几何边界；不冻结整张图的纹理和阴影。
3. 对背景和 CAD 内部可编辑区域分别做 8×8 min pooling：一个 VAE latent 单元全部落在可编辑区内才开放。再按 FLUX 的 2×2 patch 规则打包。薄件可无内部单元，此时明确记录，不强行释放。
4. 本地 `FluxControlNetInpaintPipeline` 的顺序是：Euler step → 按当前 mask 回填同一噪声等级的原图 latent → callback。回调在第 i 步结束时只设置下一步的 mask；读取 `scheduler.sigmas[scheduler.step_index]`，当该下一步输入 sigma ≤ 0.3 时，后续步骤开放 CAD 内部。首步 mask 中 CAD 必须全黑。背景持续编辑，周边/孔隙/边界持续回填。
5. 严格候选用扩大后的保护区恢复原像素。材质候选仅额外放开已释放的 CAD 内部，保留孔隙、边界和周边原像素；不能把它与 strict 的像素保证混为一谈。

## Metrics

记录 halo、保护面积、CAD 内部像素与 latent 单元数、每步当前/下一 sigma、实际首次开放步数和开放次数。照片收益、邻接新增结构与材质协调性由看图判断。边缘指标只用于定位，不能判定几何安全。

## Decision Rule

固定六件全部保留。若没有出现可辨认的材质改善，或邻接结构只是移动到保护圈外，不宣称成功；若薄件无可编辑内部，则该件材质更新未实施。仅工程检查通过不意味着照片质量或配对关系通过。

## Edge Cases

RGB/mask 尺寸不符、空或全图 mask 直接报错；原始文件/中间数据错误由外围按失败记录，不静默换样。空 CAD 内部合法，保持不编辑。没有可编辑背景时直接报错。初始 sigma 已低于释放阈值仍保持首步 CAD 锁定；从第二步才允许开放。最后一轮 callback 不会产生新采样步骤，不计为已编辑。孔槽内部未被 Canny 检出、保护圈外新增结构、材质内新增孔纹和 VAE 跨单元影响均不能排除。

## Validation / Sanity Checks

CPU 验证 min pooling、FLUX pack/unpack 次序、首步锁定、下一步释放时序、固定区始终不开放、薄件无内部单元，以及六件的 mask 面积。使用本地真实 scheduler 的 sigma 序列模拟回调；不加载权重或使用 GPU。

已完成 CPU 验证：40 步时间表、全局 strength 0.65，实际 26 步，初始 sigma 0.777133。第 18 步回调读到下一 sigma 0.284836，实际第 19 步起开放内部，共 7 步。官方 pack/unpack 逐格一致，初始 CAD 全锁定，所有固定区在全过程未开放；2 px 薄片保持无可编辑内部，空 mask 报错。

| CAD | halo / px | 可编辑内部 latent 单元 | 可编辑 CAD 像素占比 |
|---|---:|---:|---:|
| 00301161 | 60 | 71 | 39.84% |
| 00489180 | 50 | 125 | 70.22% |
| 00495463 | 38 | 7 | 20.43% |
| 00182999 | 38 | 0 | 0% |
| 00269315 | 41 | 0 | 0% |
| 00320493 | 46 | 5 | 14.36% |

两件薄结构没有完整内部 latent 单元，是当前方法的明确限制，不修改协议来强行开放；六件仍全部保留在实跑分母中。

## Implementation Constraints

只新增独立 `spatial_sampling.py` 与本规格；共享模型接口和实跑由外围完成。不新增模型权重、不修改已有图。接口采用固定单图、8× VAE 和 2×2 patch 约定，少量存在性/尺寸检查，不写重试框架。

## Deliverables

代码、CPU 验证摘要和六件诊断 mask（后两者由外围 run 保存）。机制依据为 [Differential Diffusion 原论文](https://arxiv.org/abs/2306.00950)；采用本地 [Diffusers ControlNet inpaint 的 mask callback](https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/flux/pipeline_flux_controlnet_inpainting.py)。[官方仓库的 FLUX differential 示例](https://github.com/huggingface/diffusers/blob/main/examples/community/pipeline_flux_differential_img2img.py)说明逐区域编辑可用于 FLUX，本实现沿用现有 Depth pipeline，仅借鉴分阶段释放机制，不声称完整复现论文。
