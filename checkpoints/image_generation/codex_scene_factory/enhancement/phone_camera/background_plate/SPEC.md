# 有／无 CAD 成对捕获：执行规格

## Task

检验先增强无 CAD 背景、随后利用有／无 CAD 渲染差合成，能否减少由前景描述诱导的
新增结构和错误接触影。本步骤仅准备并捕获成对输入，不运行 diffusion 或合成算法。

## Inputs

沿用上一级 samples.json 的固定六件。逐件读取其真实源 request.json 和 metadata.json，
恢复 prepared_scene.blend、该件 placement 的场景移动、相机、CAD matrix_world、
STL、30° 平滑设置及零件材质。原有五件来自 pilot，00320493 来自 full100；不能仅凭
CAD ID 混用机位。GPU 仅允许外围明确提供的 GPU7；执行由 root 统一启动。

渲染沿用源 metadata 的 seed：前五件 8–12，00320493 为 17。
后者的 diffusion 对照 seed 仍为 13，分别记录，不能当成同一概念。

## Outputs

新目录 `/mnt/d/data/cad_scene_enhancements/office_phone_background_pairs_20261027_001`：

- request.json、冻结 render_pairs.py／SPEC.md、source/ 下复用源码及摘要。
- 每件 `<id>/with_part/{rgb.png,mask.png,depth.exr,radiance.exr}`。
- 每件 `<id>/empty/{rgb.png,mask.png,depth.exr,radiance.exr}`；empty mask 全零合法。
- 每件 metadata.json，逐件 pairs.jsonl，总 summary.json。失败留在六件分母中。

RGB 为 512×512、8 bit，保留原场景 AgX/曝光。depth 为原相机轴 Z、米制。
radiance 为 Render Layers Image 直接输出的 RGB 32 bit scene-linear EXR，save_as_render=false；
保留当前 Cycles 去噪，不经过显示映射。它不是对 PNG 作逆 gamma 的结果，也不是未经去噪的
无偏辐射估计；后续线性差分必须考虑去噪及 Monte Carlo 残差。

## Processing Protocol

1. 普通 Python 执行 --prepare_only：冻结六件真实来源、配置、代码、输入和变换摘要。
2. Blender 入口复用冻结 Step3 prepare_placement，恢复场景与移动，再用源 metadata 的
   相机矩阵和 CAD 矩阵精确覆盖。零件材质复用 make_material，导入复用 load_part。
3. 每件 with_part 和 empty 均使用 256 samples、同一 seed、固定采样、相同相机与色彩管理。
   with_part 后只移除本件 CAD，场景中其他物体、材质和灯光不改，随后渲染 empty。
4. 复用 prepare_placement 的 mask/depth 输出，另加线性 radiance 输出。不能调用会拒绝空
   mask 的 capture.take_photo；独立保存函数允许 empty 全零 mask。
5. 保存每个输出摘要、渲染设备、颜色管理、相机/模型矩阵核对及源图比较。

## Metrics / Decision Rule

核对所有输入摘要；有 CAD mask 非空、empty mask 全零；新旧目标 mask 的相等性、IoU、
差异像素数；新旧 RGB 的全图/目标/背景 MAE、99 分位及最大差。
RGB 比较读取 PNG 数值通道，不把 scene-linear 差当作显示 RGB 差。
新 256 samples 与旧 64 samples、旧 CPU／新 OptiX 允许有采样/去噪差异；不编造统一阈值
将视觉差异标为已验收。summary 的 success 仅表示成对文件与基本几何检查完成。
新旧 mask 不一致时明确记录 review_required，不静默把新图当成原图的逐像素替代。

## Edge Cases / Constraints

不覆盖旧目录或修改任何原输入，不改变生产 Step3、源场景、几何与提示词。
一个逐件异常边界保留失败和已产出文件；不自动重试或换件。
仅执行源码语法、冻结来源、导入和 compositor 接口准备检查，不在准备阶段启动 GPU 渲染。
理想光传输差并不保证经背景编辑后仍物理正确；本步骤不宣称最终真实感或几何合格。

## Invocation

准备请求由本次代码任务完成。之后只需下面这一条 Blender 命令：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 PYTHONDONTWRITEBYTECODE=1 \
  /mnt/d/natural/blender-5.2.2-linux-x64/blender --background --factory-startup --threads 8 \
  --python /mnt/d/data/cad_scene_enhancements/office_phone_background_pairs_20261027_001/render_pairs.py \
  -- --request /mnt/d/data/cad_scene_enhancements/office_phone_background_pairs_20261027_001/request.json
```
