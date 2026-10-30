# 2026-10-27：空间局部修图的实际边界

补充：在已经通过 diffusion 摄影化的图上再次运行六件修图，仍无明确收益，官方版增加黑色截断。
该独立后处理实验见 [REFINEMENT_NOTE.md](REFINEMENT_NOTE.md)，不覆盖下述原CG输入实验。

## 结果

**DeepLPF原模型与受限局部光色投影均已实跑并看完六张，未取得明显的照片化收益。**
原模型主要增加冷暖对比、压暗黑色设备、提亮地面；木桌和金属仍显得像CG。
投影保留原像素结构并温和调整局部颜色，但不能改善原材质、反射和道具形态。
不继续扫描参数、不扩100张；这一结果不能推导为所有小模型无效，但说明通用FiveK修图预训练不能直接解决本批问题。

这次确实具备空间变化能力，并非重复全局LUT：官方渐变与椭圆滤镜非恒定；
均匀0.5灰图输入的输出在不同位置有差异，详见`spatial_validation.json`。
空间能力存在不等于会对合成渲染产生正确的摄影修正。

## 选择依据与版本

- [DeepLPF原论文](https://arxiv.org/abs/2003.13985)：预测空间局部的多项式、渐变和椭圆滤镜。
- [作者官方仓库](https://github.com/sjmoran/deeplpf-image-enhancement)：当前默认预训练仅6.9MB，可CPU运行；
  使用官方commit `4faeef6597f299d38acc597df9cfe970f8d90ca2`，1,715,779参数。
- 本次权重为官方2026维护版在**重建FiveK划分**上训练1000epochs的默认模型，
  `pretrained_models/adobe_dpe/deeplpf_psnr_24.180_ssim_0.917_model.pt`。
  不把它冒称为2020原论文checkpoint。训练是sRGB照片修图，不是render-to-real或iPhone ISP训练。
- [HDRNet官方实现](https://github.com/google/hdrnet)与[作者旧仓库](https://github.com/mgharbi/hdrnet)
  需要旧TensorFlow/自定义bilateral算子；HDR+权重还不能直接用于现有sRGB PNG。
  本轮选可直接运行且输入契约匹配的DeepLPF，未下载/部署HDRNet。

来源代码、MIT许可、权重和SHA保存在`/mnt/d/model/DeepLPF/source.json`与运行请求中。
官方model.py顶层训练loss导入间接要求Matplotlib，现有generate环境没有该库。
适配器只通过AST跳过`from losses import DeepLPFLoss`这一未用于推理的导入；
全部推理类/方法与权重保持原样，strict加载。未改环境、未安装依赖、未占GPU。

## 固定协议

代码是本目录`run.py`。原图清单沿用`../phone_camera/samples.json`，六件全部是training开发样本。
输出：`/mnt/d/data/cad_scene_enhancements/office_phone_deeplpf_20261027_001`。

每件保留两路：

1. `deeplpf.png`：官方网络输出；sRGB[0,1]输入，使用官方CLI的uint8截断规则。
2. `local_gain.png`：从该输出提取平滑局部RGB增益，再作用于**原图**；不是官方DeepLPF算法本身。
   原图与参考先转线性RGB，高斯sigma24px，增益为`(平滑参考+0.01)/(平滑原图+0.01)`并限制[0.5,2]。
   无逐图参数、无mix搜索、无空间位移。`gain.npy`与可视化保留。

DeepLPF含U-Net像素残差，不能因存在参数滤镜就宣称原网络严格保护结构。
投影只转移广域光色、不转移参考高频像素；仍可能改变孔槽对比，不能以“无位移”代替几何验收。
两路均允许CAD颜色改变，不回贴原CAD，也不宣称像素零误差。

## 实际观察

| CAD | 原模型 | 平滑投影 |
|---|---|---|
| 00301161 | 墙面偏冷、桌面与CAD偏黄，键盘和机箱黑色加重；CG表面未改善。 | 冷暖较温和，原金属反射不足仍在。 |
| 00489180 | 圆盘正面仍近乎平坦米色，桌缘偏黄，设备阴影加深。 | 无明显摄影化，原造型和材质观感基本保留。 |
| 00495463 | 打印机暗边加深，木纹更浓；细薄CAD仍不突出。 | 白墙/桌面影调稍变，未补真实材质层次。 |
| 00182999 | 地面提亮、局部反差改变；灰色十字依然像无反射层次的渲染片。 | 变化细微；无新增底座，但也无明显照片感提升。 |
| 00269315 | 墙面更冷、木桌更暖、机箱压黑；拱形CAD反射未改善。 | 较自然温和的色调变化，仍无足够照片化收益。 |
| 00320493 | 笔/便笺未被重画，黑设备压暗；桌纹与材质CG感仍在。 | 布局保持、变化偏弱。 |

以上是对全部12张的Agent观察，人工验收pending；没有将更大色差当质量提升。
未使用此前diffusion图拟合：当前收益主要来自被重画的纹理/道具细节，
单纯转移低频光色无法可靠继承这一收益，也不能把生成伪影当摄影目标。

## 验证与成本

`validation.json`：12/12输出尺寸/hash通过，六张顺序一致，源RGB摘要未变；投影由保存的增益场独立重算误差至多1个8-bit级别。
`projection_validation.json`：恒等参考产生恒等增益；空间变化参考产生非恒定增益，无采样坐标变化。
`spatial_validation.json`：均匀灰图测试中，graduated mask空间std约0.049–0.054，elliptical约0.005–0.013；
输出RGB空间std约0.006–0.010，证实该checkpoint具有实际的局部变化。

CPU4线程，6次网络推理累计 **1.698秒**（首张0.446秒，其余约0.24–0.26秒）；
6次平滑投影共 **1.627秒**。这些时间不含源码/权重加载、保存、制图。
没有训练、没有GPU使用、没有新增LLM调用。12张候选全部保留。

可以与新diffusion并列的六件路径和逐件时耗已整理到`comparison_manifest.json`，
字段为`model_id / original / official / projected / comparison / inference_seconds / projection_seconds`。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
/mnt/d/anaconda3/envs/generate/bin/python \
 /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/enhancement/phone_retouch_local/run.py \
 --samples /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/enhancement/phone_camera/samples.json \
 --model_dir /mnt/d/model/DeepLPF \
 --output_dir /mnt/d/data/cad_scene_enhancements/office_phone_deeplpf_NEW_RUN
```

本分支保持为“局部修图已验证可运行，但未解决当前照片质感”的证据，不作为合格训练数据生成配置。
