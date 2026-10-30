# CPU 摄影修图对照

入口是 `workflow.py`：读取固定图片 → 小 CNN 预测三张 LUT 的权重 → RGB 查表 → 与原图混合 → 保存。
`model.py` 实现模型，`validate.py` 核对官方分类器和三线性插值公式。

该分支使用 [Image-Adaptive-3DLUT](https://github.com/HuiZeng/Image-Adaptive-3DLUT) 的
paired / unpaired **sRGB** 预训练权重；不是 HDRNet，也不是 iPhone ISP 复现。
模型约59万参数，无 diffusion、无 GPU、无逐图提示词，读取当前已有 Torch/Pillow/NumPy 环境。
代码中的分类器结构改编自该项目，官方 Apache-2.0 许可随 `LICENSE` 保留。

输入为 JSON 列表，每项至少有 `id` 和 `rgb`（512×512 RGB PNG 的绝对路径）。
其他 CAD/相机来源信息可以作为额外字段保留。程序保存冻结请求、源码、摘要、逐件结果和对照图。
不覆盖原图或旧运行；固定输出 paired/unpaired × mix0.5/1.0。`mix` 只是原图和模型结果的 RGB 混合强度。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
/mnt/d/anaconda3/envs/generate/bin/python \
  /mnt/d/python_room_generation/packages/cad_scene_factory/step4/retouch/workflow.py \
  --inputs /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/enhancement/phone_retouch/inputs.json \
  --output_dir /mnt/d/data/cad_scene_enhancements/office_phone_lut_NEW_RUN \
  --model_dir /mnt/d/model/Image_Adaptive_3DLUT \
  --threads 4
```

对整幅图调色，**CAD 颜色也会改变**。查表不移动像素，但压黑/过曝仍可能使孔槽不可辨；
不能把“无空间变形”解释成“所有几何证据必定保留”。不重复给已有显示变换的 PNG 做 AgX。

首轮6张输入得到24张结果，已逐图检查。主要变化为对比度和暖色增强，整体 CG 感没有显著解决；
mix1在若干金属/木桌上增加黄调且压暗黑色道具，不推荐直接作为“手机照片”增强结果。
详细结果见 [研究记录](../../../../checkpoints/image_generation/codex_scene_factory/enhancement/phone_retouch/NOTE.md)。
