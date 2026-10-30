# 2026-10-27：在生成式增强之后使用 DeepLPF

**结论：本批六图不建议再默认接 DeepLPF 或平滑增益投影。**
换成已经具有摄影纹理的输入后，官方修图依然主要改变冷暖和反差，并引入黑色截断；
投影更温和，但实际看完六张全图与目标局部，没有明确照片化收益。
这不是“所有轻量修图均无效”，也不是对白平衡美学偏好的普遍判断。

## 输入与输出

规格在运行前冻结于 [REFINEMENT_SPEC.md](REFINEMENT_SPEC.md)。固定六图输入为
`/mnt/d/data/cad_scene_enhancements/office_phone_depth_office_surfaces_20261027_001/`
各件 `strength_0.65/protected.png`；不挑图、不改参数、不继续扫 mix。
官方 checkpoint 与源码版本和上一轮相同，见 [NOTE.md](NOTE.md)。

新结果目录：`/mnt/d/data/cad_scene_enhancements/office_phone_deeplpf_refinement_20261027_001`。
12/12 输出，6件各有官方 `deeplpf.png` 和受限 `local_gain.png`。
`refinement_comparison.jpg` 左中右分别为 **当前 enhanced 输入／官方修图／受限增益**，
`target_comparison.jpg` 为同区域放大，原图不被覆写。
完整六件路径见 [refinement_manifest.json](refinement_manifest.json)。

## 实际观察

| CAD | 官方修图 | 受限增益 |
|---|---|---|
| 00301161 | 桌面/CAD偏黄、墙偏冷；槽内暗部截断，均匀材质未改善。 | 暗部更接近输入，仍无明确真实感增益。 |
| 00489180 | 细槽保留，圆盘仍平；黑设备更黑。 | 几乎只调色，没有反射层次收益。 |
| 00495463 | 打印机与屏幕黑边明显加重；薄板仍平。 | 中间调变暗，未补材质细节。 |
| 00182999 | 地面/目标大幅提亮、柜底更黑；原来的不自然接触仍在。 | 也明显提亮，无法修复接触和目标平片感。 |
| 00269315 | 冷墙暖桌更强、机箱大块纯黑；两端孔仍在。 | 较温和，仍没有材质改善。 |
| 00320493 | 黑设备加重，平板仍是均匀色面。 | 轻微影调变化；无内部孔，不冒称验证了孔保持。 |

六件完整记录见 [refinement_visual_review.json](refinement_visual_review.json)。
未观察到修图新增明显几何结构，但这不是几何保证；两路约100%的 CAD 像素都发生颜色变化。
本轮没有把 enhanced 图既有的浮空/接触伪影归因于修图。

## 描述统计与代价

- 输入六张均无 RGB 三通道全0的像素；官方结果有 **0.483%–2.335%** 全黑像素。
- 00301161 的 CAD 内，官方结果有 **0.815%** 像素全黑，任一通道归0为 **7.172%**。
  槽仍可见，但原暗部渐变被压缩，不能将高反差等同可辨性改善。
- 平滑投影没有新增全黑像素；两路本批均没有任一通道达到255的像素，未见高光硬截断。
- 00182999 全图显示域亮度中位数由153.9变为193.6（官方）/192.4（投影）。
  曝光变化明显不代表摄影质量更好；没有灰卡/真实照明参照，不能声称白平衡更正确。
- 六次官方 CPU 推理累计 **1.759秒**；六次投影累计 **1.612秒**，4线程。
  整个运行命令约8.30秒，含加载、保存和首轮制图。评估/新增局部制图另约3.39秒。

输入/hash/尺寸通过，保存增益独立重算输出误差≤1灰度级；见
[refinement_validation.json](refinement_validation.json) 和
[refinement_metrics.json](refinement_metrics.json)。边缘梯度比仅作定位，不用于验收。
没有GPU、模型下载、依赖安装、训练或逐图LLM调用；未改共享生成代码。

## 复现

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
/mnt/d/anaconda3/envs/generate/bin/python \
 /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/enhancement/phone_retouch_local/run.py \
 --samples /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/enhancement/phone_retouch_local/refinement_samples.json \
 --model_dir /mnt/d/model/DeepLPF \
 --output_dir /mnt/d/data/cad_scene_enhancements/office_phone_deeplpf_refinement_NEW_RUN
```

对新输出目录运行同目录 `refinement_metrics.py --output_dir <新目录>` 可重算描述统计和并列图。
保留当前 enhanced 为本次对照输入，不启用此后处理作为默认生产步骤；最终人工审阅仍 pending。
