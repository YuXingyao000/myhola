# Step 3：批量拍摄 STL 零件

读取 Step 2 的 `ready.json`，把灰方体替换成 STL 零件，逐件渲染。
这里没有 Agent 或 FLUX 调用；普通 Python 启动一个 Blender 进程完成拍摄。

## 从哪里读代码

| 文件 / 函数 | 做什么 |
|---|---|
| `workflow.py → run_workflow()` | 抽样 → 保存请求 → 启动 Blender → 汇总 |
| `capture.py → capture_batch()` | 每个机位加载一次场景，循环换零件、拍照、移除零件 |
| `capture.py → prepare_placement()` | 打开原场景，恢复 Step 2 的物件位置与相机 |
| `part.py` | 导入 STL、等比缩放、底部落到支撑面、程序化 PBR 材质 |
| `config.json` | split、随机种子、采样数、CPU 线程、零件尺寸与材质 |

实际主循环位于 `capture_batch()`。同一个机位拍多件时，不重复加载场景；
切换机位时重新打开基础场景，避免上一个位置的整理动作累积。
复用 Step 2 的摄影配置、相机构造和世界矩阵设置；支撑面材质已由 Step 2 准备。

输入必须是含 `support_validation` 的新版 ready，指向 Step 2 准备并验收的场景副本。
旧 ready 缺少实际渲染表面验收时会报错；不能仅凭网格高度继续批量拍摄。
每件摆放后核对实际最低点与该位置的渲染表面中位高度，距离及判断写入 `contact`。

## 首版拍摄约定

- 默认从 **training** 中随机选 100 个不同 CAD ID，每件一张，不是 100×机位数。
  同一个 STL 字节摘要只选一次；保留原 manifest 的 split 和来源。
  这不是几何等价去重，也没有修复源数据跨 split 重复。
- 样本均衡分配到已有位置。默认最长边 15–20 cm，必须不超过 Step 2 的 20 cm 代理。
- 保留 STL 的 XYZ 姿态，只做等比缩放和平移：包围盒 XY 中心对齐支撑点，Z 最低点
  对齐支撑高度。尖点支撑、不稳定姿态允许存在，不模拟重心或重力。
- 机位、FOV、分辨率均继承 Step 2。薄片可能比方体显得小或靠下，首版不自动重构图。
- 固定缎面金属 Principled BSDF，无贴图。30° smooth-by-angle 只改变着色法线，
  不加倒角、不修改三角形；它不等于 CAD 的精确曲面法线，粗三角化轮廓也不会变圆。
- 使用原场景的灯光、世界、合成与色彩管理。默认 CPU 模式，8 线程、64 samples；
  可显式选择一张 OptiX GPU 做路径追踪。降噪与合成仍在 CPU，其他 GPU 对 Blender 隐藏。

## 运行

```bash
python packages/cad_scene_factory/step3/workflow.py \
  --ready_file /mnt/d/data/cad_scene_placements/office_step2_support_fix_20261026_001/ready.json \
  --stl_dir /mnt/d/data/deepcad_stl_total \
  --output_dir /mnt/d/data/cad_scene_renders/office_parts_001 \
  --blender /mnt/d/natural/blender-5.2.2-linux-x64/blender \
  --count 100
```

少量检查用 `--count 5`。改材质或尺寸时复制 `config.json`，用 `--config` 指定。
每次使用新输出目录；当前没有恢复、覆盖、自动重试或补抽。

使用物理编号为 7 的显卡时添加 `--device OPTIX --gpu_index 7`。GPU 编号按 PCI 总线顺序，
与本服务器 `nvidia-smi` 编号一致。请求记录所选编号，每件 `metadata.json` 的
`render_device` 记录实际设备类型、名称和 Cycles 设备 ID。GPU 改变可能带来浮点渲染差异，
不会改变冻结的 CAD 抽样、机位、尺寸或 seed。

## 输出及后续 FLUX 接口

```text
office_parts_001/
  request.json          固定样本/源 manifest 行、位置、seed、尺寸、配置与摘要
  source/               本次实际执行的源码副本（含未提交文件）
  git.diff              现有受版本管理文件的差异；新增源码由 source/ 保存
  blender_command.json
  blender.log
  renders.jsonl         每件结果；失败也写入
  summary.json          planned / success / failed / not_run
  overview.jpg          按原抽样顺序拼图，不删除失败位置
  00301161/
    rgb.png             合成渲染照片，8-bit RGB
    mask.png            可见零件二值掩码，非抗锯齿 Object Index
    depth.exr           全场景浮点深度，单位米
    metadata.json       CAD 来源、变换、机位、材质、图像路径及检查结果
```

RGB、mask、depth 来自同一次渲染。`mask.png` 不包括零件投下的阴影或在周围物体中的
反射，也不是隐藏面的完整轮廓。本机 Blender 5.2.2 的 `depth.exr` 已通过真实网格
射线核对：数值是首个表面在相机前向轴的 Z，单位米，不是到相机的欧氏距离。
背景可为远大值，使用时结合 mask / 有效范围。
未来 FLUX 可读取 RGB、mask 和深度；这一版没有编辑、重采样或生成图验收。

`metadata.json` 中 `geometry.model_to_world` 把源 STL 坐标变到场景米制坐标。
相机同时保存 Blender 的 camera-to-world、CV 的 world-to-camera（X 右/Y 下/Z 前）
及内参 K。像素坐标以图像左上角为原点，像素中心为 `(列+0.5, 行+0.5)`。
源数据当前主要来自已准备好的 PLY→STL，不宣称这一步重新执行 STEP 三角化。

## 通过意味着什么

技术通过要求零件包含于原代理、底端与已验收渲染表面齐平、包围盒完整入镜、有可见掩码。沿用 Step 2 的
支撑采样、碰撞与遮挡近似及接触带容差，不重新证明每个零件的精确接触或孔槽可辨。
透明遮挡、粗三角化、极薄件和视觉质量仍需人工判断。

逐件仅保留一个异常捕获，坏样本记失败后继续；Blender 整体崩溃时剩余样本记
`not_run`。总分母是请求清单，不能把仅成功子集当作全量完成。

实验约定与验证见 `checkpoints/image_generation/codex_scene_factory/batch_capture/`。
