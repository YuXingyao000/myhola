# 2026-10-27：输入几何与材质审计

**三件没有被导入或平滑操作压扁。薄片形状、既定机位和单一金属材质的环境反射，共同造成了低立体感。**

证据与复查代码：[input_material_audit.json](input_material_audit.json)、
[audit_input_material.py](audit_input_material.py)。脚本使用 PythonOCC 7.8.1.1 读取源 STEP，
再以 CPU Blender 5.2.2 打开实际 prepared_scene.blend，加载该次渲染冻结的 part.py。
没有渲染或改动输入。运行方式：

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /mnt/d/anaconda3/envs/torch/bin/python \
  /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/enhancement/phone_camera/audit_input_material.py
```

| 模型 | 源 STEP 对应物理尺寸（mm） | 当前机位下 Y 厚度投影 | 主要理想反射射线命中 |
|---|---|---|---|
| 00182999 十字 | 151.51 × 2.16 × 119.48 | 0.92 px | 同一块灰地毯 |
| 00301161 块 | 190.56 × 85.75 × 127.68 | 44.21 px | 大正面反射木桌，斜面反射墙 |
| 00269315 拱 | 174.37 × 12.91 × 66.74 | 7.15 px | 大正面反射木桌，上侧反射墙 |

源文件为 `/mnt/d/data/deepcad_v6/{model_id}/normalized_shape.step`；三件均为合法单实体。
实际 STL 来自历史 mesh.ply 转换，摘要符合冻结清单；导入后均无边界边或非流形边。
十字的全部面为平面，着色角法线与几何法线一致；块和拱主要 Y 向正面也没有异常平滑。
这些结果排除明显压扁与正面法线错误，但不等于完整三角化保真验收。

零件使用 Step 3 的统一程序材质：metallic=1、roughness=0.32、固定 Base Color，
没有法线纹理、粗糙度变化或各向异性；没有调用模板 build_material.py。
实际场景为 AgX、Medium High Contrast、曝光 0、gamma 1，无材质覆盖或合成调色。
灯光为两个 115 W 面光源、Sun 和天空环境。十字目标 RGB 中位数约 [131,136,133]，
不是高光截断抹掉了形体。它的 2.16 mm 厚度在当前 512 图中不足一个像素，
不能要求修图在保持几何的同时凭空恢复清晰侧面。

**提示词有实际材质语义不一致。** `krea_office_matte.md` 写的是 “Matte gray vinyl flooring”，
实际为 500 mm 方块地毯，材质名 Contract carpet tile，程序噪声 scale=230、
roughness≈0.91、bump strength≈0.42、distance≈0.0012、sheen≈0.15。
因此把地面平滑成塑胶质感，可能是在执行改材质要求，不能称为恢复原地毯真实感。
当前 spatial 机制比较继续使用同一提示词；今后校正为 carpet 时另存版本，不回改本轮记录。

反射检查只取六个较大可见三角形中心的理想镜面方向，未积分粗糙 BRDF 或分解真实光传输贡献；
它支持颜色来源解释，不能量化各光源贡献。厚度投影也仅取包围盒中心线。

后续上游优先考虑实际零件的机位可辨识度，以及制造表面的程序材质与反射构成。
不为增强立体感擅自加厚、增孔或修改 GT 几何。此前 LUT/diffusion 的失败依旧有效，
但应承认部分问题来自输入观测证据不足，不能把所有失败都解释成后期模型能力上限。
