# 第三实现切片：Blender 相机

## Task

只实现 `blender/render.py` 中的 `configure_camera()`，把已通过审阅的数学相机落实到 Blender。先验证坐标、朝向与投影一致，再继续场景装配。

## Assumptions

输入沿用[几何内核](GEOMETRY_CORE.md)的有效数值合同。调用者提供当前 scene 内专用、无父级／约束／动画的相机对象；函数配置并复用它，不创建对象。既有 Blender 4.2.0 可用于本轮验证，无需安装。归一化范围与实际零件尺寸仍未确定，本轮不依赖这些值。

## Inputs

`configure_camera(scene, camera_object, profile, *, near_clip, far_clip)`：场景、已有相机对象、`PerspectiveCamera`、与世界坐标同单位的显式近远裁剪距离。

## Outputs

原地配置相机及场景输出画幅，设置 `scene.camera`；返回 `None`。不移动 CAD 或环境，不拍照、不写文件。

## Processing Protocol

1. 由 position、target、up_reference 建立 right、up、forward。Blender 相机局部轴为 right、up、−forward，即局部 −Z 朝前。
2. 设置世界变换、透视模式与垂直传感器适配；取固定传感器高度 32 mm，用 `lens = sensor_height / (2*tan(vertical_fov/2))` 保证垂直 FOV。
3. 使用方形像素、零镜头偏移、100% 输出比例，关闭景深与局部裁剪。显式设置近远裁剪和输出尺寸，避免继承已有相机／场景的拍摄状态。

## Metrics

比较 Blender `world_to_camera_view()` 与 `geometry.project_point()` 的像素坐标及正向深度，检查方幅、横幅、竖幅和非默认 up_reference。它验证相机适配，不评价图像真实感。

## Decision Rule

测试点的每轴投影误差不超过 `0.001` 像素、深度误差不超过 `1e-5` 世界单位；此容差用于 Blender 单精度数值核对。固定属性、重复调用复用对象及目标变换不变也需通过。

## Edge Cases

零视线或平行 up_reference 无法定义朝向，抛出 ValueError；内部不逐项验证类型或参数范围。相机对象所有权、约束及动画由后续装配器保证；不静默清除用户场景对象。当前没有文件解析、批次或失败重试。

## Validation / Sanity Checks

独立 Blender 测试放在 `tests/blender/`，使用真实 bpy 与 Blender 投影函数，避免普通 Python 测试被 bpy 依赖阻断。本轮不渲染图片，不将相机测试称为完整管线验证。

2026-10-13：Blender 4.2.0（build `a51f293548ad`）下 4 项集成检查全部通过；Python 3.12.7 下原有 20 项纯数学测试全部通过。实际使用已有程序 `/mnt/d/natural/blender-4.2.0-linux-x64/blender`，未安装新依赖。设置自己机器上的 `BLENDER_BIN` 后，在项目根目录运行：

```bash
PYTHONPATH=packages/cad_scene_factory/src "$BLENDER_BIN" \
  --background --factory-startup --python-use-system-env --python-exit-code 1 \
  --python packages/cad_scene_factory/tests/blender/test_camera.py

PYTHONPATH=packages/cad_scene_factory/src python -m unittest discover \
  -s packages/cad_scene_factory/tests -v
```

`--factory-startup` 使用默认场景，`--python-use-system-env` 让 Blender 读取显式 `PYTHONPATH`；`--python-exit-code 1` 保证脚本错误反映到退出码。测试不保存或改写任何现有 `.blend` 文件。

## Implementation Constraints

仅新增一个功能文件与一个 Blender 测试文件，不修改已通过的数学模块，不实现 CAD 导入、场景、材质、进程调度或 Codex 调用。读取顺序按变换、镜头、画幅组织，不加通用配置层。

## Deliverables

[render.py](../../../packages/cad_scene_factory/src/cad_scene_factory/blender/render.py)、[test_camera.py](../../../packages/cad_scene_factory/tests/blender/test_camera.py)及本说明。2026-10-13 用户确认三个已有模块审阅通过。
