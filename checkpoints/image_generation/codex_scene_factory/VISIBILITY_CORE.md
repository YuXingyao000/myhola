# 第二实现切片：AABB 与相机到目标的保护区域

## Task

按用户确认的简化方案，只使用场景物体的世界坐标 AABB 检查保护区域，不做网格碰撞、射线采样或 Blender 集成。[几何内核](GEOMETRY_CORE.md)继续负责投影与完整入镜，本模块负责其他物体是否占用拍摄通道及目标包围盒。

## Assumptions

相机位置为 `C`，目标包围盒为 `B`。保护区域 `K=conv(C,B)` 是相机与目标八角围成的有限凸体，包含目标包围盒；目标后方背景可存在。目标包围盒三轴均有正长度；场景 AABB 可退化为面、线或点。内部数值与坐标输入可信，不重复添加类型、维数或参数范围检查。

## Inputs

`find_blockers(target_bounds, camera, scene_bounds)`：复用 `BoundingBox`、`PerspectiveCamera`；`scene_bounds` 是“对象名称 → 世界坐标 AABB”的映射，排除目标自身，包含支撑物。所有包围盒已应用场景实例变换，坐标及单位一致。

## Outputs

按输入顺序返回进入保护区域内部的对象名称列表；空列表表示检查通过。仅接触边界允许，不豁免整个支撑物。没有场景对象时返回空列表。

## Processing Protocol

保护区域也可写成一组从相机展开的盒子的并集：

```text
K = union over 0 <= t <= 1 of [C + t*(Bmin-C), C + t*(Bmax-C)]
```

对场景 AABB `A`，判断是否存在同一个 `0<t<1`，让三个轴同时满足：

```text
C + t*(Bmin-C) < Amax
C + t*(Bmax-C) > Amin
```

这六条一次不等式都写为 `slope*t < offset`，逐条收紧 `t` 的上下界即可。无需建立凸包面、碰撞树或新增几何库；检查的是整个体积，不是八条角点射线。

## Metrics

仅输出占用保护区域的对象，不评价真实感或力学稳定性。AABB 的保守误拒是当前接受的取舍，不进一步做网格精检。

## Decision Rule

最终区间满足 `t_min + 1e-12 < t_max` 时判为占用保护区域；`1e-12` 是无量纲参数的舍入容差，不是场景空间余量。相交区间小于该精度时按边界接触处理，不宣称精确实数运算。

## Edge Cases

中央遮挡、对象完全在区域内部、对象包住整个区域都需检出。内部平面／点仍算占用；支撑面贴合底部或其他物体仅接触侧边界允许。全局 AABB 相交但实际在斜侧面外时允许。输入文件解析和非法数据由上游处理，本层无重试或异常恢复。

## Validation / Sanity Checks

用解析位置验证中央遮挡、完全包含、后方背景、侧方物体、底部接触与穿入、退化平面、斜侧面接触、默认斜视相机及共同坐标变换。保留已有几何测试。

2026-10-13：Python 3.12.7 下新增 9 项及原有 11 项测试全部通过，共 20 项；尚未进行 Blender 集成验证。在项目根目录运行：

```bash
PYTHONPATH=packages/cad_scene_factory/src python -m unittest discover -s packages/cad_scene_factory/tests -v
```

## Implementation Constraints

仅新增一个功能模块 `visibility.py` 和一个测试文件。标准库实现，不修改已通过审阅的几何算法，不加入 Blender、CAD 导入、材质或模型调用。布景实例平移后用新的世界 AABB 重新调用；不跨不同实例缓存检查结果。

## Deliverables

[visibility.py](../../../packages/cad_scene_factory/src/cad_scene_factory/visibility.py)、[test_visibility.py](../../../packages/cad_scene_factory/tests/test_visibility.py)及本说明。2026-10-13 用户审阅通过。
