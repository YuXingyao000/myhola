# 第一实现切片：坐标、支撑高度与保守入镜检查

## Task

实现一个可以独立阅读的纯 Python 几何内核。已知条件来自[模板协议](TEMPLATE_CONTRACT_V0.md)：CAD 与相机保持不动，外围程序按当前 CAD 最低点平移整个环境。本轮只计算该变换和包围盒投影，不执行 Blender、STEP 转换或模型调用。

## Assumptions

- 数据：输入是当前姿态下、与相机同一世界坐标及单位的实际几何紧包围盒；本层无法验证这个来源条件。
- 代码定位：内部实验计算，信任调用者提供有效数值、三维向量、有序包围盒和合法相机参数。本层不重复检查类型、维数、NaN／Inf 或参数范围，不做输入强转；这些由后续数据准备入口负责。
- 坐标：右手系、Z 向上。透视相机使用 look-at 和向上参考；数学相机的正深度表示镜头前方，不能直接当作 Blender 相机局部 Z。
- 拍摄：默认位置 `(2,2,2)`、目标原点、向上参考 `(0,0,1)`、垂直 FOV `45°`、`512×512`。方幅下水平 FOV 也为 `45°`；非方幅保持垂直 FOV，方形像素、零镜头偏移。
- 边界：正式归一化范围和物理单位仍未确定，不采用旧最长边 `1.8`，不重缩放输入。近／远裁剪距离由调用者显式提供；测试中的距离不是生产预设。

## Inputs

- `BoundingBox(minimum, maximum)`：世界坐标的三个轴向最小值与最大值。
- `PerspectiveCamera(...)`：位置、目标、向上参考、垂直 FOV、图像宽高。
- `check_bbox_framing(..., near_clip=..., far_clip=..., margin_pixels=0)`：沿相机前向轴测量的裁剪距离，以及四边相同的连续像素余量。

## Outputs

- `plan_support_placement()` 返回 `SupportPlacement`：环境根平移 `(0,0,h)` 和支撑局部目标包围盒，其中 `h=target_bounds.minimum[2]`。
- `project_point()` 返回连续像素坐标 `(u,v)` 和正向深度。左上角为 `(0,0)`，边界为 `[0,width]×[0,height]`，实际像素中心为 `(i+0.5,j+0.5)`。
- `check_bbox_framing()` 返回布尔值：是否能用整个包围盒认证画幅及裁剪范围。`False` 不等于实际 CAD 必然被裁切。

## Processing Protocol

1. 直接使用准备层给出的包围盒与相机参数，保留类型标注和坐标说明。
2. 使用当前 CAD 的 `h`，保持 CAD／相机世界坐标不动；支撑局部包围盒为世界包围盒减去 `(0,0,h)`。
3. 相机前向为 `normalize(target-position)`，右向为 `normalize(cross(forward,up_reference))`，上向为 `cross(right,forward)`。
4. 通过相机基投影点；用包围盒八角点检查凸视锥。先检查正向深度与裁剪范围，再检查画面边界，避免跨相机平面时错误地使用投影包围框。

## Metrics

本轮只验证数学坐标与边界行为，不计算真实感、接触率或数据集指标。

## Decision Rule

八角点均在声明的裁剪范围和带余量的画幅内才返回 `True`，边界包含在内。画幅边界采用 `1e-9` 像素绝对容差，近／远裁剪分别采用自身距离的 `1e-12` 相对容差。未通过保守检查时返回 `False`；不移动相机或缩放 CAD 使检查通过。

## Edge Cases

- 保留影响几何结果的退化判断：零长度视线／向上参考无法归一化；单位视线与单位向上参考的叉积模长不超过 `1e-12` 时无法确定稳定相机方向。错误在实际计算时抛出，不在数据类构造时重复检查。
- 独立投影点位于镜头平面或后方时报错；包围盒在后方或跨过镜头平面时入镜检查返回 `False`。
- 包围盒入镜不证明无外物遮挡；底部对齐不证明有限桌面上的实际接触、无孔洞、无穿插或力学稳定。
- 不读取文件、不处理批次，因此本层不存在文件缺失、失败分母或重试策略。

## Validation / Sanity Checks

使用标准库 `unittest` 检查投影、支撑变换、裁剪及相机几何退化。移除字符串、布尔值、缺维度、错误参数类型等通用防御测试。解析几何用例的预期值独立给出，不用同一实现生成真值。

简化后，Python 3.12.7 下保留的 11 项几何检查全部通过。尚未连接 Blender，检查范围仅限数学计算。工作区根目录执行：

```bash
PYTHONPATH=packages/cad_scene_factory/src python -m unittest discover -s packages/cad_scene_factory/tests -v
```

## Implementation Constraints

只有一个功能模块 `packages/cad_scene_factory/src/cad_scene_factory/geometry.py` 和一个对应测试文件；仅使用 Python 3.10+ 标准库。本轮通过显式 `PYTHONPATH` 运行，不创建其余业务模块、CLI、安装配置或空目录。`geometry.py` 是原代码方案的纯数学辅助模块，后续由 CAD 准备及 Blender 适配共同使用。

## Deliverables

本说明、[几何模块](../../../packages/cad_scene_factory/src/cad_scene_factory/geometry.py)及[单元检查](../../../packages/cad_scene_factory/tests/test_geometry.py)。验收范围限于计算内核；用户审阅后再逐层实现 Blender 适配与实际几何检查。

建议先阅读 `plan_support_placement()`，再读相机投影与入镜检查。例如世界包围盒 `(-0.4,-0.2,-0.7)→(0.3,0.5,0.6)` 得到环境平移 `(0,0,-0.7)`、支撑局部包围盒 `(-0.4,-0.2,0)→(0.3,0.5,1.3)`；其中高度 `1.3` 按浮点精度表示。该函数不会写文件或改变输入几何。
