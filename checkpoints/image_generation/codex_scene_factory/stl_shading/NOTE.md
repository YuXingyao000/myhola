# 2026-10-16：workshop_workflow_001 视觉审阅阻塞

## 原因与状态

用户首轮工作流完成模板构建和 15/15 技术预览，随后 Codex 只读审阅返回 blocked。其明确指出四件曲面零件的分段高光：00843446、00999999、00000070、00719511。实际打开原图后，这些条带可见；当前工作流按约定把共享渲染器问题交回 host，因此停止有依据。

技术通过仅指当前已实现的构建、输入摘要、AABB 和图片检查，不包含曲面着色正确性。旧 `workflow_result.json` 保留 incomplete，不改写历史结论。

## 实施

- 旧 `_load_model()` 使用 `use_facet_normal=True`。本机 Blender 5.2.2 实测导入后的面全部为 flat，且存在自定义逐面法线；此前“默认自动平滑”的注释不准确。
- 新 STL 导入使用 `use_facet_normal=False`，将面设为 smooth，再以 30° 相邻面法线夹角设置硬边。只有着色法线及平滑/硬边属性变化，顶点、三角形和变换不动。OBJ 不套用这套规则。
- `render.py` 中集中定义阈值，并在每次渲染的 request.json 中记录 `stl_shading`。不改材质数值、灯光、相机、源 STL 或生成的模板。
- 收紧 ACCEPTANCE.md 与 inspect.md：remaining_issues 只保存当前交付前需要解决的问题。固定 checks_not_run 和人工 pending 默认是 summary 中的范围限制；若用户明确要求或缺失确实妨碍判断，才作为阻塞。观察到的 host 缺陷仍然阻塞。

Blender API 的行为与本地探针共同作为依据：[Mesh.set_sharp_from_angle](https://docs.blender.org/api/current/bpy.types.Mesh.html?highlight=transform)、[按夹角平滑说明](https://docs.blender.org/manual/en/latest/modeling/modifiers/normals/smooth_by_angle.html)。仅开启平滑而继续保留逐面自定义法线，不能当作等效修复。

## 实际预览

旧报告：`/mnt/d/data/cad_scene_templates/workshop_workflow_001/workflow/round_00/preview/preview_report.json`。

新报告与图片：`/mnt/d/data/cad_scene_workflow_validation/20261016_stl_shading/`。

同一模板重新构建、固定 12 件及三个空场景、512×512、256 samples、OptiX GPU 0、相同逐件 seed。新报告为 **15 planned / 15 success / 0 failed / 0 missing，technical pass**。模板源码/声明素材摘要、冻结测试组摘要及源 STL 摘要未变。

实际查看四件报告样本及总览：圆筒内外壁、高亮弯曲处和圆周槽周围的分段明暗明显减轻；六边孔和平面边界仍可辨。零件材质保持不变。本次没有新的 Codex 生成或审阅调用，也未宣称人工批准或照片真实感达标。

`normal_comparison.png` 是原图与新图的并排展示，没有修改图像内容。`comparison.json` 记录输入一致性。重渲染不逐像素完全一致：三个空场景 RGB 平均绝对差为约 0.000060–0.000069（0–255），最大差 1；这些数值仅说明本次对照的微小像素差异，不作为真实感指标。

工作流控制/证据 fixture 检查为 **24/24**，输出 `/tmp/cad-scene-workflow-validation-116gh6r2/results.json`；该检查没有调用模型或渲染器。真实 Blender 几何/法线检查为 **12/12**：顶点、三角形索引、变换、AABB 均未变，超过阈值的硬边保留。额外三角化立方体的六个平面及 12 条 90° 边保留；15° 设计折线则被平滑，明确证实此方法的局限。完整数值见同目录 `RESULTS_NORMALS.json`。平面保证仅针对由硬边隔开的共面连通组，不能推广到与浅角过渡相连的任意平面三角形。

## 局限与后续使用

STL 没有 B-Rep 面归属。30° 是显式网格近似，不能区分小夹角设计折线与平滑曲面的离散边，也不能改善离散轮廓。需要严格保留曲面/硬边语义时，应从原 STEP 导出面归属或解析法线，单纯调阈值不能提供这种保证。已有 15° 设计折线反例用于验证该限制。

无需重新生成当前布景；可直接打开新预览人工检查。原工作流记录及会话保留原样；当前入口还不支持对已有 workflow 目录直接续跑，重复原生成命令会按设计拒绝覆盖。后续正常新模板生成会使用新渲染器及更新的验收提示词。

## 同日补充：光照与可见布局的只读核对

用户仍认为光照假、构图空。本轮仅打开已保存的 scene.blend 检查，不改变灯光、模板或提示词，不渲染。证据与探针保存于新预览目录的 `lighting_layout_audit.json` 和 `audit_lighting_layout.py`。

实际启用两盏 Area：侧窗名义 1.55×1.20 m、energy=2300；顶灯 0.82×0.13 m、energy=850。另有恒定 World strength=0.18，以及窗外墙面 Principled emission strength=0.6，没有 HDRI。这些是设置值，不是逐源照度或贡献测量。曝光 +0.6，AgX Medium High Contrast。

固定相机、无 CAD，三个支撑高度分别用 128×128 条像素中心射线统计首个几何交点：桌面占 82.11%、87.11%、89.28%；中间高度棘轮约 2.83%、钢尺 1.26%、抹布 0.30%。该比例仅描述空场景可见几何覆盖，忽略像素抗锯齿和反射内容，不是真实感评分，也不是带 CAD 的图像占比。

实际图像中，主要是大范围灰色桌面和宽亮带，使用活动的区域差异较弱；部分道具被裁切，带 CAD 时主体还会遮住后方道具。已有提示词明确要求避免大空中心外围孤立道具、围绕近景构图，并要求反射环境，因此不能把问题归为“没有写真实感要求”。需要把空间/光照意图绑定到最终图像中可指出的位置和效果，验收同时保留环境问题。大均匀光源与简单反射环境是待逐源对照验证的原因假说；尚未测得各光源贡献，不能由瓦数或单张图断言主因。
