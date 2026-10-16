# The Shed 工作台对照检查

## 1. Task

下载 Poly Haven The Shed 原工程，对照 `workshop_20261015T083431Z` 的工作台附近模型、材质、光照、布局、可见密度与尺寸，给出可追溯的图文诊断。暂不修改生产管线或当前模板。

## 2. Assumptions

- 用户已明确：目标零件最长边约 15–20 cm，不是体积 15–20 cm³。必要时用 17.5 cm 作中间示例；这不是每个 CAD 的真实尺寸 GT。
- 当前 CAD 已归一化，当前模板声明 1 unit≈0.25 m。The Shed 的实际单位须从工程及常见物件尺寸交叉检查。
- 艺术家场景是参考案例，不是真实照片或定量真实性真值。

## 3. Inputs

- 官方工程：https://dl.polyhaven.org/file/ph-assets/Scenes/the_shed.zip
- 项目与许可：https://blog.polyhaven.com/the-shed/ 、https://polyhaven.com/license
- 当前模板：`/mnt/d/data/cad_scene_templates/workshop_20261015T083431Z`
- 最终 host review：`previews/20261015T084806497728Z_review/`
- Blender：`/mnt/d/natural/blender-5.2.2-linux-x64/blender`

## 4. Outputs

- 原压缩包、解压工程及下载 SHA256：`/mnt/d/data/cad_scene_references/the_shed/`
- 检查脚本、参数清单、日志、自行渲染图片：其 `comparison/` 子目录。
- 本目录保存分析计划、结果笔记及紧凑结果索引；大素材与图片不进入 Git。

## 5. Processing Protocol

1. 校验下载、列出 ZIP 内容并安全解压，保留原文件。
2. 禁止 .blend 自动执行脚本；只读检查原工程，记录版本、依赖、相机、色彩管理、灯光、材质、世界坐标尺寸与实例。
3. 识别主工作台与实际可见道具，复现作者相机并检查工作台近景。预览降采样与任何兼容性调整均单独记录。
4. 与当前模板进行参数与图像对照。若放入 CAD，保持原 STL 网格与材质，记录目标尺寸、刚体摆放、相机及环境差异，避免冒充完全单变量实验。
5. 分开报告直接事实、图像观察与因果假设。提出下一轮改动顺序，不自动实施。

## 6. Metrics

- 长度统一换算米／厘米；列工作台、主要道具及零件比例。
- 布局看支撑面上的物品组、远近层次、位置与朝向；对象数仅作工程清单。
- 若使用射线像素统计，定义为固定相机首交点的类别面积占比，注明分辨率、遮挡和实例处理；不能当作物理体积密度或真实性分数。
- 材质记录实际节点、贴图通道、normal/roughness、UV／坐标方式；光照记录光源尺寸、方向、功率、World 与曝光。不同世界尺度下不直接比较瓦数优劣。

## 7. Decision Rule

按可解释失败机制排序：明显比例错误、网格着色问题、支撑面／道具的材质差异、可见布局与照明。不给出未经验证的综合真实感分数。

## 8. Edge Cases

下载失败可断点续传；缺贴图或不兼容节点须记录，不能静默补材质。隐藏物件和 collection instances 单独核对。工作台放置区被占用时先找其他位置；诊断副本若隐藏道具需明确记录，源工程保持不动。

## 9. Validation / Sanity Checks

ZIP CRC、源文件摘要、Blender 实际加载日志、贴图存在性、工程尺寸、预览成功与目标入镜。兼容性诊断不等价于作者原版像素复现。

## 10. Implementation Constraints

分析脚本仅写参考输出目录。复用现有 Python / Blender，不安装大型工具，不调用新的 Codex 或收费服务，不终止其他 GPU 作业。只使用一张有余量的 GPU；不足时使用 CPU。代码保持简单。

## 11. Deliverables

可打开的本地参考工程、工作台对照图片、布局／尺度／材质／照明对照表、带证据与优先级的中文结论。
