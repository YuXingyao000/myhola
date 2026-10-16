# 首轮工作流的曲面着色阻塞

## Task

定位 workshop_workflow_001 的视觉审阅阻塞；修正共享 STL 导入的着色法线，并用原模板对照验证。区分实际缺陷与本轮未覆盖的检查。

## Assumptions

报告指出四个曲面样本存在分段高光；逐面法线是待验证原因。STL 没有原始 B-Rep 曲面与硬边语义，按夹角平滑只是网格近似，不能宣称恢复了精确 CAD 法线。名义尺寸和既定材质继续沿用。

## Inputs

原模板 `/mnt/d/data/cad_scene_templates/workshop_workflow_001`、round_00 的场景和图片、固定 review_models.json 中 12 件 STL；Blender 5.2.2 LTS、OptiX GPU 0、512×512、256 samples、原样本 seed。

## Outputs

可读的最小渲染器修改、独立新目录内的对照渲染与验证 JSON、更新的验收文字。原模板、旧报告和源 STL 保留不动。

## Processing Protocol

1. 在实际 Blender 中检查导入后的平滑标记、自定义法线和几何。
2. 验证按相邻面夹角设置硬边，仅插值着色法线；不细分、不倒角、不移动顶点，不改变材质、相机和布景。
3. 固定夹角阈值并记录。对已报告问题的样本和全体 12 件比较几何摘要、法线与渲染结果；额外保留三个空场景。
4. 在提示词中明确：既定范围外未运行的检查及人工 pending 不自动构成 blocker；有证据的共享渲染器缺陷仍会阻塞。

## Metrics

顶点坐标、三角形索引、边界框是否不变；源 STL 摘要是否与冻结面板一致；计划/成功/失败数；平面与大夹角硬边的法线；四个报告样本中明暗条带的实际图像观察。不构造真实感总分。

## Decision Rule

只在实际探针证明几何不变且硬边保留后采用修复；图像与数值证据单独报告。若低面数轮廓或浅角度硬边仍有问题，作为限制保留。旧 workflow_result.json 不改写为通过；本次不启动新的 Codex 调用。

## Edge Cases

空或损坏网格沿用现有导入失败路径；OBJ 保留自身法线策略。浅角度的设计折线与曲面离散边不能仅靠 STL 可靠区分。任何样本失败仍留在固定分母内；输出不覆盖旧目录。

## Validation / Sanity Checks

真实 Blender 探针和一次完整固定预览；平面块、圆柱和六边孔的局部法线检查。确认模板/素材/源 STL 摘要未变，空场景输出及平面零件不出现意外变化。运行与改动相关的已有检查。

## Implementation Constraints

保持现有扁平文件职责，不新增 Agent 框架、不改登录调用、不增加数据依赖。提示词正文继续放 Markdown；Python 仅处理渲染。保留其他未提交工作。

## Deliverables

render.py 的最小修复、验收提示词/README 更新、验证脚本与 NOTE/RESULTS。展示实际前后对照，明确本轮没有新的 Agent 审阅或人工批准。
