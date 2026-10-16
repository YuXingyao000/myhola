# 模板素材构建与迁移验证

本轮落实 `build.py` 的素材检查与打包功能，未接入在线素材工具或自动修复循环。

## 实现

- `assets.json` 为平坦的文件清单，必须字段为相对 `path` 和实际 `sha256`；其他来源信息原样保存。纯程序模板可以省略清单。
- 构建前核对文件与摘要，拒绝包外路径。执行两个既有 builder 后，检查实际使用的外部贴图、字体、声音是否已声明；已内嵌资源可保留。
- 初版图像范围是单张静态图片和 HDRI，不支持电影、序列图或 UDIM。
- 对象数据必须本地化。实际链接的库数据会被拒绝；合法 append 留下的来源库记录会清除，避免误拒绝。
- 打包后验证资源确实内嵌，为新打包资源保存相对路径，输出 `.blend` 与 `build_report.json`。源码和素材清单摘要亦保存在 `.blend` 中。
- 两个 builder 的函数接口保持不变；`render.py` 继续从 `.blend` 批量替换零件。

Developer 协议已对齐 Blender 5.2.2 LTS，预留空间按已审计 STL 范围向外取整到每轴 ±0.901、最大高度 1.801，支撑高度范围为 [-0.901, 0]。相机和 CAD 坐标未改变；完整背景要求已补入协议和 Example。

## 已执行验证

实际使用 Blender 5.2.2 LTS，临时验证目录为 `/tmp/cad-build-assets-czdxjbt3`。

- 声明贴图和 `.blend` 模型资产，使用 append 导入带内嵌贴图的对象，构建与打包成功。
- 摘要不匹配、文件缺失、包外路径、未声明贴图和真正 linked 的库数据均被拒绝。
- 无素材清单的程序布景构建成功，零件材质的 fake user 随场景保留。
- 复制模板到新目录，将原目录改名使旧路径不可访问；重新从副本构建成功。
- 将 `.blend` 单独复制到另一目录，再让重建用的模板和素材路径不可访问。
- 从独立 `.blend` 使用 OptiX 渲染三个真实 training STL：00000061、00000070、00999999；均通过入镜、支撑和 AABB 遮挡检查，得到 512×512 RGB 图片。

完整验证记录：[validation.json](/tmp/cad-build-assets-czdxjbt3/validation.json)。功能渲染使用 8 samples，目的为验证依赖打包与渲染行为，不作为照片真实感结果。

## 首轮用户测试

直接使用 [PROMPT.EXAMPLE.md](/mnt/d/python/packages/cad_scene_factory/PROMPT.EXAMPLE.md)。完整三步命令见 [README](/mnt/d/python/packages/cad_scene_factory/README.md) 的“首轮 Example 测试”。

当前素材工具没有实现；没有供应该工具时，协议要求 Codex 使用程序布景。Codex 生成、Blender 构建、三零件预览仍需分别执行。首个模板经人工看图后，再整理可复用的 skill、素材获取和预览修复流程。本轮未调用额外 Codex 生成任务，也未修改账号／网络权限配置。
