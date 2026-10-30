# 环境工具的真实 Blender 验证（2026-10-19）

## 范围与结论

验证 `room_tools.py` 的构建、依赖解析、搬移、相机投影、渲染结果记录，以及
`room_workflow.preview_room()` 与这些报告的衔接。使用 Blender 5.2.2 LTS；
小场景在 CPU 上以 1 sample 渲染。这是实现检查，不是场景真实感评估，未调用 Codex。

机器可读结果见 [TOOLS_RESULTS.json](TOOLS_RESULTS.json)。可复现脚本见
[validate_room_tools.py](validate_room_tools.py)。所有临时工程、图片和日志留在
JSON 记录的 `/tmp` 目录，未写入版本管理目录。

## 具体覆盖

- 构建脚本从模板内打开一个基础 `.blend`，其中立方体使用带 1001、1002 两个
  瓦片的 UDIM 材质。主工具保存 `room.blend`、场景概要和构建报告。
- 复制模板并移走原目录；从保存的 `.blend` 渲染，并从复制后的脚本重新构建。
  旧绝对路径不可用，因此不能靠原目录里的素材帮助通过。
- 横幅 64×48、竖幅 48×64 均检查实际投影矩阵。垂直 FOV 设为 45°，回算为
  45.0000024°；实际 PNG 尺寸匹配。源曝光 0.3 保留。
- 删除 1002 瓦片时构建失败，报告指出具体瓦片。依赖模板外素材时构建失败。
- 两个视角中故意加入一个无效相机，另一张继续渲染；报告保留 planned=2、
  success=1、failed=1。
- 外围 `preview_room()` 真正调用 Blender 构建并渲染六张图，验证绝对路径、
  PNG 读取、画幅以及报告字段。另测六张中一张失败、构建缺少瓦片的传递。
- 对原 Shed 只读导出概要：1103 个对象、124 个材质、4 个灯、425 个文件／瓦片
  依赖；未发现缺文件。检查前后源工程 SHA256 一致：
  `fe6294a84839e2225f2d9702a05ff5bc4472ff78743daabbcc60a7a95761c6e1`。

## 实现中确认的问题

Blender 的 `bpy.utils.blend_paths()` 会同时返回历史素材来源的 weak references。
原 Shed 的该列表有 1252 项，包含原作者机器上早已不可访问的路径，而实际
`bpy.data.libraries` 为空。工具按真实图像、链接库、字体、音视频、缓存文件、
体积 datablock 检查，避免把历史来源路径当作丢失的渲染依赖。

初次检查发现外围 `_blender()` 非零退出时，`preview_room()` 还没有读取已存在的
结构化报告，导致缺失资源细节和部分成功计数丢失。主流程随后修复；本验证的
最终版本包含该失败路径，详见 JSON 中的两个 `workflow_*_details` 项。

外围和本验证调用 Blender 时均显式传入 `--python-exit-code 1`，使脚本异常对应
非零退出码。冻结的 Agent 工具示例暂未展示该参数；下次提示词修订时建议补充。
本轮不因文档建议修改已经启动任务的提示词摘要；Agent 仍应读取结构化报告，
不能只凭命令返回判断成功。

## 复现

输出目录必须尚不存在。此命令将二进制产物全部保存在 `/tmp`：

```bash
/mnt/d/anaconda3/bin/python \
  /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/room_workflow/validate_room_tools.py \
  --blender /mnt/d/natural/blender-5.2.2-linux-x64/blender \
  --shed_blend /mnt/d/data/cad_scene_references/the_shed/source/the_shed/the_shed.blend \
  --output_dir /tmp/room_tools_reproduction
```

## 尚未由这些检查证明的事项

本验证没有评审完整场景图片，没有进行 Codex 制作／返修，没有验证 GPU 渲染成本。
六视角接口测试故意使用简单场景，不能证明机位具有有意义的空间覆盖。
依赖检查不是通用 Blender 文件安全分析：不枚举任意 Python 代码的读文件行为，
不检查视频的每一帧、流体模拟目录或外部 OSL 脚本的完整性。实际交付另需完整模板
搬移后的重新构建和渲染（本次结果见文末追加）；本微型场景没有覆盖这些额外依赖类型。

## 2026-10-20 追加：多层 EXR 状态下的 PNG 回归

真实制作首次预览发现，Shed 基础工程保留 `media_type="MULTI_LAYER_IMAGE"`，
直接设为 PNG 会在 Blender 5.2 中失败。作者当时在构建脚本内修复；外围审阅后，
主控将该兼容处理统一放入 `room_tools.render()`：在设置 PNG 之前先设置
`media_type="IMAGE"`。原制作脚本、源工程和贴图未作改动。

本次修改 [validate_room_tools.py](validate_room_tools.py) 的微型基础工程，
明确保存 `MULTI_LAYER_IMAGE` 和 `OPEN_EXR_MULTILAYER`。其 builder 只打开基础
工程，不自行改变输出模式；因此后续成功输出 PNG 实际经过主工具的兼容处理，
没有依靠模板预先消除这个状态。

使用 Blender 5.2.2 LTS、CPU、1 sample 重跑，23 项检查全部通过，耗时约
14.46 秒。结果见 [TOOLS_RESULTS_20261020.json](TOOLS_RESULTS_20261020.json)，
临时产物位于 `/tmp/room_tools_png_regression_20261020/`；旧日结果继续保留。
除 PNG 路径外，还复查了 UDIM 两瓦片、缺失／外部依赖拒绝、部分失败计数、
横竖相机投影、微型模板搬移及外围六视角接口。只读检查的 Shed 源工程 SHA256
仍为 `fe6294a84839e2225f2d9702a05ff5bc4472ff78743daabbcc60a7a95761c6e1`。

新工具 SHA256 为 `4f8801f45f77d07ff1925cc4a3d6556b720d3f409a9fba30028ac9b8f61e5bfb`；
实际任务的 `workflow/host_png_fix_20261020/migration.json` 记录此次变更、新快照
及显式更新的 `host_files`，原制作版本的快照继续保留。此前建议的
`--python-exit-code 1` 也已在 2026-10-20 首次恢复的提示词迁移中补入。
本追加只证明工具回归，不替代完整派生模板的搬移六图验证或场景视觉判断。

## 完整模板搬移结果（2026-10-20 UTC；2026-10-23 核对记录）

工具回归后，另用 [validate_relocation.py](validate_relocation.py) 复制完整派生模板，
在原模板路径不可用时从新路径重建并渲染六图。实际
[迁移报告](/mnt/d/data/cad_scene_templates/workshop_room_20261019_001_relocated/relocation_result.json)
为 `pass`，planned=6、success=6、failed=0；原模板目录随后恢复。源工程及打包基础
资产摘要未变，紧凑记录见 [RUN_RESULTS.json](RUN_RESULTS.json)。

这次完整房间验证使用 Blender 5.2.2 LTS、Cycles、OPTIX、GPU 2、128 samples、seed 8，
与上面的 CPU 微型回归分别记录。六图均非逐像素完全相同，8-bit 平均绝对差约为
`0.000081–0.000356`；这证明同一 Linux 环境下的目录搬移可重建，没有验证 Windows
或跨硬件一致性。主控对六图的另行视觉判断及局限见 [NOTE.md](NOTE.md)，用户审阅仍为
`pending`。本次文档核对没有启动新的 Blender 或模型调用。
