# 跨环境场景生成：2026-10-23

## 目标与边界

用户选择机械装配车间、电子维修工作室、普通办公室三个环境，继续使用独立的
room workflow 生成可复用场景。此轮不放置目标 CAD，不进行 FLUX 编辑。
每个场景新建 Codex 会话和输出目录；主控只准备任务、启动程序、留存和核对证据，
不手工编写或修改生成的布景，不向制作会话补充临时艺术指导。

Shed 只提供两张质量参考图和说明，采用 `quality_only`；不提供其基础工程。
布局和素材选择交给制作 Agent。办公室与电子工作室不以工业污渍或仿 Shed 布局
作为真实感要求。验收关注活动与布局、比例、表面、光照及六视角一致性。

## 固定设置

- `gpt-6-astra` / `ultra`，ChatGPT 登录，独立初始会话。
- Blender `/mnt/d/natural/blender-5.2.2-linux-x64/blender`，Cycles / OPTIX。
- 内部快速预览按既有合同使用 64 samples；外围六图 128 samples、seed 8。
- 每任务最多 3 次外围返修；内部工具迭代与外围返修分开记录。
- `danger-full-access` 沿用本机已验证配置；不宣称具有文件系统隔离。
- 启动前查询：八张 RTX 4090 均 0% / 15 MiB，计算进程列表为空。
- 三任务使用 GPU 0、1、2；保持统一设备可见性，不做编号重映射。
- 共享素材缓存 `/mnt/d/data/cad_scene_assets`；所有下载通过同一 `flock` 串行化。
  搜索、制作与不同 GPU 的渲染可并行。这个操作要求附在本次实际输入中。
- 仅使用既有免费素材工具；不假设 Sketchfab 凭据可用。

| 场景 | 示例提示词 | 输出目录 | GPU |
| --- | --- | --- | --- |
| 机械装配车间 | `PROMPT.ASSEMBLY.EXAMPLE.md` | `/mnt/d/data/cad_scene_templates/assembly_20261023_001` | 0 |
| 电子维修工作室 | `PROMPT.ELECTRONICS.EXAMPLE.md` | `/mnt/d/data/cad_scene_templates/electronics_20261023_001` | 1 |
| 普通办公室 | `PROMPT.OFFICE.EXAMPLE.md` | `/mnt/d/data/cad_scene_templates/office_20261023_001` | 2 |

## 留存与完成判定

本目录保存计划、实际提交的用户输入、命令、Git 状态和最终紧凑报告。
模板保留 request、阶段提示词、事件、用量、每轮构建／六图／审阅报告。
沿用旁路 `capture_versions.py` 保存四个生成文件的稳定版本；不改变 Agent 文件。
宿主与参考摘要匹配 request；正式审阅证据与最终交付源码进行摘要核对。

`ready_for_review` 只表示程序已提交人工审阅。所有失败、缺失、资源缺口、返修次数
和未通过原因均保留；不通过更换任务目录隐藏首次结果。首轮与返修结果分别记录。
人工验收仍由用户决定，不能以 Agent 的 ready 填写 human_review=approved。
