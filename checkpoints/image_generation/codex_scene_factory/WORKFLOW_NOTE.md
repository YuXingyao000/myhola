# 布景指南与工具接入（2026-10-15）

## 为什么修改

用户对 workshop_002 的观察是环境太干净、显眼道具与大面积表面缺乏真实材质；当前目标零件材质可继续沿用。旧 Example 的“干净、宽松”要求和只用程序环境材质的约束，与新目标冲突。本次修改工作指导、素材输入和实际预览证据，不以文字宣称真实感已经改善。

## 已实现

- `SCENE_GUIDE.md`：中文布景指导，覆盖使用状态、相对尺度、素材选择、材质调整、现实光源与反射、看图修复。
- `ACCEPTANCE.md`：固定 quick/review、技术检查的范围与未验证项、视觉自查和任务退出后的人工审阅。
- `PROMPT.EXAMPLE.md`：中文的长期使用、有人维护的车间。目标铝材保留已接受的参数，改变环境要求。
- `assets.py` / `ASSET_TOOLS.md`：Poly Haven、ambientCG、Sketchfab 的官方接口；CGBookcase 和自有素材走本地 catalog。下载文件、缓存、逐文件摘要和清单合并，模板不依赖缓存路径。
- `preview.py`：重新构建当前模板、固定预览、完整样本分母、实际图片校验、输入前后摘要、联系表和报告。
- `render.py`：默认仍遇错停止；预览可在单件失败后清理并继续，最终以失败退出。支持固定逐件 seed 和空环境预览。
- `call_llm.py`：显式组合指南、验收和素材工具说明；保存全文、来源摘要和工具配置。新增 dry_run；提供 Blender/STL 后，agent 可自检，退出后外围再次 review。保留 ChatGPT 登录、新会话和个人上下文隔离，不回退 API key。

没有创建原生 skill，也没有自动重新调用 Codex 的外层修复循环。旧提示词中“controller 已管理 max_retries=3”的错误描述已移除。

## 固定样本

沿用首轮已看过的 12 个训练零件，读取 `deepcad_v6/<id>/normalized_shape.step`，用 PythonOCC 的唯一 `TopAbs_FACE` 数排序。当前环境 IndexedMap 使用 `Size()`，不是 `Extent()`。

面数从 7 到 28；quick 为 `00000061`（7）、`00000070`（14）、`00951201`（28）。完整 ID、STEP/STL SHA-256 和源训练清单 SHA-256 保存在 `review_models.json`。未增加形状类别配额，未声称该组代表全库分布。

quick：3 件、64 samples；review：12 件、256 samples，加高度 -0.901、-0.4505、0 的三张空环境。均为 512×512。同一件的 seed 是完整 review 列表索引，两档一致。

## 实际验证

1. 不调用模型，使用替身子进程核对 call_llm：dry_run 不启动进程；实际任务路径为一次 Codex 调用后一次外围 review；API key 从子进程环境移除，仍使用 Astra / ultra / ChatGPT。真实 dry_run 已保存完整上下文。
2. Blender 5.2.2 LTS / OptiX GPU 0，对原 workshop_002 执行 review：15/15 成功，输入摘要不变，visual=pending。输出：`/mnt/d/data/cad_scene_templates/workshop_002/workflow_validation_20261015/`。
3. OptiX GPU 6，对故意缺少 `00000070.stl` 的独立目录执行 quick：计划 3、成功 2、失败 1，退出码 1，后续零件继续生成。输出：`/tmp/cad-scene-workflow-failure-20261015/`。没有改动真实 STL。
4. 公开素材检索三家均返回实际候选。Poly Haven 实际下载 1k HDRI（1,488,808 字节）及 1k PBR（5,135,621 字节）；核对上游大小/MD5、本地 SHA-256，禁用网络后缓存仍可复制到第二模板。
5. 外部 PBR 材质在 Blender 中 append，由现有 build.py 打包四张贴图；复制模板并暂时移走原模板目录后重建成功。该验证针对独立素材夹具，不冒充每个生产模板都已迁移验证。
6. 夹具验证缺失/损坏图片、构建失败完整分母、源 STL 摘要变化、导入失败清理、清单合并、缓存损坏、ZIP 路径越界和软链接；相关 Python 语法检查通过。

紧凑记录及代码摘要见 `WORKFLOW_VALIDATION.json`。全部真实素材测试约 6.6 MB，没有额外调用 Codex 或图像生成模型。指南/代码协作使用当前会话的子任务，不计作模板生成实验。

## 尚未证明或完成的部分

本轮未制作新场景，不能判断新指南对真实感的改善。workshop_002 的视觉问题仍然存在。技术通过只覆盖实际执行的 AABB 与文件检查；不验证真实支撑面、像素可见率、STEP/STL 对应或下游真实照片效果。

ambientCG 下载、Sketchfab 认证下载未进行实际联网验证；Sketchfab 首版只读取外部 access token，未实现登录/自动刷新，也未读取任何真实凭证。CGBookcase 当前需要已有素材 catalog。

迁移检查没有集成为每次 preview 的自动步骤，报告明确列为未执行。模板视觉状态由人工在任务退出后审阅，不是 agent 自己给出 approved。
