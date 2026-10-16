# 五步工作流实施与验证

2026-10-16 布局更新：按用户要求删除 `workflow_prompts.py`，三阶段文本原样移到
`prompts/stages/{create,inspect,repair}.md`，审阅结构移到`prompts/review.schema.json`。
原五份协议／指南／示例统一放进`prompts/`；调用层静态运行说明提取到`prompts/RUNTIME.md`。
Python仅加载文件并填入动态配置；提示词目录也纳入外围输入摘要，旧CLI入口仍可使用，
用户示例文件路径需改为`prompts/PROMPT.EXAMPLE.md`。24项工作流、13项调用验证通过；
逐字核对原五份Markdown和三阶段文本、语义核对schema，并从不同工作目录验证两个CLI的dry-run。
没有启动真实Codex或重新渲染。具体记录见`PROMPT_LAYOUT_VALIDATION.json`。
下文保留10月15日实现和验证历史，其中`workflow_prompts.py`为当时的组织方式。

2026-10-15。代码已实现并完成控制边界、CLI参数和真实Blender接线验证。**未启动真实Codex生成或视觉审阅，没有消耗模型额度；没有据此宣称图片质量提高。**

## 为什么改

原流程已经有布景指南、工具和最终预览，但外围失败不会返回Agent修复，视觉审阅也没有独立阶段。新增的外围循环将“制作、实际检查、看图判断、定向修复”明确分开，保持单一模板会话、固定样本和当前版本证据。

## 实际实现

- `workflow.py`：先读`run_workflow()`。首次任务同时写短计划和制作模板；外围执行完整review；通过后附上15张原图进行只读审阅。`revise`触发单独repair后重做检查，默认最多两轮。`blocked`、预算耗尽和执行异常保留incomplete。
- `workflow_prompts.py`：制作、审阅、返修三个英文任务及小型审阅schema。完整技术报告留在磁盘；阶段提示只带必要结论、失败及报告路径，避免反复塞入相同资产摘要。
- `call_llm.py`：保留prepare_request与一次生成兼容函数；run_codex_turn负责明确会话ID续接、附图、JSON事件、回复和usage。首轮与续接都使用ChatGPT登录并剥离模型API keys、显式关闭个人上下文自动来源。原CLI转入workflow，Python旧call_llm函数仍为仅生成。
- `preview.py`：复用原渲染与几何检查，只公开snapshot_inputs/file_sha256供工作流使用，并扩充辅助Python模块、scene_plan及外围代码的摘要覆盖。
- 首次Agent运行前冻结外围代码和固定测试组；create/repair/preview后核对。只读审阅核对模板、素材、源STL、PNG、scene.blend、构建/渲染记录及预览报告，检测漂移即拒绝交付。它不是操作系统安全边界。
- 统一`workflow_result.json`记录状态、session、实际回合、usage、修复数和逐轮报告。`final_report`只用于可交付版本；失败状态用`last_report`指向最近一次检查，不声称它仍对应失败后改过的模板。
- 技术启动错误和固定STL摘要不匹配属于外围问题，直接blocked；一般模板构建或几何失败才进入repair。畸形审阅、声称ready却有未解决问题、图片清单遗漏均不能通过。
- 英文项目指南更新为分阶段协议。Example采用17.5cm名义长边（用户范围15–20cm），表达维修中的物件组合、局部清理、大表面状态和相关光照。目标材质五项数值未变。

初版通过通常需要一次制作回合和一次审阅回合；默认两次返修时最多六个外围模型回合。这个上限不限制每个Agent回合的token、时间和内部工具调用，也不能区分账号套餐额度与额外credits。

## 验证结果

| 验证 | 实际结果 | 范围 |
|---|---|---|
| 语法、CLI帮助、两入口dry-run | 通过 | 不调用模型；原call_llm CLI可继续使用 |
| 调用层 | 13/13 | 替身子进程+本机0.160.1 help-only参数解析；新会话/续接ID、账户设置、APIkey剥离、schema、失败记录、兼容函数 |
| 工作流边界 | 24/24 | 技术/视觉返修、两轮上限、blocked、缺文件、畸形回复、图像遗漏、模板/素材/图片/产物/源数据漂移、外围基线改变、无Blender和dry-run |
| 真实Blender集成 | 15/15，0失败，0缺失 | 复制现有模板，真实Blender构建和固定256 samples review；制作与审阅均为替身 |

调用层输出：`/tmp/cad-scene-transport-agrcyx9v/results.json`。
最终控制流输出：`/tmp/cad-scene-workflow-validation-v3n_kz8p/results.json`。
最终Blender集成：`/mnt/d/data/cad_scene_workflow_validation/20261015_blender_final/validation.json`。

真实Blender使用5.2.2 LTS、OptiX GPU0、512²。被复制的源模板为
`/mnt/d/data/cad_scene_templates/workshop_20261015T083431Z`，源builder及assets清单摘要前后相同。
该副本保留原来约45cm零件的尺度解释，仅用于兼容性检查；没有伪称已生成17.5cm的新布景。
替身审阅返回ready仅用于走通交付分支；`human_review`仍为pending，且validation.json明确标记
`stub_no_visual_judgment`。第一次集成在非最终代码下也通过，保留在同根目录的`20261015_blender`。

## 可复现入口

同目录三个验证脚本独立可读，不恢复已删除的项目tests目录：

```bash
/mnt/d/anaconda3/bin/python checkpoints/image_generation/codex_scene_factory/workflow_loop/validate_call_llm.py
/mnt/d/anaconda3/bin/python checkpoints/image_generation/codex_scene_factory/workflow_loop/validate_workflow.py
/mnt/d/anaconda3/bin/python checkpoints/image_generation/codex_scene_factory/workflow_loop/validate_blender.py \
  --source_template /mnt/d/data/cad_scene_templates/workshop_20261015T083431Z \
  --output_dir /path/to/new/validation_directory \
  --blender /mnt/d/natural/blender-5.2.2-linux-x64/blender \
  --stl_dir /mnt/d/data/deepcad_stl_total --gpu_index 0
```

正式调用命令见项目README，默认使用新的模板目录。dry-run后可在同目录正式开始；一旦产生workflow或builder则拒绝重新覆盖。尚未实现进程中断后的命令行恢复，记录中的会话ID供后续显式恢复功能使用，不自动选择最近会话。

## 边界与后续研究

这轮未改公共STL法线、几何、相机或现有模板。已发现的曲面分段高光仍存在，真实Agent可能据此报告host blocker，需要单独处理。新Example、真实会话续接的端到端表现及视觉审阅有效性尚未通过模型运行验证。`inspected_images`是覆盖声明，不能独立证明图像真正被理解；最终审美和任务适用性仍需人看。

技术检查继续不包括模板搬迁、精确表面接触、像素可见性、STEP→STL语义保真和全库覆盖；固定15图也不证明下游真实照片重建收益。

参考：[官方非交互调用与续接](https://learn.chatgpt.com/docs/non-interactive-mode)。仓库分支`codex/data-generation`，基于`c6ac62adb8f5702e60af6dc36f0444a78177ec4e`；既有未提交修改保留，本轮未提交或推送。
