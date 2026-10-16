# 模板制作、审阅与返修闭环

## Task

将已确认的五步流程接入现有扁平脚本：方案和制作合并为首次任务，外围真实预览，明确的只读审阅，有界返修，最后提交人工审阅。保持主循环可直接阅读。

## Assumptions

- 继续使用现有 builder、固定相机、STL、12件 review 样本和三个空场景，不修改源数据与既有模板。
- 首次制作和返修可以自主 quick；外围负责 final review。同一模板只续接它自己的明确会话 ID。
- 最多两轮外围返修；不是限制 Agent 内部编辑次数，也不保证单回合的 token 或墙钟预算。
- 没有 Blender 配置时保留仅生成模式，明确未验收。dry_run 不执行模型或渲染。
- 首轮以17.5 cm为名义最长边，符合用户15–20 cm范围；这是物理尺度先验，不改变CAD或相机。

## Inputs

现有英文项目指南、用户需求、可选参考图、模型与推理强度、显式 Blender/STL/素材/设备路径；只使用本机已有 ChatGPT 登录。

## Outputs

模板脚本、scene_plan.md、素材、逐回合提示词/事件/回复、逐轮预览及审阅结果、统一 workflow_result.json。研究记录不包含模型或素材二进制。

## Processing Protocol

1. 保存完整输入。首次生成新会话，方案和构建在该回合完成。
2. 直接复用 preview_template(review)，保留技术失败和计划分母。
3. 技术成功后将本轮实际图片与报告交给同一会话，只审阅不修改模板。
4. 审阅输出固定字段；核对图片覆盖、脚本/素材/检查器/测试输入摘要与证据未变。
5. ready且技术/版本条件成立才交付待人工审阅。技术失败或revise进入repair后重做预览；blocked或预算耗尽交付incomplete。执行异常保留失败状态和已有证据。

## Metrics

记录实际Codex回合及usage、外围修复轮数、各轮技术结果及样本计数、Agent审阅决定和剩余问题。不新增自动真实感评分。

## Decision Rule

ready_for_review不是human approved。仅当当前版本通过已实现技术检查、对应图片已提交审阅且返回ready、没有未解决事项或版本漂移时使用该状态。未执行的迁移等检查继续明确保留。

## Edge Cases

缺少builder或plan纳入修复反馈；技术失败跳过视觉模型回合；审阅修改模板、篡改预览或输入漂移不得沿用旧结果；未知decision或畸形输出不能视为通过。原目录有workflow产物时不覆盖；不自动重试认证或CLI失败。公共渲染器问题应报告，模板Agent不得绕过检查器。

## Validation / Sanity Checks

不调用真实Codex。替身CLI验证首次/续接、附图、schema、账号与上下文参数、错误留痕。控制流验证首次通过、技术返修、视觉返修、两次修复上限、blocked、畸形回复、图片遗漏、模板及图片漂移。复用已冻结模板副本执行一次真实Blender review，并用替身审阅验证集成；其结果只证明接线，不证明模型视觉能力。

## Implementation Constraints

不恢复旧src/examples/tests，不增加Agent框架。call_llm只处理调用，workflow控制顺序，preview继续负责真实检查。阶段提示词可独立成文件；校验集中在模型输出和交付边界，不逐层重复检查。

## Deliverables

可运行主循环、兼容原CLI入口、更新的英文任务指南与中文README、验证脚本及紧凑结果和NOTE。本轮不启动付费/账号额度模板生成，不修改法线、灯光或现有模板。
