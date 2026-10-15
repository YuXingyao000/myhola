# Task

对新值消息固定末次 checkpoint（epoch39/step14520）做完整失败溯源，帮助解释当前可续训端点的结果。该快照由“本次运行最后完整保存”预先确定，不依据 test 表现搜索 checkpoint，不替换主比较的 validation 最优 step1089。

# Assumptions

沿用上级 [完整诊断协议](../PLAN.md) 的所有测量、分母、未知与解释规则。固定末次模型只与历史 conditioned 比较；原 GAT 对照只有早期训练，不能将此端点比较写成14520步等曝光结构消融。是否续训由 root 汇总端点结果与用户最新条件授权决定，本 runner 不启动训练。

# Inputs

- 方法 `encoder_edge_value`；评估根 `checkpoints/encoder_edge_value/evaluation_20261015/endpoint_14520/candidate`。
- 权重 `checkpoints/encoder_edge_value/candidate/20261015T073112.923985Z/weights/last.ckpt`；SHA256 `e9120c78254d9b94aaaef88333c0f0656fd6a01346225d092fb8341297bff2bf`，epoch39/step14520。
- 历史 `conditioned` 评估与诊断、GT和2417 split同上级协议，仅只读复用。
- `same_geometry_source`：同源 CAD、规范GT面顺序与各自真实 posterior mean，禁止强制两Encoder latent相等。

# Outputs

本独立目录下保存 endpoint manifest/cohorts/来源、全部失败+救回的成对阶段重放、缺失baseline补跑、fit-only→optimized退步路径、四图/中文报告/离线HTML/独立与浏览器核验。主step1089两组目录不改写。

# Processing Protocol

1. endpoint两种post全部终态后核对方法、快照、同CAD合同、split与预算，再冻结全部失败+救回。
2. 用相同历史来源复用规则和只读baseline核验；原终态与STEP DATA smoke通过后完整重放。
3. 每 runner8并发、单census最多4runner=32，与所有其他post/census总并发不超过128；不延迟主step1089两组已经启动的诊断。
4. 所有fit-only成功而optimized失败做同raw/选择后数组路径对照。阈值、采样、后处理完全沿用，不增加重试择优。

# Metrics

同上级协议：.002/.006/.01/.08归一化长度、正/负/未知、多标签、互斥末端、四cohort、逐ID同侧/双侧连续量和最终失败面检查。端点固定2,417分母，不缩分母。

# Decision Rule

此处解释固定末次模型与conditioned相比的实际结果；结合整体几何分布、solid与主结构对照判断续训价值。不得按本次test把端点重新命名为“最佳”，不得将训练预算差异归于结构。

# Edge Cases

身份/来源/协议冲突拒绝；失败、缺失、超时与重放未知保留分母；空组明确0；合法成功也可能有几何告警。阶段标签不直接等于原因。

# Validation / Sanity Checks

沿用已通过35项合同回归与已冻结科学源码；此目录另存source freeze和本协议SHA。真实smoke后全集；独立重算28观察×4阈值及paired转换、来源/STEP/fit数组，普通和离线浏览器检查4组×4阈值、ID与四图/390px/无离线HTTP。

# Implementation Constraints

只在任务本地runner增加显式endpoint分组和展示身份；不改已冻结科学工具、生产post、测量公式及主模型目录。所有端点产物独立。

# Deliverables

固定last端点完整失败统计/解释、四图/自包含离线页、独立验证与未知清单；向root报告结果供续训判断，不在诊断任务中训练。
