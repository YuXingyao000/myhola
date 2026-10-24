# Task

冻结于2026-10-24：制作可离线打开的单文件HTML，逐件查看no_edge_feature64的全部316件OCC失败，定位失败面、输入边及逐容差裁剪日志，展示距离和分布。已有事实为307件裁剪失败、6件solid构造失败、3件STEP检查失败。任务不包含训练、调整后处理或重算正式合法率。

# Assumptions

数据：固定validation 3495件；失败清单来自原failure_analysis配对结果的regressed及both_failed并集。面索引保持不变，post边索引须按双owner映射，不能直接当raw半边索引。

边界：GT仅作带标签的诊断参照；所有几何使用原归一化坐标，无额外配准，不标毫米。最后恢复路径历史中间几何未保存，允许独立诊断重放并严格核对逐face日志。

实现：现代Chromium，单HTML内嵌Plotly及逐件gzip数据，无CDN和外部请求。显示采样不等于解析曲面精度；所有未知值保留为空。

# Inputs

本目录父级原始预测raw_predictions/posterior_mean/no_edge_feature64、post/optimized/no_edge_feature64、failure_analysis/raw/no_edge_feature64及summary.json；固定3495件source32对应记录；GT为/mnt/d/data/deepcad_v7。checkpoint、split、后处理与原评估一致，详见父级SPEC.md与REPORT.md。读取当前生产代码和实际历史日志，不覆盖任何历史产物。

# Outputs

failure_viewer.html为可交付单文件；README.md说明解释边界；metrics.csv及summary.json为紧凑统计；data/cases/*.json.gz为逐件打包源；replay/为独立重放记录；build脚本、页面源、验证记录与截图。生成的较大几何及HTML作为本地产物保留，源码和紧凑审计记录版本管理。

# Processing Protocol

1. 固定316件清单，保留原终态和全部恢复路径逐面日志。
2. 用原raw、稳定样本seed、CPU单线程及原优化/拟合参数重放最后fit_score_v2_roundtrip_optimized路径。仅局部包装拟合函数截取优化后网格和周期替换后的拓扑边，不改生产算法。最终solid构造可跳过并明确标记；不能据此更新合法率。
3. 对重放与原记录的faces、selected_edge_count、periodic_edge_replacements严格核验。只有一致者标注“日志一致的诊断重放”，不称历史保存快照；不一致保留并突出标记。
4. 原始预测、GT、source32及拟合后重放几何分开显示。拟合面网格用于显示；拟合边以64个参数采样点测量解析owner曲面的有界投影距离，保存投影连线。投影失败记null，不用显示网格距离替代。
5. 端点测量保留同owner不同边的最近端点，并额外记录每条曲线自身首尾间距。2026-10-24诊断发现近闭合整圈曲线可能被不同边代理连到远处另一环，因此页面默认展示二者较小的“最近连接候选（含自身首尾）”，逐条标明来源，保留原不同边代理与unknown。此值仅为局部候选距离下界，不是全局wire配对或可闭合证明，不改OCC规则。raw历史端点另有GT辅助闭合标记，明确说明。
6. 所有316件可浏览/检索/按阶段、配对变化过滤，选择面、边并查看尝试记录；显示整体、孤立面、GT和距离连线；全体3495件raw指标背景用于观察失败与成功的重叠。

# Metrics

距离为L2归一化长度；分别显示边到owner曲面每点距离的mean/P95/max、同owner端点间距mean/max以及unknown计数。每CAD的mean按边-owner等权平均（每记录同样64点）；跨CAD汇总按CAD等权。raw网格距离复用原16点到16×16三角网格测量；raw分布的P95是各边-owner P95的最大值，不是全点合并P95。CDF/直方图明确分母和缺失。GT Chamfer如展示使用平方距离单独标注，不混入长度曲线。显示六位小数仅用于压缩几何，测量保持原浮点精度。

# Decision Rule

这是观察工具，无模型/后处理版本选择，无阈值优化。滑杆若提供仅为查看分布，不能被称为OCC合法性门槛。OCC原流程已尝试0.002/0.006/0.01/0.015/0.02/0.025/0.05/0.08；开放wire、丢环和solid失败分开解释。

# Edge Cases

空边、无owner、错误索引、退化/闭合曲线、周期曲面、缺失文件、非有限值、投影失败及重放不一致均保留对象并记录错误。不以默认0填充，不排除失败样本。没有距离的对象仍能查看日志。重放异常不影响原评估结论；原始几何作为可识别的备选视图。

# Validation / Sanity Checks

核对316唯一ID及307/6/3阶段计数、767失败面；原文件hash和代码hash保持不变。比较逐face重放日志；核验raw每条记录与原缓存、投影连线长度与测量值、所有索引和有限坐标。用平面/直线几何检查投影；真实浏览器离线加载、切换案例/阶段/面/边/来源、相机旋转缩放、直方图/CDF，捕获JS错误与外部请求，并保存截图。缺失数据必须在验证报告和UI可见。

# Implementation Constraints

仅写本failure_viewer目录和必要的父级交付链接，保留已有dirty工作。CPU并行仅用于316件独立诊断，禁止占用训练GPU。使用现有torch环境和PythonOCC，不重装。页面中文为主，Plotly本地打包；显示层不能改变测量语义。按用户偏好采用思源中文字体子集及Archivo/Jost/CommitMono字体栈。

# Deliverables

交付HTML绝对路径、简短使用方法、距离定义与重放核验结果，以及紧凑的失败原因和距离分布统计。不能只交静态截图或启动命令。
