# 旧 VAE optimized 失败全集与 50 件选择审计

本审计使用已经完成的 posterior-mean 重建，不重新推理、不修改原比较产物。所有数值来自原始逐对象表；抽样在深入逐件几何诊断和查看渲染结果之前冻结。后续诊断结果另行记录，不用新的重放替换原始分类。

## 失败口径与总体

`native_valid_solid` 同时要求原生读取的根形状为 `TopAbs_SOLID`、`BRepCheck_Analyzer.IsValid()` 通过。读取采用 `read.precision.mode=0`，检查先于新增容差修改和三角化；来源为 [runner](../../../tools/vae_compare_post.py)。`compat_valid_solid` 是重新读取后强制 0.1 容差的另一指标，`success.txt` 也不能代替 native 判定。

2,417 accepted 对象中，旧版 optimized 有 2,048 件 native-valid solid、369 件失败。原始 test 2,424 件，7 件数据契约不支持，未从原覆盖率中消失。

| 失败观测 | 数量 |
|---|---:|
| Compound（根类型 0） | 242 |
| Shell（根类型 3） | 126 |
| Solid 但 BRepCheck 未通过（根类型 2） | 1 |
| BRepCheck 本身通过，但不是 solid | 319 |
| BRepCheck 未通过 | 50 |
| 原始 merged pair topology 完全匹配 GT | 268 |
| 原始 merged pair topology 不完全匹配 | 101 |
| 后处理几何指标可计算 | 292 |
| 后处理预测几何采样失败 | 77 |

369 件全部存在且能读取 STEP，原后处理均正常返回，无 post_error 或 timeout。唯一 invalid-solid 对象 `00480996` 的 native BRepCheck 为 false，但 compat 为 true，且有 `success.txt`；必须保留此例，不能按成功标记排除。

BRepCheck 允许一个内部有效的开放 shell 或 compound；319 件通过该检查不表示闭合 solid。268 件 raw pair 精确也只表示合并的不同面邻接集合正确，不能推出 wire／seam／方向／几何一致性正确。

## 输出与日志可用性

每个失败对象都有 `post/optimized/legacy/{id}/recon_brep.step` 和 `separate_faces.ply`。后者是在拟合后、裁剪前导出的独立曲面 mesh，不能冒充最终 STEP mesh。只有 `00480996` 另有 `recon_brep.stl` 和 `success.txt`。

原日志位置为 `comparison_20261011/metadata/post/optimized/logs/legacy/{id}.log`。292 件日志为空，77 件有异常 traceback：全部发生在**预测 STEP** 的 `sample_native_geometry`，65 件提示某面无 triangulation，12 件提示无有限正 mesh 面积；不是 GT 采样失败。它们表明测量遇到的现象，不能直接当成神经网络失败根因。

原运行 `is_save_data=False`，没有保存优化后逐条曲线数组、每面 wire／trim／sew 的详细 trace。独立诊断可使用已有 STEP、原始 NPZ 和日志；定位后处理内部阶段需要另外重放，输出写到本目录，记录与原结果是否一致。

## 已冻结的 50 件

选择工具为 [legacy_failure_select.py](../../../tools/legacy_failure_select.py)。规则见 [PLAN](PLAN.md)：shape type × native BRepCheck × raw pair exactness，共九层；每层先 1 件，剩余 41 件按 `n−1` 容量用整数 Hamilton 最大余数分配。层内按 GT 面数、`SHA256("legacy-failure-v1:{id}")`、ID 排序，取均匀秩位置。没有按图是否好画或分数是否支持某假设进行替换。

| 原生类型 | BRepCheck | Raw pair | 失败全集 | 选中 |
|---|---|---|---:|---:|
| Compound | 未通过 | 精确 | 10 | 2 |
| Compound | 未通过 | 不精确 | 26 | 4 |
| Compound | 通过 | 精确 | 149 | 18 |
| Compound | 通过 | 不精确 | 57 | 7 |
| Invalid solid | 未通过 | 精确 | 1 | 1 |
| Shell | 未通过 | 精确 | 7 | 2 |
| Shell | 未通过 | 不精确 | 6 | 2 |
| Shell | 通过 | 精确 | 101 | 12 |
| Shell | 通过 | 不精确 | 12 | 2 |

最终 50 件包括 compound 31、shell 18、invalid solid 1；BRepCheck 未通过 11；raw pair 精确 35；原几何可算 37、采样失败 13。GT 面数 1–9／10–19／20–30 分别 9／29／12。这是用于诊断覆盖的案例集，比例不估计失败机制的总体发生率。

- [selected_cases.json](selected_cases.json)：顶层 `cases`；每项 `id`、`stratum`、`stratum_dimensions`、`gt_face_count`、`gt_face_bin`，以及完整数值化的 `raw_legacy`／`raw_joint`／`post_legacy`／`post_joint`／`fit_legacy` 原记录。空值为 JSON null，布尔与数字保留类型，八位 ID 保留字符串。
- `paths` 提供 GT NPZ／STEP、两模型 raw NPZ、optimized STEP、旧版 fit-only STEP、独立拟合面 mesh、旧版 post 日志及状态 JSON 的绝对路径；`path_exists` 明确可用性。选中对象这些路径全部存在。
- [selection_summary.json](selection_summary.json)：完整来源指纹、各层容量／配额／排序位置、失败全集与50件分布、全部成功／失败对象的原始指标统计。
- [selected_ids.txt](selected_ids.txt)：50 个唯一 ID，字典序；SHA256 为 `a85e5fe91773f29ac6539fa7acdc3d149854a0469176b2477ef865604a0072ae`。

原始 CSV SHA256：raw `92e94d6c9adb4c6ec04c6fc73bf75f11e7f9eaef9713618e43335180a39b7fe5`；optimized `10849cb6f0fce73be8475b03f45cb9b007ef8146e0d639597c69586e4b10bb0d`；fit-only `1739d4df263c508547115801a32916474ecd88fab7f407c0eb77cc5909938c38`。工具和输入来源也保存在 manifest 中。

## 成功／失败总体的描述性对照

对照使用全体 2,048／369，**不是选中的 50 件**。JSON 同时提供未控制总体、三档面数、每个精确面数的 mean／median／p90／可算数量。

精确面数标准化：在每个两组共同存在的 GT 面数内分别求均值，再按失败总体的面数分布给两组同样权重。27 面只有 4 个失败而没有成功对照，故共同支持为 365／369；这 4 件仍留在总体与选例框架中，不填补或虚构对照。

| Raw 指标均值 | Native valid | Native failed |
|---|---:|---:|
| 对应面 MAE，未控制面数 | 0.00194581 | 0.00278051 |
| 边 CD²，未控制面数 | 0.0000459508 | 0.000329471 |
| 对应面 MAE，精确面数标准化 | 0.00197695 | 0.00278052 |
| 边 CD²，精确面数标准化 | 0.0000468429 | 0.000331908 |
| 互反半边 CD²，精确面数标准化 | 0.0000443015 | 0.000310841 |
| Pair F1，精确面数标准化 | 0.999614 | 0.989452 |

控制面数后，失败组仍有更大的边几何与互反不一致误差。这支持把它们纳入逐件诊断，不能证明哪个量导致失败。面数控制没有控制曲面类型、尺度、曲率、孔洞和数据难度；分组本身又按最终有效性定义，不是因果干预。

## 下一步诊断的证据边界

已有 raw NPZ 保存对应 GT 面顺序下的 `pred_face[F,16,16,3]`、`pred_edge[E,16,3]`、半边连接及共享输入 latent 摘要；它支持逐 pair 误边／漏边、互反曲线差异、曲线到采样面的距离和端点分离等有单位的离散诊断。不要将采样面最近距离当成精确连续曲面距离。

已有 STEP 可独立提取面／边／wire／shell／solid 数量、逐子形状 BRepCheck 状态、按拓扑身份的 edge-face incidence、可三角化状态及部分可显示几何。seam 和退化边必须单列；一条边只关联一个不同面并不总是缺口，拓扑 free-edge 计数也不是曲面间物理距离。OCC 的重排、合并与修补意味着原 raw face/edge ID 不自动对应最终 STEP 子形状。

后续页面应并列 GT、raw、原 optimized STEP，显示缺失三角面的数量，并提供日志和数值。拟合／裁剪／缝合 trace 用于定位阶段，是否暴露架构限制仍需要结合几何证据讨论。本审计尚未给 50 件逐件归因。

## 验证与复现

运行 `/mnt/d/anaconda3/envs/torch/bin/python -m tools.legacy_failure_select` 生成上述冻结文件；已有文件时工具拒绝覆盖。4 项针对性测试通过，覆盖配额／稀有层／容量和余数并列、输入顺序不变性／等距秩、CSV 类型／重复身份拒绝，以及能够消除纯面数构成差异的标准化对照。真实清单为 50 个唯一失败 ID，九层配额吻合，`00480996` 已包含。原比较文件未修改。

## 控制复杂度之后，哪些关联保留／消失

以下都是观察性描述，不使用“统计显著”或因果措辞；标准化共同支持仍为 365／369。

- 边几何与互反不一致的差异保留：失败／成功边 CD² 由未控制的 7.17 倍变为 7.09 倍，互反半边平均 CD² 由 7.34 倍变为 7.02 倍；曲面 MAE 为 1.41 倍。面数构成差异不能解释这些描述性均值差异。
- 差异不完全由少数均值异常贡献：未控制总体中，边 CD² 中位数为成功 `0.0000319761`、失败 `0.0000734178`，互反 CD² 中位数为 `0.0000319665`／`0.0000674751`。但中位数差距约 2 倍、小于均值差距，说明尾部影响仍大；这不是标准化中位数。
- Pair F1 的平均差距仍约 1.016 个百分点，但两组中位数均为 1。失败总体有 268／369 件 pair 完全精确，因此这类邻接指标不能单独解释失败，也不能描述完整 B-Rep 拓扑。
- “失败对象边更多”在面数控制后不成立：GT 边界数未控制时成功／失败 `28.48／33.44`，标准化后为 `34.89／33.15`；raw 有向半边数由 `56.96／67.66` 变为 `69.80／67.05`。原数量差异随面数构成改变而消失并反向，不能据未控制均值认定“边多导致失败”。GT 面数自身从 `12.31／14.92` 变为相同的 `14.78` 是标准化的定义性结果，不是额外发现。

这些结果优先支持逐件检查边几何、互反不一致及面-边／端点关系；它们没有识别哪个改动能提高成功率，也没有排除曲面类型、曲率、孔洞等未控制因素。

## 优化造成有效性反转的总体案例

全 2,417 件中有 **8 件**旧版 `fit_only.native_valid_solid=true`、但 `optimized.native_valid_solid=false`：`00007551`、`00071931`、`00212409`、`00326520`、`00423247`、`00677369`、`00730675`、`00933890`。这 8 件均未进入冻结 50 件；不因事后发现而替换抽样案例。

这说明既有几何优化加后续 OCC 构造的结果并非逐对象单调改善。现有原日志不能区分几何位移、曲面／曲线拟合、裁剪、缝合选择和容差处理各自的贡献；“optimized 后失败”不能直接归因为神经 Decoder，也不能单凭此表确认优化器某个损失有错误。页面若额外列出这 8 件，应标为全集补充观察而不是50件样本成员。
