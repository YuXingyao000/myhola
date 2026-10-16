# DeepCAD 渲染输入与 Blender 5.2 准备记录

本轮仅执行模型整理与 Blender 下载／运行验证。素材获取、模板打包与背景覆盖属于待讨论方案；未修改 `call_llm.py`、Developer/User 提示词、相机或渲染程序。

## 为什么与协议

首轮布景前先建立完整、可追溯的 STL 输入，分清源目录重叠、编号重复和几何内容重复。冻结规格见 [PLAN.md](PLAN.md)，完整文件库存见 [SOURCE_INVENTORY.json](SOURCE_INVENTORY.json)，紧凑结果见 [RESULT.json](RESULT.json)。

复用已有 `mesh.ply` 三角网格，保留源坐标和三角形绕序；不重新归一化、不旋转、不修复，也不使用 `post_processed_shape.step` 替换源 CAD。完整记录该编号的 `normalized_shape.step` 路径和摘要。此转换验证不等于逐件认证原 PLY 与 STEP 的完整几何一致性。

## 已完成：全库整理

| 项目 | 结果 |
|---|---:|
| `deepcad_v6` 编号／PLY／normalized STEP | 102,904／102,904／102,904 |
| `deepcad_v6_cond` 编号 | 55,448，全部已在原始目录内 |
| 两目录唯一编号并集 | 102,904 |
| 成功 STL／转换失败 | 102,904／0 |
| training | 46,627，全部存在 |
| validation | 3,505，全部存在 |
| testing | 2,424，全部存在 |
| 未在三份清单分配 split | 50,348，保留并标记未分配 |
| 清单内部重复编号／跨 split 相同编号／缺失编号 | 0／0／0 |

全库没有发现已有 STL。条件目录不提供 STEP 或三角网格，不能作为额外 55,448 个 CAD 计数。本次实际转换全部为 PLY → STL，未触发 STEP 回退；转换耗时约 577 秒，使用一个 Python 进程、16 个工作线程。

输出根目录：[deepcad_stl_total](/mnt/d/data/deepcad_stl_total)。STL 文件直接命名为 `{model_id}.stl`，保留八位编号和前导零。

- [manifest.jsonl](/mnt/d/data/deepcad_stl_total/manifest.jsonl)：全部逐件结果、来源目录、PLY/STEP 摘要、split、包围盒与 STL 摘要。
- [summary.json](/mnt/d/data/deepcad_stl_total/summary.json)：完整转换和 split 统计；`split_rendering_ready_by_id` 只表示编号齐全，不表示相机检查或几何去重通过。
- [splits](/mnt/d/data/deepcad_stl_total/splits)：三份原清单的逐字节副本及未分配编号。
- [validation.json](/mnt/d/data/deepcad_stl_total/validation.json)：输出核对和当前相机的 AABB 入镜审计。
- [metadata](/mnt/d/data/deepcad_stl_total/metadata)：实际执行的转换脚本快照、SHA-256、Python 路径和 Git 状态；执行时脚本尚未提交，不能只凭 Git commit 复现。

转换脚本：[prepare_stl.py](/mnt/d/python/packages/cad_scene_factory/prepare_stl.py)。运行环境是已有 `/mnt/d/anaconda3/envs/torch/bin/python`，Python 3.10.13、trimesh 4.5.3、NumPy 2.0.1，无新依赖安装。复现命令见 [包 README](/mnt/d/python/packages/cad_scene_factory/README.md)。

## 已发现：精确内容重复

导出 STL 有 4,590 组字节完全相同的文件，涉及 20,639 个源编号。其中 **316 组横跨研究 split，涉及 739 个已分配编号**（另有未分配编号也在这些组内）。

| split 对 | 横跨该对的精确重复组 |
|---|---:|
| training / validation | 250 |
| training / testing | 71 |
| validation / testing | 11 |

一个组可能覆盖三个 split，所以上表不能相加当作唯一组数。示例：`00002718` 在 testing，`00002719` 在 training，两者导出 STL 字节相同。

完整分组见 [duplicate_stl_groups.json](/mnt/d/data/deepcad_stl_total/duplicate_stl_groups.json)，带 split 的跨集分组见 [cross_split_duplicates.json](/mnt/d/data/deepcad_stl_total/cross_split_duplicates.json)。这是精确 STL 内容检查，不是近似几何去重；STL 相同也不证明 native B-Rep 的拓扑表示相同。既有“deduplicated”文件名不能据此被解释成已保证几何无泄漏。未自动删除、合并编号或重划 split。

## 已发现：现有相机与源归一化范围

源坐标整体约在每轴 `[-0.9, 0.9]`，最大高度约 1.8；支撑世界高度约在 `[-0.9, -0.00007844]`。这些明显大于旧提示词示例的 `[-0.5, 0.5]` 范围。

保持相机 `(2, 2, 2)` 看向原点、垂直 FOV 45°、512×512，按现有保守 AABB 检查：

| split | 通过入镜检查 | 未通过 |
|---|---:|---:|
| training | 41,650 | 4,977 |
| validation | 3,091 | 414 |
| testing | 2,161 | 263 |
| unassigned | 44,075 | 6,273 |

共 11,927 个未通过，其中研究 split 有 5,654 个。失败指 AABB 八角不能全部进入画幅，不等于已观察到每个实际网格被裁切。未过滤这些模型，也未调整相机或坐标。首轮模板前需讨论统一渲染归一化及其与 CAD 标签的变换记录，或者重新确定相机；仅扩大背景不能解决入镜问题。

## 验证与边界

少量真实数据验证了 PLY 三角形逐项保留、STEP 回退、缺失来源、同编号来源冲突、清单重复与跨 split 编号处理。全库导出时核对全部三角形在 float32 精度下的坐标及绕序；最终 STL 编号集合与 manifest 一致，文件长度符合二进制 STL 格式，三份清单副本与原文件完全一致。

独立从磁盘读回 11 个覆盖不同 split 和位置的真实模型，对比源 PLY 三角形和记录的输出摘要，均通过；另核对了两组跨 split 重复文件的实际磁盘摘要。未宣称全库 B-Rep 合法、PLY/STEP 已逐件配准、曲面法线适合照片渲染或照片真实感提升。

## 已完成：Blender 5.2.2 LTS

- 安装目录：[blender-5.2.2-linux-x64](/mnt/d/natural/blender-5.2.2-linux-x64)。可执行文件为该目录下 `blender`。
- 原始压缩包：[blender-5.2.2-linux-x64.tar.xz](/mnt/d/natural/blender-5.2.2-linux-x64.tar.xz)，383,295,504 字节。
- 官方 SHA-256：`84098912789dc450e95697c4184fb8a90acbe5111c2ba4aede3fecb57806a168`，已逐字节核验。
- 实际版本 `5.2.2 LTS`，构建 hash `d13f752e3b9c`。无界面启动通过，识别 8 张 RTX 4090，选一张 OptiX GPU 完成 32×32、2 samples 的功能渲染；不是照片级质量验证。
- 完整下载与运行记录：[blender-5.2.2-download.json](/mnt/d/natural/blender-5.2.2-download.json)。保留原 4.2，没有切换系统默认路径，也未把现有生成协议改成 5.2。

官方文件来源：[发行包](https://download.blender.org/release/Blender5.2/blender-5.2.2-linux-x64.tar.xz)、[校验文件](https://download.blender.org/release/Blender5.2/blender-5.2.2.sha256)、[5.2 LTS 说明](https://www.blender.org/download/releases/5-2/)。

## 候选方案：素材包与完整背景，尚未实现

Codex 可选择已有素材、编写程序材质和场景几何、设置贴图并将素材放入模板；照片纹理、扫描模型和实测 HDRI 需要真实文件或相应获取工具，不能用文字声明替代。首轮可提供小型本地素材库，让 Codex 选取并复制到 `assets/`，记录来源、许可、尺寸与摘要；脚本使用相对路径。未来若允许在线获取，还需提供检索／下载／缓存入口与相应运行权限，不能只修改提示词。目前协议禁止 agent 自行下载，未修改该规则。

最小模板包可包含两个构建脚本、`assets/`、简单素材清单、外围构建的 `scene.blend` 和预览记录。已有 `pack_all()` 可打包已加载的贴图；保留源素材仍有助于从脚本重新构建。首轮零件材质继续使用参数 PBR，场景可使用程序材质与外部 PBR 素材。

背景要求应覆盖整个画幅及允许的支撑高度范围。室内场景需合理的墙面、地面和窗外环境；室外可采用完整地面及天空／HDRI。默认灰色 World、无限空白、透明边缘或露出布景边界不能视为完成。关闭透明背景只能保证像素不透明；还需要预览与背景覆盖检查，并与零件遮挡检查分别判断。

参考：[Poly Haven 素材许可](https://polyhaven.com/license)、[Blender World Environment](https://docs.blender.org/manual/en/latest/render/lights/world.html)、[Codex 运行权限配置](https://learn.chatgpt.com/docs/config-file/config-reference)。
