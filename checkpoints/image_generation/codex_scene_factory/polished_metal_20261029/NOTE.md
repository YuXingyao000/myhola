# 抛光金属反射对照

## 研究问题

用户允许改变反射属性，要求几何不动。此前 LUT、AgX look、有限的重打光及程序纹理对照没有找到明确的统一照片化方案。本轮检验一个具体机制：原金属统一使用 roughness=0.32，是否把环境反射过度模糊；降低平均粗糙度和引入粗糙度贴图各自贡献多少。

这是渲染阶段的材质实验，需要重新计算光传输，不是只处理已有 PNG 的 Step4 滤镜。没有修改生产脚本、源场景、默认材质或训练数据。

## 输入与变量

固定两个 CAD（00301161、00489180）与办公室、机加工车间、机械装配车间，共六个组合。沿用此前 LUT 对照的实际模型矩阵、相机、灯光和色彩管理；两个 CAD 均来自原 training 清单，是外观开发对照，不是 test 或下游重建评估。

| 材质 | Roughness | 其余条件 |
|---|---|---|
| baseline | 0.32 | 原始均匀金属 |
| polished_constant | 0.0549445358 | 素材粗糙度像素的均值 |
| polished_map | 素材原始粗糙度图 | Non-Color，BOX 投影，blend=0.2 |

三组均保持 linear Base Color=(0.55,0.57,0.60)、Metallic=1，不修改 Normal、Displacement、各向异性、顶点、面、法线、姿态或相机。512×512、Cycles CPU、8线程/进程、128 samples、seed=8，沿用 OIDN 与源场景 AgX。每个组合都重新渲染同预算 baseline。

素材为 [ambientCG Metal050A](https://ambientcg.com/a/Metal050A)，CC0；官方工程确认贴图直接接 Roughness。它是程序生成素材，不是实测 BRDF 或扫描精度的证据。像素范围0.03529–0.08627，标准差0.0059551。原素材没有米制尺寸，本实验明确假设纹理宽度0.1m。

贴图与其均值并非严格等效 BRDF：Cycles 的粗糙度变换及 BOX 混合都会影响实际分布；该对照用于检验可见纹理增益，不宣称只改变一个严格独立的频率分量。原色图、法线图、高度图均未使用。

## 实现与复现

- `spec.json`：运行前固定的假设、分母、判读方式。
- `material.py`：三种 Roughness 输入；几何与灯光不参与修改。
- `render.py`：读取旧机位与姿态 → 替换材质 → 渲染 → 验证几何。
- `compare.py`：汇总预定18张，按原512像素拼图，不再调色。
- `../metal_asset_20261029/metal050a_audit.json`：来源及节点连接审查。

运行产物在 `/mnt/d/data/cad_scene_enhancements/polished_metal_20261029_001/`。`request.json` 保存49项输入 SHA256；`source/` 保存实际执行脚本；`assets/` 包含本轮粗糙度图；完整素材来源记录位于 request 指向的 assets.json。原始 source/render.py 的“九个机位”首行说明继承自旧实验，实际循环由 request 的六个组合控制。

已执行的先导命令如下；其余运行只替换 scene/model 列表。脚本要求使用新的输出目录，不覆盖既有样本。

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 \
  /mnt/d/natural/blender-5.2.2-linux-x64/blender \
  -b --factory-startup --threads 8 --python-exit-code 1 \
  --python /mnt/d/data/cad_scene_enhancements/polished_metal_20261029_001/source/render.py -- \
  --request /mnt/d/data/cad_scene_enhancements/polished_metal_20261029_001/request.json \
  --scene factory --models 00489180
```

## 判断范围

网格、角法线和相机摘要，以及 mask/depth 一致性只验证几何未变。它们不保证几何线索在 RGB 中同样清楚，也不是真实感分数。视觉审查分别关注金属可辨性、环境反射、孔槽/轮廓辨识及整体照片感；Agent 看图属于非盲主观观察，人工验收单列。

本轮不因高光更多、反射更清楚或素材有外部来源，就将候选设为默认。

## 完成结果

18/18张完成，无失败、缺失。全部网格/法线/相机摘要一致；候选相对新baseline的mask差异像素为0，深度最大误差为0；与此前原图的mask也一致。49项冻结输入摘要均未改变。CPU渲染耗时按每张相加为682.2秒，实际使用最多三个并行进程，没有使用GPU。

本次baseline与旧64 samples图像存在采样预算差异，最大逐案例PNG平均误差为1.526/255，因此比较图只使用本次128 samples的baseline。贴图与同均值常量的目标区域PNG平均绝对差为0.316–0.979/255，降低平均粗糙度相对原材质的差异为12.149–61.759/255。这些数值只支持变化量判断，不能证明照片感收益。

实际看图得到的结论：抛光材质形成可辨的环境倒影，圆盘细槽更清楚；部分外轮廓却与桌面反射混合，多面零件的孔内明暗和表面边界也更难读。粗糙度贴图相对同均值常量没有明确可见优势。三场景的反射变化说明的是加工表面状态发生改变，尚未证明一个统一的照片化方案。逐例Agent观察见`visual_review.json`，人类评价仍为pending，`selected_default=null`。

这一轮的研究取舍是：保留已有原图及完整对照，不再用缺乏新机制的后处理/粗糙度扫描寻找显眼效果。若继续材质路线，需要先界定目标加工表面及实物参照；在此之前，抛光、拉丝等只作为可控的合成外观候选，不称为已验证的真实感提升。生产默认值保持不变。

工作区render.py只修正了继承来的“九个机位”说明文字，实际运行快照仍完整保留在数据目录source/；渲染逻辑未改变。
