# 零件反射属性消融（2026-10-29）

## 为什么做

用户已明确允许调整微观反射属性，要求几何不动。此前 LUT 调色未获认可；LightGroup 与 PIXLRelight 的有限对照没有给出明确的真实感收益。九张原图里的目标零件使用完全均匀的金属底色与粗糙度，与桌面/工具柜的表面细节形成落差。这是一项视觉诊断，不证明光照完全没有问题。

旧 `surface_finish` 中，提亮基色的 A 与同时增加粗糙度变化、各向异性的 B 使用同一基色。旧 A/B 隔离的是两项反射变化的联合效果，没有分离贡献；其人工验收仍为 pending，不能称为用户已否定。详见 `prior_audit.json`。

## 本次实施

这是渲染阶段的材质消融，不是纯二维 Step4 修图。生产脚本、源场景均不修改。

固定此前 3 个零件 × 3 个场景的实际相机、模型矩阵和材质底色，CPU、8线程/进程、64 samples、seed 8、512×512、OIDN。所有候选同预算，保留场景 AgX 设置。

| 候选 | 改动 |
|---|---|
| baseline | 原材质：linear RGB (.55,.57,.60)，metallic=1，roughness=.32 |
| roughness | 仅粗糙度变为 `.32 + .10*(2*Noise.Fac-1)`；物体局部尺度25/.8/.8 mm |
| anisotropy | 仅各向异性，最大.5；切线取局部X在表面的投影，强度按法线方向衰减 |

不连接 Normal / Displacement，不修改法线、顶点、三角形、UV、灯光、底色或相机。材质改变后仍重新计算光传输，因此零件反射到环境的微小变化是渲染结果的一部分。

纹理尺度和强度是预先冻结的诊断参数，不是测量得到的加工工艺参数。局部X固定方向不代表所有曲面都应采用同一种加工纹理；不能把本候选称为已验证的铣削/拉丝材质模型。

## 文件与复现

- `spec.json`：三臂与27张分母，开始渲染前冻结。
- `material.py`：只构造材质。
- `render.py`：恢复既定机位 → 换材质 → 渲染 → 验证几何。
- `compare.py`：完整样本汇总和原尺寸拼图。
- 数据：`/mnt/d/data/cad_scene_enhancements/reflectance_20261029_001/`。

数据目录 `request.json` 保存源场景、STL、旧相机/输出和脚本摘要。`source_pilot/` 保留先导脚本；`source/` 是后续脚本，仅修正PNG指标命名和成功/失败状态写入顺序，未改变渲染行为，见 `script_revision.json`。

运行脚本要求每次提供独立未存在的样本目录。下面是已执行的先导命令，其余场景使用相同入口、对应 `--scene/--models`，不可直接覆盖原运行目录：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 \
  /mnt/d/natural/blender-5.2.2-linux-x64/blender \
  -b --factory-startup --threads 8 --python-exit-code 1 \
  --python /mnt/d/data/cad_scene_enhancements/reflectance_20261029_001/source_pilot/render.py -- \
  --request /mnt/d/data/cad_scene_enhancements/reflectance_20261029_001/request.json \
  --scene factory --models 00489180
```

## 判读边界

几何通过条件是网格/角法线/相机摘要一致、同次实验 baseline 与候选的 mask 和深度逐像素一致。这保证了该渲染几何不变，不等于孔槽在所有材质下都同样容易辨认。

PNG差异仅用于描述变化量，不是场景线性辐射差，也不是真实感指标。baseline 对旧PNG的重渲差异单独报告。对照图使用本次等预算重渲的 baseline，避免混入旧渲染差异。

先导观察：圆盘正面出现明显拉丝式反射，侧环为细颗粒；原有大尺度高光仍存在。纹理覆盖较均匀、幅度偏强，有程序纹理感的风险。各向异性主要改变细微明暗，尚未体现稳定真实感收益。最终数量、全组结果与人工状态见 `result.json`。

## 完成结果

27/27完成，无失败/缺失。所有候选对本次baseline的mask差异像素为0、深度最大误差为0，网格、角法线、模型/相机矩阵摘要完全一致。源文件摘要不变。重渲baseline对旧PNG的最大逐案例平均绝对误差为0.153/255，故图板统一使用本次baseline。27张渲染时间合计551.7秒；采用三个CPU进程并行，没有使用GPU。

三个CAD均来自原`deduplicated_deepcad_training_7_30.txt`，本轮是开发外观对照，不是独立test评估。不是新模型训练或已证明的下游性能提升。

全量看图：粗糙度变化在9组都可辨，但部分形状出现方向条纹与颗粒面的突变，尚不能作为统一照片化默认。各向异性没有明确的稳定收益。逐例的Agent主观观察见`visual_review.json`；参与者非盲评，用户结论仍待定。
