# 2026-10-28：先增强空背景，再合成零件光影

后续[目标程序材质对照](../surface_finish/NOTE.md)已完成：只提亮/增加轻微方向性
没有补足金属层次，未升级材质默认。这里的背景候选保持不变。

本轮实际完成6组配对渲染、12次背景增强，以及每次增强后的两种合成，共24张
主要合成结果。当前值得保留的候选是 **空背景 + Canny约束 + 相对明暗合成**。
六件未观察到此前新增底座、拱内多生横板或十字下方脱离的生成阴影；
但这不是任意输入都安全的保证，也没有解决原零件表面偏平的问题。

两种模型和两种合成公式全部保留。没有挑seed、重抽失败例、升级生产默认或扩大100件。
当前六件属于既有training开发输入；不是独立测试或实拍质量评估。

## 这次改变的机制

之前让生成器看到CAD并修改其周围；现在生成器只看到完全移除CAD的同机位空背景。
Blender分别拍摄有零件图I和空背景图B；生成器将B变成E。
合成阶段转移I与B的差别，并将零件mask及其2px欧氏外扩区域逐字节恢复为I。
因此生成器不负责理解薄片、孔槽或为这些形状补画接触影。

- `render_pairs.py`：冻结原场景、STL、相机和材质，输出配对RGB、mask、Z和EXR。
- `enhance_plates.py`：固定六件、只增强空背景，调用同一加载器，保存两种合成。
- `composite.py`：纯CPU光影差分合成；不调用生成模型。
- `profiles.json`、`../krea_office_surfaces.md`：参数与提示词独立存放。
- `make_comparisons.py`：全量等尺寸排图和带来源摘要的展示清单。

公共`step4/model.py`新增可选`control_image`及`control_guidance_end`，保留旧depth调用。
研究驱动新增`flux_control`别名；原生产CLI与默认配置未升级。

## 实际比较

| 比较 | 观察 | 判断 |
|---|---|---|
| 直接相加 vs 相对明暗 | 00301161直接相加产生黑红条纹，00489180接触附近也有细纹；相对明暗明显更连续 | 淘汰直接相加作为当前推荐，但保存其失败证据 |
| Depth vs Canny | Canny中的键盘、显示器底座、机箱和打印机更规整；Depth常较自由地重画这些物品 | 当前优先Canny；不把所有背景内容变化都视为失败 |
| 新方案 vs 原masked增强 | 十字下方的脱离暗条消失；没有重新生成底座或拱内横板 | 分离背景生成和CAD光影值得保留 |
| 原零件外观 | 薄片、圆盘与拱仍较平，金属颜色受原木桌/地毯反射影响 | 这是被原像素保护保留下来的源渲染局限，背景增强不会自动修复 |

主要观察来自完整六件实际看图；独立审图在`visual_review_depth.json`和
`visual_review_canny.json`。背景机箱型号、纸张或咖啡内容变化属于内容重解释，
不单据此判失败。明显破碎/不合理的物品、CAD位置新增占位物、支撑或接触错误
才是与当前用途直接相关的失败。Canny仍会改写背景，不能作为严格几何锁。

| CAD | 主要观察（相对明暗合成） |
|---|---|
| 00301161 | 木纹更细，原接触阴影较连贯；Canny键盘比Depth更规整；零件仍显均匀、偏暖。 |
| 00489180 | 圆盘细槽保持；Canny机箱不再重画成明显多层堆叠；底部仍需看高倍边界。 |
| 00495463 | 打印机在Canny中更规整，未在薄片附近新增结构；零件仍非常薄。 |
| 00182999 | 原masked增强的脱离暗条未重现；十字仍像灰色薄片，接触感弱在源图中已存在。 |
| 00269315 | 拱内部显示新木纹，没有新横板；杯子/主机保留可识别结构，反光仍需人工判断。 |
| 00320493 | 木面纹理更接近照片，键盘与主机较规整；平板本身仍是均匀色面。 |

## 数值、资源与边界

- 渲染6/6，48/48文件。新旧CAD mask完全一致、空图mask全零；相机和模型变换、
  接触检查通过。180条独立射线最大轴向Z误差为2.45e-6m。
- 渲染从原64 samples提高至256；新I与旧RGB的MAE为0.5501–0.7525/255，
  不能说原RGB字节不变。本轮保护承诺针对新的配对图I。
- 两路各6/6完成；512×512，40步时间表实际26步，strength=.65，guidance=4.5，
  增强seed8–13；Depth控制.5/end1，Canny控制.7/end.8、阈值100/200。
- Depth实际去噪累计45.14秒、峰值15.74GiB；Canny45.84秒、16.77GiB。
  不含加载、摘要、渲染、保存与制图。只使用GPU7。
- 24张主合成均通过目标和2px保护区字节一致检查；恒等E=B时还原I。
  身体mask内不变不代表整图几何正确，孔内新背景和外部遮挡仍需看图。
- 直接相加在00301161有0.491%/0.562%的通道值小于0，裁剪后出现可见条纹。
  其中保护圈外实际保留在最终合成中的负通道为1549/1755个；相对明暗该件为0。
  这支持解释伪影，但不单独证明全部条纹的成因。
  具体裁剪位置和两种公式重算见`enhancement_audit_20261028.json`。
- 独立审计复算168张PNG、核对12张对照图，31个模型/配置文件摘要全部匹配。
  CAD外扩使用独立13偏移模板重算；24张主合成及其诊断图逐字节一致。
  Canny的end=.8在本次26步有效时间表里对应前20步施加控制；没有mode embedding，
  不传control_mode是本权重的接口约定。
- PNG已烘焙AgX。逆sRGB得到的只是display-linear，不能当作原始辐射值。
  EXR虽已保存，本轮合成未使用它。新背景若大幅改变几何或光源，原光影不能自动匹配。
- 35个外部素材文件在渲染后补核摘要；34个匹配已有清单，1个日程贴图补记。
  不追溯声称它们都在本轮request中预先冻结。详见`pair_capture_audit_20261028.json`。
- 已知非本轮输入边界：mask外扩后占满图像时，合成诊断的空数组统计会失败。
  当前六件保护面积仅2435–12226/262144，不涉及该情况；未加入额外通用防御代码。

Depth与Canny不仅控制图不同，还更换了ControlNet和控制强度，因此是工程方案比较，
不能声称这是单变量的Canny因果实验。新Union Pro 2模型版本/来源/摘要在
`/mnt/d/model/Flux1_ControlNet_Union_Pro_2/source.json`，各次request保存实际权重摘要。

## 产物与复现

路径根均为`/mnt/d/data/`：

- 配对：`cad_scene_renders/office_phone_pairs_20261028_001/`
- Depth：`cad_scene_enhancements/office_phone_plate_depth_20261028_001/`
- Canny：`cad_scene_enhancements/office_phone_plate_canny_20261028_001/`
- 全量三页对照：`cad_scene_enhancements/office_phone_plate_comparison_20261028_001/all_00.jpg`至`all_02.jpg`
- 进展对照：同目录`progress.jpg`，包含改善例00301161及未解决薄片感的00182999。

`progress.jpg`左为新的256样本配对图，中为旧64样本输入的masked增强，右为本轮Canny。
它只用于展示进展；原始输入不同，不把这张排图当作单变量实验。

```bash
OMP_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1 \
/mnt/d/anaconda3/envs/generate/bin/python \
 /mnt/d/python_room_generation/checkpoints/image_generation/codex_scene_factory/enhancement/phone_camera/background_plate/enhance_plates.py \
 --pairs_dir /mnt/d/data/cad_scene_renders/office_phone_pairs_20261028_001 \
 --profile canny \
 --output_dir /mnt/d/data/cad_scene_enhancements/office_phone_plate_canny_NEW_RUN
```

Depth用同一入口的`--profile depth`和另一个新目录。脚本准备后执行保存的源码快照。
完整配置、提示词、输入、权重、版本、Git和源码摘要在各输出`request.json`。

## 与修图路线的关系

本轮还检验了对已有增强结果再做DeepLPF和受限局部增益：12/12完成，主要改变影调，
没有解决材质；官方结果新增0.483%–2.335%全黑像素。暂不作为默认后处理。
详见[修图后接实验](../../phone_retouch_local/REFINEMENT_NOTE.md)。

背景照片化与目标真实感应分开判断。当前结果支持继续保留分层生成方式，
没有证据证明整体已达到用户所需的手机随手拍观感。下一项明确缺口是目标表面与
原场景光照的组合：应在几何不变的前提下改善其可观察的金属反射，而非继续放大
背景生成强度来掩盖。最终人工判断仍pending。
