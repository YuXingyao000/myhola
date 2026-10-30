# 2026-10-28：程序化金属表面对照

**结论：本次两种新材质不升级为默认。** 均匀银色主要提亮，定向表面相对同基底
未带来可明确辨认的摄影感收益；十字和竖立薄片反而更接近背景亮度。
背景增强仍保留上一轮Canny+相对明暗方案，本次没有重跑生成模型或轻量修图。

## 实际实施

固定六件、原网格/角法线/机位/姿态/尺寸/照明不变，256 samples。
同一组增强空背景E复用；每种新材质都重新渲染有件图I，再与原empty B做差分合成。
因此新材质的阴影、反射、间接光也重新渲染，不能只把旧阴影作为固定遮罩套用。

两种预设在运行前固定于[SPEC.md](SPEC.md)与[profiles.json](profiles.json)：

- A：linear Base Color由原[.55,.57,.60]改为[.80,.81,.82]，metallic=1、roughness=.32。
- B：同A基底，各向异性最大.30，小幅粗糙度变化±.02；程序坐标按实际毫米转换，
  X向/横向相关尺度20mm/0.25mm。显式切线不依赖不存在的UV。
- 没有Bump、位移、bevel、假刻槽、划痕、污渍；没有改变几何或可见轮廓。

这些是外观假设，不是测量的具体铝合金，也不代表所有面都具有真实加工路径。
512图像下0.25mm细节多数低于一个像素，不能要求它表现成肉眼清晰的刻线。
接口参照Blender v5.2.2官方[Principled节点声明](https://raw.githubusercontent.com/blender/blender/v5.2.2/source/blender/nodes/shader/nodes/node_shader_bsdf_principled.cc)
及[Cycles实现](https://raw.githubusercontent.com/blender/blender/v5.2.2/intern/cycles/kernel/osl/shaders/node_principled_bsdf.osl)。

## 实际观察

全部六件的完整图与2倍同区域放大已检查；[独立审图](visual_review.json)结论一致。

| CAD | 观察 |
|---|---|
| 00301161 | A变亮；B斜面高光稍变，但大正面仍均匀、偏暖，没有明确金属层次改善。 |
| 00489180 | 盘面仍平且反射桌面暖色，B没有明显加工纹理收益。 |
| 00495463 | 薄片变淡，与桌面更接近，B与A视觉差异弱。 |
| 00182999 | 十字A/B更接近地毯亮度，轮廓对比下降；仍不能显出不足1px的侧面。 |
| 00269315 | 拱没有新横板，但表面仍均匀，A/B主要是影调差异。 |
| 00320493 | 平板仍是一块均匀色面；未出现可辨认的材质真实感提升。 |

未观察到新增结构或假刻槽，但“没有改坏几何”不能代替真实感收益。
前轮射线审计已经显示主要镜面方向命中木桌或灰地毯；本轮结果进一步表明，
单靠提高反射基色和轻微各向异性，不能在这些既定视角下补足反射层次。
这不证明程序材质普遍无效，也不能归因于diffusion没有发挥作用——本轮根本没有新采样。

## 技术证据与失败记录

CPU构造检查：[material_validation.json](material_validation.json)。实际渲染与合成
独立审计：[technical_audit.json](technical_audit.json)。

- 12/12渲染完成，48个RGB/mask/depth/radiance文件；12/12合成完成。
- mask与depth解码值、相机及模型矩阵与旧pair完全一致。
- 从源STL重新构造，网格/锐边/平滑标记/角法线摘要均一致；节点逐项复核。
- 12张合成与raw独立重算误差0；目标和2px保护圈与各自新I误差0，越界通道比例0。
- 122输入、28冻结文件摘要通过，六张增强背景未改变。新目标RGB有意改变，
  因而本次不再声称它与原材质图像的像素相同。
- 渲染累计46.07秒，不含准备/场景载入/合成/审计；仅GPU7，结束后空闲。
- 001启动时`import render_pairs`失败，没有渲染任何样本；Blender的`--python`
  不自动将脚本目录放入sys.path。002仅补显式目录入口，配置和seed完全不变。
  001保留`failure_summary.json`，planned12/success0/missing12；002为planned12/success12。
  两次尝试均保留，不把启动失败隐藏成一次无失败运行。

## 代码与复现

- [surface_material.py](surface_material.py)：只构造材质并返回参数，不改对象或场景。
- [render_finishes.py](render_finishes.py)：冻结输入→恢复六件场景→两种材质拍摄→核对几何。
- [compare_finishes.py](compare_finishes.py)：复用增强空背景，CPU合成与全量排图。

输出：`/mnt/d/data/cad_scene_renders/office_phone_finish_20261028_002/`。
`comparison_00..02.jpg`为全部六件，`target_00..02.jpg`为同区域2倍最近邻显示；
`preview.jpg`保留块状件与十字，不只展示较容易的输入。三列依次为原材质、A、B。
冻结源码/配置/输入摘要在`request.json`，不是只保存口头参数。

准备使用研究目录的render_finishes.py `--prepare_only`，输入旧pair目录、旧Canny目录、
`--dependency_audit ../background_plate/pair_capture_audit_20261028.json`和新的output_dir。
随后执行新目录内的冻结脚本：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=8 \
/mnt/d/natural/blender-5.2.2-linux-x64/blender --background --factory-startup \
 --python-exit-code 1 --threads 8 --python <新目录>/render_finishes.py \
 -- --request <新目录>/request.json

OMP_NUM_THREADS=4 /mnt/d/anaconda3/envs/generate/bin/python \
 <新目录>/compare_finishes.py --output_dir <新目录>
```

## 下一步判断

当前不继续扫描颜色、粗糙度或提高划痕强度。更值得检验的是让图像重光照模型
依据背景调整前景的光照与反射层次，并单独检查其形状幻觉；或改变实际渲染的入射
光照构成。这些在本记录中仍是下一步机制候选，不能记作已实现收益。
用户要求的整体手机照片观感仍未完成，人工验收pending。
