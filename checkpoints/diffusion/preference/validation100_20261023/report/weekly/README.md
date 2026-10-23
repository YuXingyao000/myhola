# 周报独立图

数据来自上一层的 `summary.json`，只重新排版，不重新运行模型或修改评估。
灰蓝色为原SVR，绿色为DPO微调5,000步。每张图包含300 dpi PNG、SVG和PDF；
SVG中的文字转为路径，可避免周报电脑缺少中文字体。

| 内容 | PNG | SVG | PDF | 配对样本 |
|---|---|---|---|---|
| 合法实体生成率 | [PNG](01_validity.png) | [SVG](01_validity.svg) | [PDF](01_validity.pdf) | 100件 |
| 后处理前的几何误差 | [PNG](02_geometry_before.png) | [SVG](02_geometry_before.svg) | [PDF](02_geometry_before.pdf) | 100件 |
| 后处理后的几何误差 | [PNG](03_geometry_after.png) | [SVG](03_geometry_after.svg) | [PDF](03_geometry_after.pdf) | 共同70件 |
| 面匹配F1 | [PNG](04_face_matching.png) | [SVG](04_face_matching.svg) | [PDF](04_face_matching.pdf) | 处理前100件、处理后共同70件 |
| 面数量误差 | [PNG](05_face_count.png) | [SVG](05_face_count.svg) | [PDF](05_face_count.pdf) | 100件 |

几何误差是归一化坐标下的双向平方Chamfer距离，数值未缩放，不表示毫米。
后处理前使用VAE直接解码，后处理后使用最终STEP；后处理仅比较双方均合法且三次
几何采样均完整的同一组70件。各自可测的71/89件不混用于该配对比较。
前后样本集合不同，不能把前后两张图当成后处理效果对照。
F1是面匹配的精确率与召回率的调和平均，不是点匹配F1或准确率。
面数量误差是每件生成面数与GT面数差的绝对值的均值，单位为个面。

绘图直接读取 `paired_common` 均值；完整未缩放数值见
[plotted_values.csv](plotted_values.csv)。科学计数标签保留3位有效数字，
F1保留2位百分数小数。几何差异的配对95%置信区间均跨0，图中不作显著改善或退步的结论。

在仓库根目录复现：

```bash
/mnt/d/anaconda3/envs/torch/bin/python \
  checkpoints/diffusion/preference/validation100_20261023/report/plot_weekly.py \
  --font /root/.cache/hola-report-fonts/SourceHanSansSC-Regular.otf
```

中文字体为[Adobe思源黑体SC](https://github.com/adobe-fonts/source-han-sans/tree/release/OTF/SimplifiedChinese)，
本机缓存位于命令所示位置，字体文件不提交。可通过`--font`显式指定其他机器上的同一字体。
已核对导出值与原始汇总一致，并检查中文、科学计数刻度及标签排版。
