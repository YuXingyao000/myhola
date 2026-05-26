# Rotation Basis Findings - 2026-05-26

## 这次真正确认的结论

今天要解决的问题不是简单的“图片顺时针旋转 90 度”，而是有三套概念被混在一起了：

- Blender cube24 的 view id：`render_cube24.py` 输出的 `00.png..23.png`，这是现在希望采用的标准。
- OCC / SVR 旧 `imgs.npz` 的 euler64 id：旧数据里 `svr_imgs/sketch_imgs` 是 64 个 Euler 旋转，id 不是 Blender cube24 id。
- `single_view.npz`：这是单视图数据，不是 cube24 的第 0 张。实测它对应的是 Blender cube24 的 identity view，也就是 id 18。

正确的 basis 应该是：

```text
所有 24-view 数据的 index i 都表示 Blender cube24 的第 i 张图。

svr_imgs[i] / sketch_imgs[i] / mvr_imgs[..., i] / FLUX cube24[i] / Blender cube24[i]
都必须是同一个 Blender rotation id = i。
```

也就是说，Blender 的 `00.png` 就应该对应迁移后所有条件里的第 0 张：SVR 第 0 张、sketch 第 0 张、FLUX cube24 第 0 张、AE cache 第 0 个 latent。

## 关键映射

旧 euler64 数据迁移到 Blender cube24 basis 时，应该用下面这张表从旧 `imgs.npz` 取图：

```python
BLENDER24_TO_EULER64 = [
    4, 14, 17, 25, 12, 6, 27, 19,
    15, 7, 11, 3, 5, 13, 1, 9,
    26, 16, 0, 10, 18, 24, 8, 2,
]
```

含义：

```text
new_imgs["svr_imgs"][blender_id] = old_imgs["svr_imgs"][BLENDER24_TO_EULER64[blender_id]]
new_imgs["sketch_imgs"][blender_id] = old_imgs["sketch_imgs"][BLENDER24_TO_EULER64[blender_id]]
new_imgs["mvr_imgs"][camera_id, blender_id] = old_imgs["mvr_imgs"][camera_id, BLENDER24_TO_EULER64[blender_id]]
new_ae_cache/{model_id}_{blender_id}/features.npy = old_ae_cache/{model_id}_{euler64_id}/features.npy
```

其中：

```text
Blender cube24 id 0  -> legacy euler64 id 4
Blender cube24 id 18 -> legacy euler64 id 0
legacy euler64 id 0 -> identity -> Blender cube24 id 18
```

所以之前说“Blender 0 对应 cube24 18”的表达是不准确的。更准确地说：

```text
legacy euler64 0 / single_view identity 对应 Blender cube24 18。
Blender cube24 0 对应 legacy euler64 4。
```

## Blender cube24 与 OCC cube24

用 `blender_cube24_sample` 里的两个样例做过对照：

- `00000797` 单模型 silhouette 匹配显示 Blender24 -> OCC24 是 identity。
- `00005807` 单模型因为几何对称，24x24 匹配有歧义。
- 两个样例聚合后，Blender24 -> OCC24 恢复为 identity。

结论：

```text
如果 OCC/SVR 的 24-view 是用同一套 cube24 枚举生成的，
那么 Blender24 id 和 OCC24 id 可以按 identity 对齐。
真正需要映射的是 legacy euler64 -> Blender24。
```

## single_view / FLUX 的区别

`single_view.npz` 不是 cube24 的 0 号 view。

实测：

```text
single_view.npz["blender"] best match = Blender cube24 id 18
single_view.npz["flux"]    应该跟随同一个 single-view camera，也就是 id 18 语义
```

因此：

- `single_view.npz` 只能作为 single-image / real-photo 条件。
- 不应该把 `single_view.npz["flux"]` 当成 cube24 id 0。
- 如果要 24-view FLUX，应使用 cube24 FLUX 数据，也就是 `natural.npz` 里的 24 张栈，而不是 `single_view.npz`。

今天检查到旧条件目录中已经存在一些 `natural.npz`，例如：

```text
/mnt/d/data/deepcad_v6_cond/00000797/natural.npz
  natural_imgs:                  (24, 1024, 1024, 3)
  natural_imgs_compress:         (24, 224, 224, 3)
  blender_natural_imgs:          (24, 1024, 1024, 3)
  blender_natural_imgs_compress: (24, 224, 224, 3)
```

这个文件才是 cube24 FLUX / Blender 图像迁移时应该保留的对象。它的 index 应该已经是 Blender cube24 的 `00..23` 顺序。

## 迁移目标应该长什么样

如果以 Blender rotation 为唯一 basis，建议迁移后的每个 model 保持下面的语义：

```text
cond_root/{model_id}/imgs.npz
  svr_imgs:    (24, 224, 224, 3), index = Blender id
  sketch_imgs: (24, 224, 224, 3), index = Blender id
  mvr_imgs:    (8 * 24, 224, 224, 3), layout = camera_id * 24 + Blender id

cond_root/{model_id}/natural.npz
  natural_imgs / natural_imgs_compress: FLUX cube24, index = Blender id
  blender_natural_imgs / blender_natural_imgs_compress: Blender cube24, index = Blender id

cond_root/{model_id}/single_view.npz
  optional; only for single-view experiments, not cube24 id 0

ae_cache_24/{model_id}_{blender_id}/features.npy
  latent rotation id = Blender id
```

如果继续使用当前 `dataset.py` 的旧读取路径，它目前默认读的是：

```text
imgs.npz["svr_imgs"]
imgs.npz["sketch_imgs"]
imgs.npz["mvr_imgs"]
single_view.npz
sketch_and_natural.npz
img_feature_dinov2.npy
```

所以迁移脚本如果输出成 `svr.npz/sketch.npz/mvr.npz/real_photo.npz`，除非同步改 dataset，否则训练代码不会自然读到这些文件。

## img_feature_dinov2.npy 也需要迁移

旧 `img_feature_dinov2.npy` 实测 shape 是：

```text
(640, 1024)
```

这对应旧布局：

```text
64 svr + 64 sketch + 8 * 64 mvr = 640
```

如果迁移到 24-view basis，cached condition 也应该重排为：

```text
24 svr + 24 sketch + 8 * 24 mvr = 240
```

否则开启 `cached_condition=True` 时，虽然图片迁移对了，DINO feature 仍会按旧 euler64 顺序错配。

## 当前 tools 的风险记录

今天对下面两个文件做过修改，但最后发现输出结构不符合你想要的“Blender id 作为所有数据共同 index”的直觉语义：

```text
tools/migrate_64_to_24.py
tools/debug_rotation_migration.py
```

主要问题：

- 迁移脚本曾输出 `svr.npz/sketch.npz/mvr.npz/real_photo.npz`，但当前 dataset 更自然使用 `imgs.npz/natural.npz/single_view.npz`。
- `real_photo.npz` 复制自 `single_view.npz`，这会把 single-view identity 的语义和 cube24 数据混在一起。
- debug 输出里把 single_view 标成 `blender24_18` 是事实记录，但不应该让迁移后的 24-view 主数据也围绕 single_view 组织。
- 还没有完成对 `natural.npz` 和 `img_feature_dinov2.npy` 的迁移处理。

因此，在继续正式迁移前，应先审查或恢复这两个 tools，再按本文件里的目标格式重写。

## dry-run / debug 结果

曾跑过全量 dry-run，结果：

```text
total models: 53225
ok: 52556
skip_missing_imgs: 669
skip_incomplete_cache: 0
```

含义：

- 52,556 个模型具备旧 `imgs.npz` 和完整 24 个所需 euler64 latent cache。
- 669 个模型缺旧 `imgs.npz`。
- 按所需 24 个 euler64 id 检查时，没有发现 AE cache 不完整。

## 后续建议

先不要重新生成 Blender / FLUX；它们太重，而且 cube24 顺序本身是可以作为 basis 的。

最稳妥路线：

1. 以 Blender cube24 `00..23` 为唯一 id 标准。
2. 迁移旧 SVR/sketch/MVR：按 `BLENDER24_TO_EULER64` 重排，落回 `imgs.npz`。
3. 复制或保留 cube24 `natural.npz`，确认 index 已是 Blender id。
4. 迁移 AE cache 到 `{model_id}_{blender_id}`。
5. 同步修改 dataset：如果 latent_root 指向 `_24` cache，就不要再做 `cube24_to_euler64`，直接用 Blender id。
6. 对 `img_feature_dinov2.npy` 做 640 -> 240 重排，或禁用 cached condition 重新生成。
7. `single_view.npz` 保留为单图条件，不参与 cube24 id 0 的对齐。

