# 2026-06-17 实验笔记

## 今日主题

**没有跑实验。**今天的核心产出是一次任务定义层面的反思，和本周（剩 3 天）的重新规划。

---

## 方向反思（Phase 2，按 [planning skill](../../skills/brepnet-experiment-planning/SKILL.md)）

### 反思 1：长期目标卡在哪一步

North Star 没变：`image -> (topology mask) -> HoLa diffusion -> B-Rep`。

但瓶颈位置在今天被重新识别：

- **不是** diffusion model 不够强：oracle topology mask 接入后 100 测试集 validity = 100%，几何 / 重建指标也很好（[2026-05-25 系列实验](../2026-05-25/NOTE.md) 和后续 oracle 复现）。这意味着当前 HoLa diffusion 拿到正确 topology constraint 就能稳定生成合法 B-Rep。
- **不是** 06-08 topology VAE decoder 不够强：[06-12 周报](../2026-06-12/weekly_report_0606_0612.md) 的 corruption-only 在 ar_recon F1=0.9479 / valid=0.9922 / exact=0.7826。decoder 能力上限够。
- **是** image → topology mask 这一段：[06-11](../2026-06-11/NOTE.md) / [06-13](../2026-06-13/NOTE.md) / [06-14](../2026-06-14/NOTE.md) / [06-16 审计](../2026-06-16/NOTE.md) 三种结构都没把 `sample8_best_f1` 推过 0.7242，face_count_acc 卡在 0.48。

但今天最关键的发现不是「瓶颈在哪」，而是「我们一直在解决的任务定义本身是错的」。

### 反思 2：anti-local-loop 检查

按 [planning skill](../../skills/brepnet-experiment-planning/SKILL.md) Anti-Local-Loop trigger：

- 06-11 / 06-13 / 06-14 三天**都在调** image-conditioned CVAE 结构（in-context / frozen prior / gated adapter），同一个目标指标 `sample8_best_f1`，没有 breakthrough。
- 这条 trigger 已经命中：**不应再提同类 adapter 变种**。

我（Claude）今天最初提的下一步实验是 "把 06-11 mask 接到 HoLa diffusion 跑 no-mask / cvae-mask / oracle-mask 三方 ablation"。这跳出了 adapter-loop，但**仍停留在原来的任务定义上**——假设 CVAE 的"目标"就是去 fit GT face_adj，只是想验证这个目标值不值得做。

用户在反思中纠正了这一点。这是 [planning skill](../../skills/brepnet-experiment-planning/SKILL.md) 当前还没覆盖的盲点：跳出 metric-loop 不等于跳出错误的任务定义。

### 反思 3：任务定义合理性（new, 这是今天的核心）

**当前 CVAE 训练 loss 假设**：

```python
loss = sum over (i, j) ∈ all 435 face pairs:
    CE(predicted_pair_token[i,j], gt_pair_token[i,j])
```

每一对 face pair 都被视作"必须严格 match GT"的硬约束。

**但实际任务是 1-to-many**：

- 输入：单张 image（最多看到一半 face，其余 occluded）。
- 期望输出：(a) 在图像可见的 face 区域，topology 和 GT 一致；(b) 在不可见区域，**多面少面、怎么连都行**，只要整体合法（connected ∧ no_isolated）就 OK。

也就是说，GT face_adj 只是 "众多合法 topology 中的一个采样"。但当前 loss 强迫 model 在所有 435 对 face pair 上都完全 match 这一个采样——**包括不可见区域**。这不是 1-to-many supervision，是 1-to-1 supervision 喂给一个 1-to-many task。

**症状反推**：

| 观察 | 用 1-to-1 loss 解释 |
|-----|---------------------|
| `cond_prior_mu_edge_f1=0.55` 上不去 | 不可见区域必然瞎猜，f1 永远到不了 1 |
| `face_count_acc=0.48` | 同一物体不同视角对应不同合法 face count；强迫单一答案必然错一半 |
| `sample8_diversity` 06-14 比 06-11 还低（0.27→0.22） | model collapse 到平均答案，loss 不允许探索其他合法解 |
| `ar_recon_f1=0.91` 但 `cond_prior_mu_f1=0.55` | decoder 本身能 fit topology，但 image prior 在不可见区域无信息可学，被迫乱猜 |

### 反思 4：今日方向选择

按 skill 的 Phase 2 question 4：「继续推进上一条 / 跳出局部、换下一个瓶颈 / 重新定义任务本身」。

**选择：重新定义任务本身**。

理由：

- 在错误的 supervision target 下做 downstream ablation（Claude 最初提的方案）会得到一个有歧义的数字——cvae-mask 输给 oracle-mask 时无法分离 "CVAE 不够强" 和 "CVAE 在错的目标上学" 这两个原因。
- 在错误 supervision target 下继续做结构变种（spanning tree / count head / latent diffusion prior）也是解决错的问题，sample diversity 持续下降的趋势会被同样的 loss 打回去。
- 任务定义的修正方向是清晰的：**把 supervision 从 "fit 整张 GT face_adj" 改成 "fit visible region of GT + global validity"**。

---

## 长期研究目标（用户视角，记录下来防止后续偏离）

### 真正在做的是什么

用单张照片生成合法、视觉一致的 B-Rep CAD 模型。"视觉一致"的精确含义：

- **强约束**：照片视角能看到的 face / 面之间的 adjacency / 边界形状必须和照片匹配。
- **弱约束**：照片视角看不到的 face（背面 / occluded 区域）只要让整个 B-Rep 合法，可以任意。多一个面、少一个面、怎么连都允许。

这不是 reconstruction（一对一还原），是 conditional generation（在条件约束下采样合法解）。

### 现在做的两个核心模型

1. **Topology Predictor (CVAE)**：image → topology mask（face adjacency）。今天的反思针对它的 supervision 修正。
2. **HoLa-BRep Latent Diffusion Model**：(latent + topology mask) → B-Rep latent → STEP file。oracle topology 下已经非常强；瓶颈是 topology mask 怎么从 image 来。

**HoLa-BRep VAE 不动**：本阶段不优化 VAE encoder/decoder，只优化 topology predictor 和 latent diffusion 的训练协议。

### 缺什么数据

**Per-face visibility annotation**：对于每一个 (model_id, view_id) 对，标注每个 face 在该视角下的可见性。

可以从已有数据构造（不需要新采集）：
- GT STEP file 已存：`/mnt/d/data/deepcad_v7/{model_id}/data.npz`
- Camera extrinsic 已存（FLUX 生成时使用的视角）
- 用 trimesh / OpenCASCADE face-id buffer rendering 可以构造 `visibility_mask[face_id] ∈ [0, 1]`（visible_fraction）

工程量估计：1-2 天（先做 100-sample 验证，再扩到全集）。

### 任务优先级

P0（今天/明天起做）：
1. **先做"oracle topology mask 上限基准"实验**——快速验证当前 HoLa diffusion + 06-08 topology decoder 在 deepcad_v7 + real_photo 协议下，oracle topology mask 是不是还能保住 100% validity（之前 100-sample 实验在 deepcad_v6 + 白模上做的，需要在新协议复现，作为 visibility CVAE 的 ground truth ceiling）。
2. **visibility annotation pipeline 设计 + 100-sample 验证**（不写训练代码，先确认数据 pipeline）。

P1（下周起做）：
3. visibility-aware CVAE v1 训练（loss 改造）。
4. 跑 visibility-aware CVAE → diffusion mask → STEP file 全链路。

P2（更后面）：
5. 跑 no-mask / cvae-mask (visibility-aware) / oracle-mask 三方 ablation——**这才是有意义的 ablation，因为 cvae-mask 来自正确的任务定义**。

### 本周（剩 3 天）日历

今天是周三 06-17，需要在周五 06-19 交周报。日历：

| 日期 | 计划 | 产出 |
|------|------|------|
| 06-17（今天） | 反思 + 选今日快速实验 + 在服务器上交给 Codex 执行 | 06-17 NOTE（这份）+ ROADMAP / skill 更新 + 服务器侧由 Codex 推进的事项（见下） |
| 06-18（周四） | 根据 Codex 昨晚产出，决定走 visibility-aware CVAE 编码 还是 oracle baseline 训练 | visibility 可行性结论 + （可能）开始训练 |
| 06-19（周五） | 写周报 + 整理本周结论 | 周报 PPT |

**约束**：HoLa latent diffusion 完整训练 ~1 天 → 今天必须挑一个能在 24h 内出数字的实验。

---

## 今日代办（服务器侧，交给 Codex 执行）

> 用户今天回服务器（Ubuntu）后，会让 Codex 接力执行。Codex 应优先把任务 0 跑完再决定后面方向。

### 任务 0（必做，决定后续路线）：确认 oracle 在 deepcad_v7 + real_photo + cube24 协议下是否已经跑过 full training

**为什么必须先做**

- 我（Claude，今天主对话方）在 NOTE 里搜遍 [05-25](../2026-05-25/NOTE.md) / [5-30](../2026-5-30/) / [06-01](../2026-06-01/NOTE.md) / [06-05](../2026-06-05/NOTE.md) / [06-06](../2026-06-06/NOTE.md) / [06-08](../2026-06-08/NOTE.md) / [06-10](../2026-06-10/NOTE.md) / [06-11](../2026-06-11/NOTE.md) / [06-13](../2026-06-13/NOTE.md) / [06-14](../2026-06-14/NOTE.md) / [06-16](../2026-06-16/NOTE.md) / [ROADMAP](../ROADMAP.md)，**没有任何 NOTE 写出 oracle 在 v7 + real_photo + cube24 协议下的 validity 或几何指标**。
- 但 [2026-5-30 launch.json](../2026-5-30/refactored_diffusion/launch.json) 显示该协议的 `train_diffusion_oracle_topology` config 已经在 debug 状态通了（max_steps=1）。
- 用户记忆里有"oracle 100 模型 100% validity"的数字，但很可能是早期 deepcad_v6 + 白模 + euler64 协议（[05-25 commands.sh](../2026-05-25/commands.sh) 的 `topo_bias_white_corrected`）。
- 状态混乱必须先消除：如果 v7 + real_photo + cube24 已经训练过且数字 OK，就**不需要重训 oracle baseline**，今天剩余预算可以全部投入 visibility 路线。

**Codex 具体动作**

1. 登录 wandb，查看 `hola-brep` 项目下含 `oracle` 或 `topo_bias` 的 run。
2. 检查每个 run 的 args：找 `dataset.raw_data_root=/mnt/d/data/deepcad_v7` + `dataset.real_photo_ratio=1.0` + `dataset.cached_latent_root` 含 `11k_24` 的 run。
3. 同时检查 `/mnt/d/data/diffusion_topo_experiments/` 和 `/mnt/d/data/new_cond_ckpt/` 下是否有这个协议的 checkpoint + test_metrics 输出。
4. 找到的话，记录：
   - exp_name 全名
   - validation loss 最低值
   - 如果有 inference 输出：valid STEP rate、几何指标
5. 把找到的数字回写到 [ROADMAP.md](../ROADMAP.md) 的 `当前最优 / 基线` 表里 "End-to-end diffusion" 行（目前是空的）。
6. 同时回写到本 NOTE 的"任务 0 结果"小节（占位见下）。

**判定（决定下一步走哪个）**

- **A. 找到了 + validity ≥ 0.95**：跳过 oracle baseline 训练，跳到任务 1（visibility prototyping）。这是最理想路径。
- **B. 找到了但 validity 在 0.7-0.95**：协议层有 domain gap，但 oracle 通路仍可用作 ceiling。也跳到任务 1。
- **C. 找到了但 validity < 0.7**：oracle 在新协议下不行，CVAE 路线整体要重新评估，先停下来在 NOTE 里补一段分析，等用户决定。
- **D. 没找到（或只有 debug run、没 full training）**：启动 oracle baseline full training（命令见任务 2），同时**并行**做任务 1（visibility prototyping）。

### 任务 1（条件执行：A/B/D 路径都做）：visibility annotation pipeline 可行性 prototyping

**为什么这么做**

- [今日反思](#反思-3任务定义合理性new这是今天的核心) 的核心结论：image→topology 是 1-to-many，CVAE 应只在"图像可见"face 区域贴合 GT，不可见区域只要保 validity。
- 实现这个 supervision 需要 per-face visibility annotation：`(model_id, view_id, face_id) → visible_fraction ∈ [0, 1]`。
- 现在不知道这个标注好不好造、造出来合不合理；**先在 1-3 个 sample 上做端到端的可视化验证**，再决定要不要扩到全集 + 改 CVAE loss。

**Codex 具体动作**

1. 找一个 `model_id`（用 `src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt` 第一行就行）。
2. 读 `/mnt/d/data/deepcad_v7/{model_id}/data.npz`，列出所有字段（`face_adj`、`edge_face_connectivity`、`zero_positions`、可能还有 face mesh 顶点）。同时确认 STEP 文件路径是否存在。
3. 找 svr 渲染时使用的相机外参从哪里来（FLUX 生成时用的视角应该和 svr 视角一致；查 [src/brepnet/data/](../../src/brepnet/data/) 下渲染相关代码，找 cube24 rotation matrix → camera extrinsic 的转换）。
4. 选一个 view（cube24 id=0，就是 real_photo FLUX view）：
   - 把每个 GT face 单独提取成 mesh（OpenCASCADE → trimesh，每个 face 一个独立颜色）
   - 用 trimesh / pyrender 在 cube24 view 0 的相机参数下做 face-id buffer rendering
   - 对每个 face_id 统计 `visible_pixel_count / total_face_area_pixels` = visible_fraction
5. 输出 debug 图：
   - 原 svr 渲染（白模）
   - face-id colored render（每个 face 一个颜色）
   - visibility heatmap（按 visible_fraction 着色）
   - 一个 JSON：`{face_id: visible_fraction}`
6. 在另外 2 个 model 上重复，人眼检查可见性是否合理（被遮挡的面是不是 visible_fraction=0；正面的面是不是 ≈ 1）。
7. 把过程 + 三组 debug 输出 + 可行性结论写进 [06-17/visibility_prototype/](./visibility_prototype/) 子目录，同时回写到本 NOTE 的"任务 1 结果"小节。

**判定**

- **win**：3/3 sample 可见性合理，可以扩展到全集生成 visibility lookup table。06-18 起开始改 CVAE loss。
- **iterate**：2/3 合理，有一个 corner case（比如 spline face 渲染歧义）。06-18 先解决 corner case 再扩集。
- **kill**：visibility 计算本身就不靠谱（face 边界模糊 / OCC mesh 退化），需要换数据源（比如直接从 STEP 取 face boundary 投影到图像平面）。这种情况要写一段 alternative plan。

### 任务 2（条件执行：仅在任务 0 走 D 路径时）：启动 oracle baseline full training

**Codex 具体动作**

```bash
cd /mnt/d/python
python -m src.brepnet.train --config-name train_diffusion_oracle_topology \
  trainer.devices=8 trainer.batch_size=64 \
  trainer.max_steps=100000 \
  trainer.exp_name=20260617_oracle_real_photo_v7_cube24 \
  trainer.output_dir=/mnt/d/data/diffusion_topo_experiments \
  trainer.wandb.enabled=true trainer.wandb.project=hola-brep \
  dataset.raw_data_root=/mnt/d/data/deepcad_v7 \
  dataset.cached_latent_root=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k_24 \
  dataset.condition_data_root=/mnt/d/data/deepcad_v7_cond \
  dataset.real_photo_ratio=1.0 \
  dataset.load_topology=true \
  dataset.padding=random \
  dataset.is_aug=0 \
  model.autoencoder.checkpoint=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt \
  model.topology_bias.mode=soft_bias \
  hydra.job.chdir=false
```

启动前先做 1 step smoke test（同样命令但加 `trainer.max_steps=1 trainer.devices=1 trainer.batch_size=1`）确认 config 通。

训练启动后**不要等**——同时执行任务 1。训练大约 1 天后用同 config 加 `trainer.evaluate=true trainer.resume_from_checkpoint=...` 跑 inference。

---

## 任务 0 结果（Codex 在服务器执行后回填）

> （Codex 填）

## 任务 1 结果（Codex 在服务器执行后回填）

> （Codex 填）

## 任务 2 结果（仅 D 路径才会有）

> （Codex 填）

---

## 今日已完成的更新

1. **[planning skill](../../skills/brepnet-experiment-planning/SKILL.md) Phase 2 加 Question 3**：Task Validity Check。明确询问 "supervision target 是不是和实际任务对齐"。同步到 `.claude/skills/`、`.cursor/skills/` 副本。
2. **Anti-Local-Loop Rules 加第 6 条**：task validity outranks structural search.
3. **[ROADMAP](../ROADMAP.md)**：
   - 时间线加 06-17 row。
   - 开放问题列表把 "1-to-1 vs 1-to-many supervision" 提到 OQ 1。
   - Anti-goals 加两条：不要在错误任务定义下优化结构；不要默认 GT face_adj = ground truth supervision。
   - 下一步候选重排：visibility-aware CVAE 升 P0，topology mask oracle ceiling 实验加为 P0 快速版本。

## 给 Codex 的执行清单（用户回服务器后执行）

按顺序：

- [ ] **任务 0**（必做，先做）：在 wandb / `/mnt/d/data/diffusion_topo_experiments/` 找 deepcad_v7 + real_photo + cube24 协议下的 oracle full training run。回填本 NOTE 的"任务 0 结果"+ 更新 [ROADMAP](../ROADMAP.md) 的 `当前最优` 表 "End-to-end diffusion" 行。
- [ ] **任务 1**（A/B/D 路径都做）：visibility annotation 在 1+2 个 model 上的可行性 prototyping。输出三组 debug 图 + visibility JSON + 可行性结论。回填本 NOTE 的"任务 1 结果"。
- [ ] **任务 2**（仅 D 路径）：启动 oracle baseline full training（命令已给）。先 1-step smoke test 再上 8 卡。

## 之后的待办（06-18 开始）

- [ ] 根据任务 0 + 任务 1 结果，决定 06-18 是「写 visibility-aware CVAE loss + 启动训练」还是「等 oracle 训完 + visibility 扩集」。
- [ ] 06-19 周五周报：本周反思 + oracle ceiling 数据 + visibility pipeline 可行性结论 + 下周路线（visibility-aware CVAE v1）。
