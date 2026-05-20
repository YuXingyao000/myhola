# 实验计划表

## 当前进度基线

| 指标 | 原论文 (白模Image Cond) | 你现在 (真实照片) | 目标 |
|------|------------------------|------------------|------|
| Valid Rate | ~0.82 | 0.60 (alignment) / 0.84 (ControlNet) | ≥0.80 |
| Face Chamfer Distance | ~0.001 | ~0.21 | ≤0.005 |
| F1 Score | ≥0.80 | ~0.21 | ≥0.70 |

**注意 Scale**：Chamfer 合格标准是 1e-3 量级，现在差了约 200 倍。F1 合格标准是 0.8+。

**核心发现**：ControlNet 能提升 valid rate 但 Chamfer/F1 不动 → 问题在特征质量，不在注入深度。

---

## 实验路线图

```
Week 1-2                    Week 3-4                   Week 5+
─────────────────────────────────────────────────────────────────
                                                        
 ┌──────────────────┐                                   
 │ Exp 1: Feature   │──── 有效 ────→ 写论文            
 │ Mapper (方案B)   │                                   
 └────────┬─────────┘                                   
          │ 无效/不够                                   
          ▼                                             
 ┌──────────────────┐                                   
 │ Exp 2: KD        │──── 有效 ────→ Exp 1+2 组合      
 │ (方案A)          │                                   
 └────────┬─────────┘                                   
          │ 都不够                                      
          ▼                                             
 ┌──────────────────┐      ┌──────────────────┐        
 │ Exp 3: Topology  │─────→│ Exp 4: VAE改进   │        
 │ Attention Bias   │      │ (TokenVAE/扩展)   │        
 └──────────────────┘      └──────────────────┘        
```

---

## Exp 1: Feature Domain Mapper

**目标**：验证"特征域不对齐"是否是 Chamfer/F1 差的主因。

**假设**：如果把真实照片的 DINOv2 特征映射到白模特征空间，原有的 diffusion 就能生成正确几何。

| 项目 | 内容 |
|------|------|
| 改动 | 只训一个 6-layer Transformer mapper |
| 训练数据 | 配对数据：同一物体的 (real_photo_dinov2_feat, white_model_dinov2_feat) |
| 训练时间 | 2-4小时 (1 GPU) |
| 评估方式 | 训完后 plug 进原 diffusion pipeline，跑完整 eval |
| 成功标准 | Face CD < 0.01, F1 > 0.60 (相比现在的 0.21/0.21 有数量级改善) |

**消融实验**：
- mapper_layers = [2, 4, 6, 8]
- loss = [mse_only, cosine_only, mse+cosine]
- residual_gate 初始值 = [0.5, 0.7, 0.9, 1.0]

**如果失败了说明什么**：
- DINOv2(real) 和 DINOv2(white) 之间不存在简单映射
- 可能真实照片里某些几何信息确实丢失了（背面不可见等）
- → 需要方案A(蒸馏)来在 denoising level 补偿

---

## Exp 2: Knowledge Distillation

**目标**：让 student 在 denoising 的每一步都模仿 teacher 的行为。

**假设**：即使特征层面不能完美对齐，通过逐步匹配去噪行为，student 也能学到正确的生成路径。

| 项目 | 内容 |
|------|------|
| 改动 | Teacher(frozen白模diffusion) + Student(trainable, 从teacher初始化) |
| 训练数据 | 100% 真实照片，每个 batch 需要同时提供白模渲染图给 teacher |
| 训练时间 | 1-2天 (4 GPU) |
| 评估方式 | 用 student 直接推理，跑完整 eval |
| 成功标准 | Face CD < 0.005, F1 > 0.70 |

**消融实验**：
- kd_weight vs diffusion_weight 比例：[1:1, 2:1, 1:2, 3:1]
- Student 初始化方式：[deepcopy_all, reset_crossattn, reset_all_but_backbone]
- 是否加 Exp1 的 mapper 作为预处理

**如果失败了说明什么**：
- 问题不仅是条件域不对齐
- 可能是 VAE latent space 本身的表达力问题
- → 需要改 VAE（Exp 4）

---

## Exp 3: Topology Attention Bias

**目标**：验证"给 diffusion 显式拓扑先验"能否提升 valid rate 和 F1。

**假设**：Diffusion 在高噪声阶段如果有拓扑引导，生成的 face latent 之间的结构关系更正确。

| 项目 | 内容 |
|------|------|
| 前提 | 训一个 TopologyPredictor (image → adj_matrix)，用 GT 训几分钟就行 |
| 改动 | Diffusion backbone 的 self-attention 加 topology bias |
| 训练时间 | 和普通 diffusion 一样 |
| 评估方式 | Valid rate + F1 (拓扑指标) |
| 成功标准 | Valid rate > 0.85, F1 > 0.30 |

**消融实验**：
- Bias 强度调度：[constant, linear_decay, cosine_decay, step_at_500]
- TopologyPredictor 精度的影响：[GT_topo, predicted_topo, random_topo]
- 能否和 Exp1/Exp2 叠加

**可以和 Exp1 或 Exp2 组合**：这不冲突，属于正交改进方向。

---

## Exp 4: VAE 改进（如果前面不够）

**目标**：提升 latent space 的表达力和结构性。

**Sub-experiments：**

| 编号 | 实验 | 改动 | 假设 |
|------|------|------|------|
| 4a | TokenVAE (k=8) | Attention encoder 替代 Conv2D | 去除空间分辨率依赖 |
| 4b | TokenVAE (k=16) | 更大容量 | k=4 太少可能欠拟合 |
| 4c | 扩展 latent (+topo) | 32 → 40 维 | 明确的拓扑子空间 |
| 4d | Face count predictor | 推理时动态长度 | 去除 dedup 的经验阈值 |

**评估**：先在 VAE 层面评 reconstruction quality，再接 diffusion。

---

## 可视化策略

### 怎么挑选 case

**不要随机挑！** 按以下分类系统地挑选：

#### 1. 按 Chamfer Distance 分桶

```python
# 把所有测试结果按 Chamfer 排序
results = sorted(all_results, key=lambda x: x['chamfer'])

# 挑选代表性 case
best_cases = results[:5]           # Top-5 最好的
median_cases = results[len//2-2 : len//2+3]  # 中位数附近5个
worst_cases = results[-5:]          # Top-5 最差的

# 另外：挑选"valid但几何差"的 —— 这是最有分析价值的
interesting = [r for r in results if r['valid'] and r['chamfer'] > 0.4]
```

#### 2. 分析 failure mode 分类

| Failure Mode | 怎么检测 | 可视化要看什么 |
|---|---|---|
| **Invalid topology** (valid=False) | BRepCheck 不通过 | 看 face 之间的 edge 是否闭合 |
| **Valid but wrong geometry** | valid=True, Chamfer>0.3 | 看生成面和GT面的对比 |
| **Face 数量错误** | pred_faces ≠ gt_faces | 是多了还是少了？哪种面漏掉了？ |
| **Topology 正确但位置偏** | F1 高但 Chamfer 大 | 面的相对位置/朝向 |
| **Topology 错误但几何像** | F1 低但面形状对 | 面是对的但连接关系错了 |

#### 3. 给导师的可视化 layout

每个 case 展示一行：

```
[输入图片] | [GT BRep渲染] | [生成的BRep渲染] | [误差热力图] | [指标数字]
```

**每组实验展示 3×5 = 15 个 case**：
- 5 个 best (证明方法能 work)
- 5 个 median (展示典型表现)
- 5 个 worst (分析 failure mode → 指导下一步改进)

#### 4. 对比可视化（最重要！）

```
同一个输入图片:
  [Baseline 结果] vs [你的方法结果] vs [GT]

选择标准：挑"baseline失败但你的方法成功"的case → 证明改进有效
也要挑"你的方法仍然失败"的case → 诚实分析，展示还有哪些挑战
```

---

## 实验结果记录模板

每个实验完成后，填写：

```markdown
## Exp X: [名称]

### 配置
- Checkpoint: xxx
- 关键超参: xxx
- 训练时长: xxx

### 定量结果
| Metric | Baseline | This Exp | Δ |
|--------|----------|----------|---|
| Valid Rate | 0.60 | ? | ? |
| Chamfer | 0.50 | ? | ? |
| F1 | 0.20 | ? | ? |

### 可视化分析
- Best cases: [为什么好？共同特征？]
- Worst cases: [failure mode 是什么？]
- 对比 baseline: [哪类物体改善最大？哪类没改善？]

### 结论 & 下一步
- 假设验证了吗？
- 下一个实验应该改什么？
```

---

## 可视化代码建议

```python
def select_visualization_cases(results: list, num_per_category=5):
    """
    系统性地挑选可视化 case。
    
    Args:
        results: list of dicts, 每个包含 {
            'id': str, 'valid': bool, 'chamfer': float, 
            'f1': float, 'num_faces_pred': int, 'num_faces_gt': int
        }
    
    Returns:
        dict of category → list of case ids
    """
    valid_results = [r for r in results if r['valid']]
    invalid_results = [r for r in results if not r['valid']]
    
    # 按 Chamfer 排序 (只看 valid 的)
    valid_sorted = sorted(valid_results, key=lambda x: x['chamfer'])
    
    cases = {
        'best': valid_sorted[:num_per_category],
        'median': valid_sorted[len(valid_sorted)//2 - num_per_category//2 :
                               len(valid_sorted)//2 + num_per_category//2 + 1],
        'worst_valid': valid_sorted[-num_per_category:],
        'invalid': invalid_results[:num_per_category],  # 一些 invalid 的
    }
    
    # 额外：找 "face count 错误" 的
    face_count_wrong = [r for r in valid_results 
                        if abs(r['num_faces_pred'] - r['num_faces_gt']) > 2]
    cases['face_count_error'] = sorted(face_count_wrong, 
                                        key=lambda x: x['chamfer'])[:num_per_category]
    
    # 额外：找 "拓扑错但几何对" 的 (chamfer小但f1低)
    topo_wrong = [r for r in valid_results if r['chamfer'] < 0.3 and r['f1'] < 0.3]
    cases['topo_wrong_geom_right'] = topo_wrong[:num_per_category]
    
    return cases
```

---

## Dataset Quality Metrics (FLUX.1-Kontext 生成质量评估)

### 主体形状保真度（CAD 形状是否保留）

| 指标 | 计算方法 | 合格标准 |
|------|----------|----------|
| **Silhouette IoU** | SAM2分割FLUX前景mask vs 白模渲染mask → IoU | > 0.85 |
| **DINOv2 CLS Cosine** | cos_sim(DINOv2_CLS(white), DINOv2_CLS(flux)) | > 0.75 |
| **Edge Correlation** | Canny(white) vs Canny(flux_foreground) → Pearson r | > 0.60 |
| **LPIPS (foreground)** | 前景区域 LPIPS | < 0.30 |

用途：筛选 FLUX 生成失败的图片（形状被改变的），保证训练数据质量。

### 背景多样性/复杂度

| 指标 | 计算方法 | 意义 |
|------|----------|------|
| **背景面积比** | 1 - foreground_pixel_ratio | 大=更有挑战性 |
| **背景纹理 Entropy** | 背景区域 GLCM entropy | 高=纹理复杂 |
| **跨样本多样性** | 背景DINOv2 features的标准差 | 数据集是否足够多变 |
| **光照变化度** | 前景亮度直方图 vs 白模直方图的KL散度 | 光影变化程度 |

### 难度分级

```
Level 0: 白模原图（baseline）
Level 1: 简单背景（纯色/渐变，轻微阴影）
Level 2: 中等背景（桌面/简单场景）
Level 3: 复杂背景（杂乱环境，强光影，部分遮挡）
```

建议：先在 Level 1-2 验证 condition fusion work，再挑战 Level 3。

---

## 时间线建议

| 时间 | 任务 | 产出 |
|------|------|------|
| 本周 | 跑 Exp 1 (Feature Mapper) | 数值 + 15个case可视化 |
| 下周前半 | 分析 Exp 1 结果，决定是否做 Exp 2 | 实验报告 |
| 下周后半 | Exp 2 (蒸馏) 或 Exp 3 (Topo Bias) | 数值 + 对比可视化 |
| 第3周 | 组合最佳方案，跑 ablation | 完整 ablation table |
| 第4周 | Exp 4 (VAE改进) 如果需要 | 或开始写论文 |
