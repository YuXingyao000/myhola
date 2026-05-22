# 数据集构造：Real-Photo CAD Dataset

## 1. Pipeline 总览

```
输入:
  - ABC + DeepCAD 数据集中的 CAD 模型 (STEP files)
  - 每个模型已有 24 视角的白模渲染 (Blender, 无材质, 白色背景)

输出:
  - 每个 CAD 模型对应 1~N 张"真实风格照片"
  - 配对关系: (real_photo, white_render, CAD_model) 三元组
  - 质量标注: 每张图的 quality metrics

Pipeline:
  Stage 1: 白模渲染 (已完成)
  Stage 2: 真实照片生成 (FLUX.1-Kontext)
  Stage 3: 自动质量过滤
  Stage 4: 数据集统计与发布
```

---

## 2. Stage 2: 真实照片生成 (详细)

### 2.1 为什么选 FLUX.1-Kontext

| 模型 | 方式 | 形状保留能力 | 成本 |
|------|------|------------|------|
| FLUX.1-Kontext | Reference-based (给参考图 + 文字) | ⭐⭐⭐ 强 (参考图约束形状) | 本地推理, 免费 |
| Midjourney | 纯文字描述 | ⭐ 弱 (文字很难描述精确3D形状) | API收费 |
| DALL-E 3 | 纯文字描述 | ⭐ 弱 | API收费 |
| ControlNet + SD | Depth/Edge引导 | ⭐⭐ 中等 | 本地推理, 免费 |

FLUX.1-Kontext 的核心优势: **它接受一张参考图作为输入，生成的图片会保留参考图中物体的形状**。这正好是我们需要的——白模作为参考，生成保留相同形状但有真实材质/背景/光照的照片。

### 2.2 生成参数

```python
# 当前使用的模型
model: FLUX.1-Kontext-dev (Q8_0 GGUF 量化)
# 注意: 4bit/8bit 量化会损失一定的图像质量和指令遵循能力

# 推理参数
inference_params = {
    "num_inference_steps": 28,     # 采样步数 (Kontext 默认推荐)
    "guidance_scale": 3.5,         # CFG 强度 (越高越遵循prompt, 但可能过饱和)
    "width": 1024,                 # 输出分辨率
    "height": 1024,
    "seed": random,                # 每次不同, 增加多样性
}

# 输入
input = {
    "reference_image": white_render,  # 白模渲染图 (224×224 或 512×512)
    "prompt": text_prompt,            # 描述目标风格的文字
}
```

### 2.3 Prompt 设计策略

**当前问题**: 提示词全靠手动调整, 不系统, 无法量化好坏。

**改进: 结构化 Prompt 模板系统**

```python
# Prompt 由多个可替换的 slot 组成:
template = "{object_description}, {material}, {placement}, {background}, {lighting}, {photography_style}"

# 每个 slot 有一个候选列表:
SLOTS = {
    "object_description": [
        "a mechanical part",
        "an industrial component", 
        "a precision-machined object",
        "a 3D printed prototype",
    ],
    "material": [
        "made of brushed aluminum",
        "made of matte black plastic",
        "made of polished steel",
        "made of white ceramic",
        "made of dark anodized metal",
        "",  # 空 = 不指定材质, 让模型自由发挥
    ],
    "placement": [
        "placed on a wooden desk",
        "sitting on a white table",
        "held in a hand",
        "placed on a workbench",
        "on a dark surface",
        "on a concrete floor",
    ],
    "background": [
        "with a blurred office background",
        "with a clean white background",
        "in a workshop environment",
        "with natural outdoor lighting",
        "in a studio with soft lighting",
        "",  # 空 = 不指定
    ],
    "lighting": [
        "natural daylight",
        "studio lighting with soft shadows",
        "warm indoor lighting",
        "harsh directional light",
        "",
    ],
    "photography_style": [
        "product photography",
        "iPhone photo",
        "DSLR photo, shallow depth of field",
        "casual snapshot",
        "professional studio shot",
    ],
}

# 生成一条 prompt:
def generate_prompt(difficulty_level="medium"):
    """
    difficulty_level 控制背景/光照的复杂度:
      easy: 简单背景, 中性光, 类似白模但有材质
      medium: 普通场景, 正常光照
      hard: 复杂背景, 强光影, 遮挡
    """
    if difficulty_level == "easy":
        prompt = random.choice(SLOTS["object_description"])
        prompt += ", " + random.choice(SLOTS["material"])
        prompt += ", on a clean surface, studio lighting, product photography"
    elif difficulty_level == "medium":
        prompt = random.choice(SLOTS["object_description"])
        prompt += ", " + random.choice(SLOTS["material"])
        prompt += ", " + random.choice(SLOTS["placement"])
        prompt += ", " + random.choice(SLOTS["lighting"])
        prompt += ", " + random.choice(SLOTS["photography_style"])
    elif difficulty_level == "hard":
        prompt = random.choice(SLOTS["object_description"])
        prompt += ", " + random.choice(SLOTS["material"])
        prompt += ", " + random.choice(SLOTS["placement"])
        prompt += ", " + random.choice(SLOTS["background"])
        prompt += ", " + random.choice(SLOTS["lighting"])
        prompt += ", casual photo, slight motion blur"
    return prompt
```

### 2.4 量化模型的限制与应对

**问题**: Q8/Q4 量化会导致:
- 指令遵循能力下降 (prompt 复杂时容易忽略细节)
- 图像细节模糊 (特别是小物体的边缘)
- 形状变形概率增大

**应对策略**:
1. **Prompt 简短化**: 量化模型对长/复杂 prompt 的理解差, 所以 prompt 尽量短而精
2. **多生成 + 过滤**: 每个模型生成 3-5 张, 自动过滤保留最好的 1-2 张
3. **分辨率调整**: 在能力范围内尽量用高分辨率 (但量化后显存有限)
4. **质量自动检测**: 通过指标自动判断哪些需要重新生成

---

## 3. Stage 3: 自动质量过滤

### 3.1 过滤指标定义

```python
class DatasetQualityMetrics:
    """评估单张生成图片的质量"""
    
    def shape_preservation(self, gen_image, white_render):
        """
        形状保真度: 生成图片中的物体是否保留了原始形状
        
        方法: 分割前景 → 计算轮廓 IoU
        工具: SAM2 / GroundingDINO + SAM
        
        Returns: float in [0, 1], 越高越好
        """
        mask_gen = segment_foreground(gen_image)     # SAM2 分割
        mask_white = binarize(white_render)          # 白模直接阈值化
        mask_white_resized = resize(mask_white, mask_gen.shape)
        
        intersection = (mask_gen & mask_white_resized).sum()
        union = (mask_gen | mask_white_resized).sum()
        return intersection / (union + 1e-6)
    
    def semantic_similarity(self, gen_image, white_render):
        """
        语义一致性: DINOv2 CLS token 的余弦相似度
        
        衡量: 高层语义上两张图是否描述同一个物体
        
        Returns: float in [-1, 1], 通常在 [0.3, 0.9]
        """
        feat_gen = dinov2.get_cls_token(gen_image)       # [1024]
        feat_white = dinov2.get_cls_token(white_render)  # [1024]
        return cosine_similarity(feat_gen, feat_white)
    
    def realism_score(self, gen_image):
        """
        真实感: 生成的图片看起来多像"真实照片"
        
        方法: CLIP score (图片 vs "a real photograph")
        
        Returns: float, 越高越真实
        """
        return clip_score(gen_image, "a real photograph of a product")
    
    def background_complexity(self, gen_image):
        """
        背景复杂度: 衡量背景的信息量
        
        方法: 分割出背景 → 计算纹理entropy
        
        Returns: float, 0=纯色背景, 高=复杂纹理
        """
        mask_fg = segment_foreground(gen_image)
        background = gen_image * (1 - mask_fg)
        return compute_entropy(background)
    
    def edge_preservation(self, gen_image, white_render):
        """
        边缘保留: 物体轮廓的边缘结构是否一致
        
        方法: Canny 边缘检测 → 比较
        
        Returns: float in [0, 1]
        """
        edges_gen = canny(segment_foreground(gen_image) * gen_image)
        edges_white = canny(white_render)
        # F1 score between edge maps
        return edge_f1(edges_gen, edges_white)
```

### 3.2 过滤规则

```python
def should_keep(metrics):
    """
    决定一张生成图是否保留
    
    核心原则: 形状保真度是硬约束, 其他是软约束
    """
    # 硬约束: 形状必须保留 (否则下游B-Rep生成无意义)
    if metrics["shape_preservation"] < 0.70:
        return False, "shape_deformed"
    
    # 硬约束: 语义一致性 (确保DINOv2认为是同一物体)
    if metrics["semantic_similarity"] < 0.50:
        return False, "semantically_different"
    
    # 软约束: 真实感 (太假的也不要, 但不用特别高)
    if metrics["realism_score"] < 0.20:
        return False, "unrealistic"
    
    return True, "pass"

# 阈值确定方法:
# 1. 人工标注 100-200 张图片 (good / bad / borderline)
# 2. 计算各指标在 good/bad 上的分布
# 3. 选择 threshold 使得:
#    - good 中 >95% 通过
#    - bad 中 >90% 被过滤
# 4. 报告: precision / recall of the filter
```

### 3.3 过滤后的处理

```python
for model_id in all_models:
    generated_images = generate_multiple(model_id, n=5)  # 每个模型生成5张
    
    passed = []
    for img in generated_images:
        metrics = compute_all_metrics(img, white_render)
        keep, reason = should_keep(metrics)
        if keep:
            passed.append((img, metrics))
    
    if len(passed) == 0:
        # 全部失败 → 换 prompt 策略重新生成
        log_failure(model_id, reason_distribution)
        # 可能原因: 物体太复杂, 量化模型无法保持形状
        # 应对: 降低背景复杂度, 用更简单的prompt
    else:
        # 选最好的1-2张保留
        passed.sort(key=lambda x: x[1]["shape_preservation"], reverse=True)
        save(passed[:2])
```

---

## 4. 实验目标与改进步骤

### 4.1 当前瓶颈诊断

**你不知道的是**: 当前数据质量差, 到底是哪个环节的问题?

```
可能原因 A: 量化模型能力不足 (Q8 vs 原始 FP16)
可能原因 B: Prompt 不够好 (描述不精确)
可能原因 C: Reference 方式有限 (FLUX Kontext 的参考图约束不够强)
可能原因 D: 某些 CAD 模型形状太复杂, 任何模型都保不住
```

### 4.2 诊断实验

**实验 D1: 量化 vs 原始 (小规模)**

```
选 50 个模型
  - 用你的 Q8 量化模型生成
  - 用 FLUX.1-Kontext 官方 API (如果有试用额度) 生成
  - 对比 shape_preservation 和 semantic_similarity 的分布

结果:
  如果 API 明显更好 → 量化是瓶颈, 考虑省钱方案 (Q6? 部分模型用API?)
  如果 API 也差不多 → 量化不是主因, 问题在 prompt 或模型设计
```

**实验 D2: Prompt 消融**

```
选 50 个模型, 固定同一个 seed
  - Prompt A: "a photo of this object" (最简单)
  - Prompt B: "product photography of this mechanical part, studio lighting"
  - Prompt C: 完整结构化 prompt (场景+材质+光照+风格)
  - Prompt D: "Keep the exact same shape. Place it on a desk. Natural lighting."
             (强调形状保留)

对比各 prompt 的 shape_preservation 分布
→ 找出"形状保留最好"的 prompt 风格
```

**实验 D3: 物体复杂度 vs 生成质量**

```
按 CAD 模型面数分组:
  简单 (5-10面) | 中等 (10-20面) | 复杂 (20-30面)
  
对比各组的 shape_preservation
→ 如果复杂模型过滤率极高 → 可能需要对复杂模型用更保守的prompt
```

### 4.3 改进思路

**不止是改 prompt。还可以改生成策略:**

```
方向 1: Prompt 工程 (免费)
  - 系统化 prompt 模板
  - 强调形状保留的关键词 ("exact same shape", "same geometry")
  - 控制背景复杂度 (简单→复杂递进)

方向 2: 多次生成 + 过滤 (免费, 但慢)
  - 每个模型生成 5-10 张
  - 自动过滤保留最好的
  - 对失败模型降级 prompt (复杂→简单)

方向 3: 图像后处理 (免费)
  - 对生成图用 SAM 分割前景
  - 检查前景轮廓是否和白模一致
  - 不一致的区域做 inpainting 修复 (用 FLUX inpaint)

方向 4: 换更好的生成策略 (如果上面都不够)
  - ControlNet + SDXL (用 depth/canny 控制形状, 文字控制风格)
    → 形状保留更可控, 但真实感可能不如 FLUX
  - IP-Adapter (用白模作为 image prompt)
    → 另一种 reference-based 方式
  - 如果有经费: 少量用商业API (Midjourney v7 / DALL-E 4) 做对比实验
    → 论文里可以报告不同模型的质量对比

方向 5: 量化改善 (如果诊断为量化瓶颈)
  - Q8 → Q6 (显存多一点, 质量好一点)
  - 关键步骤不量化: 前几步 denoise 用 FP16, 后面用 Q8
  - 或: 攒钱买/租更大显存的卡跑 FP16
```

### 4.4 数据集规模规划

```
目标规模:
  ABC: ~10,000 模型 × 2张 = 20,000 张
  DeepCAD: ~10,000 模型 × 2张 = 20,000 张
  总计: ~40,000 配对图片

生成成本估算 (本地 Q8, 单卡 A100/4090):
  每张图: ~10秒 (28步采样)
  5张候选/模型 × 20000模型 = 100,000 张待生成
  总时间: ~280 GPU-hours ≈ 12天 (单卡)
  
过滤后保留: 假设 60% 通过率 → 60,000 张
每个模型选2张最好的 → 最终 40,000 张
```

---

## 5. 论文中数据集部分的 Ablation 设计

```
Table X: Dataset Quality Analysis

(a) 各质量指标统计
| Metric | Mean | Std | Min | Max |
| Shape IoU | 0.82 | 0.08 | 0.70 | 0.98 |
| DINO Cosine | 0.71 | 0.10 | 0.50 | 0.92 |
| Realism CLIP | 0.28 | 0.04 | 0.20 | 0.35 |
| Bg Complexity | 3.2 | 1.5 | 0.5 | 6.8 |

(b) 过滤阈值对下游性能的影响
| Filter Threshold | Dataset Size | B-Rep CD | B-Rep F1 | Valid Rate |
| 无过滤 | 40,000 | 0.15 | 0.25 | 0.55 |
| IoU>0.7 | 32,000 | 0.08 | 0.35 | 0.65 |
| IoU>0.8 | 24,000 | 0.03 | 0.50 | 0.72 |
| IoU>0.9 | 12,000 | 0.02 | 0.55 | 0.75 |
→ 证明: 数据质量比数量重要 (或两者的trade-off)

(c) 不同生成模型对比 (如果条件允许)
| Generation Model | Shape IoU ↑ | Realism ↑ | Cost |
| FLUX.1-Kontext Q8 | 0.82 | 0.28 | 免费 |
| FLUX.1-Kontext FP16 | 0.87 | 0.30 | 免费(需大显存) |
| ControlNet+SDXL | 0.90 | 0.24 | 免费 |
| Commercial API | 0.85 | 0.32 | $$$ |

(d) Prompt 策略对比
| Prompt Strategy | Shape IoU ↑ | Diversity ↑ |
| 固定简单prompt | 0.88 | 低 |
| 结构化模板(ours) | 0.82 | 高 |
| 随机自由prompt | 0.72 | 最高 |
→ 结构化模板在形状保留和多样性之间取得平衡
```

---

## 6. 和你论文其他部分的连接

```
数据集 ←→ Condition Fusion:
  数据集质量直接影响 condition fusion 的效果
  过滤后的高质量数据 → Feature Mapper 学得更好
  → 论文里可以做: "filtered dataset + mapper" vs "unfiltered + mapper"

数据集 ←→ VAE:
  VAE 不直接用真实照片 (它用 B-Rep 几何训练)
  但数据集质量影响 diffusion 训练 → 间接影响生成结果

数据集 ←→ Evaluation:
  测试时用数据集中的真实照片作为输入
  → 数据集同时是训练集和测试集 (需要 train/val/test split)
```

---

## 7. 执行优先级

```
本周 (和其他实验并行):
  1. 实现自动质量评估脚本 (shape_IoU + DINO_cosine)
  2. 对现有 FLUX 生成结果跑一遍 → 看当前通过率是多少
  3. 分析: 哪类模型失败率高? 哪类 prompt 效果好?

下周:
  4. 实验 D2 (prompt 消融, 50个模型 × 4种prompt)
  5. 确定最终 prompt 策略

之后 (大批量生成):
  6. 用确定好的策略批量生成 (2周 GPU time)
  7. 自动过滤 + 统计
  8. 写论文数据集章节
```
