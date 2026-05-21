# 待实施实验笔记

## 实验优先级（更新后）

```
0. Intrinsic Decomposition (Normal Map输入)  ← 最快验证，可能一步到位
1. VAE Adversarial Topology Training         ← 3行代码，几小时
2. Diffusion Topology Attention Bias         ← 需要改dataset+backbone，1-2天
3. Feature Mapper                            ← 如果0不够好再做
4. Knowledge Distillation                    ← 重炮，最后
```

---

## 实验0：Intrinsic Decomposition (Normal Map 作为条件输入)

### 动机

DINOv2 从真实照片提取的特征里，几何信息被纹理/光照/背景淹没。
如果先把照片分解为 Normal Map（纯几何表示），再喂给 DINOv2，特征里就只有几何信息了。

Normal Map 是 domain-invariant 的：白模渲染的 normal map ≈ 真实照片估计的 normal map（同一个物体形状一样）。

### 做法

```
Step 1: 对所有真实照片，用 StableNormal/DSINE/Marigold 提取 Normal Map
        real_photo → Normal估计模型 → normal_map [H, W, 3]

Step 2: 把 normal_map 作为 DINOv2 的输入（替换 real_photo）
        改 dataset.py 的 prepare_condition 里的图片加载

Step 3: 用现有的白模 conditioned diffusion 权重直接推理（不需要重训！）
        因为 normal_map 的 DINOv2 特征应该和白模特征在同一个域里

Step 4: 跑 eval 看 Chamfer 和 F1
```

### 可选的 Normal 估计模型（2024-2025）

| 模型 | 论文 | 特点 |
|------|------|------|
| StableNormal (2025) | https://stable-normal.github.io/ | 基于diffusion，最稳定 |
| Marigold (2024) | https://marigoldmonodepth.github.io/ | Depth+Normal联合 |
| GeoWizard (2024) | https://fuxiao0719.github.io/projects/geowizard/ | 联合估计 |
| DSINE (2024) | https://github.com/baegwangbin/DSINE | 专注normal精度 |
| Lotus (2024) | https://lotus3d.github.io/ | 扩散先验 |

### 关键对比实验

```
IC-1: normal_map → DINOv2 → 原有diffusion推理 (不重训)
IC-2: normal_map + depth_map 双通道
IC-3: IC vs Feature Mapper vs IC+Mapper

对比 baseline:
  - real_photo → DINOv2 (现在, CD≈0.21)
  - white_render → DINOv2 (oracle上限, CD≈0.001)
  - normal_map → DINOv2 (期望接近白模)
```

### 代码改动

主要改 `dataset.py` 的 `prepare_condition` 函数里的图片加载：

```python
# 原来:
if "single_img" in v_condition_names:
    imgs = ori_data["svr_imgs"][v_id_aug][None, :]
    if pick_natural < 0.2:
        natural_data = np.load(v_cond_root / v_folder_path / "single_view.npz")
        imgs = natural_data["flux"][None, :]

# 改为:
if "single_img" in v_condition_names:
    imgs = ori_data["svr_imgs"][v_id_aug][None, :]
    if pick_natural < real_photo_ratio:
        # 加载 normal map 而不是原始照片
        normal_path = v_cond_root / v_folder_path / "normal_map.png"
        normal_img = np.array(Image.open(normal_path))  # [H, W, 3]
        imgs = normal_img[None, :]
```

### 为什么这可能比 Feature Mapper 还好

- Feature Mapper: 在特征空间修补 domain gap（间接，需要训练）
- Normal Map: 在图像空间消除 domain gap（直接，不需要训练）
- Normal Map 输入后，DINOv2 看到的就是"纯几何图像"，和白模本质相同

---

## 实验1：VAE Adversarial Topology Training

### 动机

Decoder robustness 实验显示：σ=0.10 的噪声就让 valid solid 从 87% 暴跌到 44%。
原因是 AttnIntersection + classifier 对 face_z 扰动极度敏感。
通过在训练时给 intersection 输入加噪声，让它学会在扰动下仍正确预测拓扑。

### 做法

```python
# 在 VAE 的 decode 方法中，intersection prediction 之前加噪声

def decode(self, encoding_result, v_data=None):
    face_z = encoding_result["face_z"]
    
    # ... face_attn2 正常流程 ...
    
    # ====== 新增: adversarial noise for topology ======
    if v_data is not None and self.training:  # 只在训练时加
        noise_std = 0.05  # 可以试 0.05, 0.08, 0.10
        face_z_for_topo = face_z + noise_std * torch.randn_like(face_z)
    else:
        face_z_for_topo = face_z
    # ====== 结束 ======
    
    # 原来的 intersection prediction，改用 face_z_for_topo
    if v_data is None:  # 推理
        indexes = ...
        feature_pair = face_z[indexes]  # 推理时不加噪声
    else:  # 训练
        true_intersection_embedding = face_z_for_topo[edge_face_connectivity[:, 1:]]
        false_intersection_embedding = face_z_for_topo[v_zero_positions]
        feature_pair = torch.cat((true_intersection_embedding, false_intersection_embedding), dim=0)
    
    feature_pair = self.inter(feature_pair)
    pred = self.classifier(feature_pair)
    # ... 后面不变 ...
```

### Fine-tune 设置

```python
# 加载预训练 VAE
vae = load_vae(checkpoint_path)

# 冻住所有参数
for param in vae.parameters():
    param.requires_grad = False

# 只解冻 intersection 相关的
for name, param in vae.named_parameters():
    if "inter" in name or "classifier" in name:
        param.requires_grad = True

# 用正常 lr 训练（因为只训一小部分）
optimizer = AdamW(filter(lambda p: p.requires_grad, vae.parameters()), lr=1e-4)
```

### 验证

训完后重新跑 noise sensitivity 实验：
- 对比 σ=0.10 时的 valid rate: 原来 44% → 目标 >70%
- 对比 σ=0.15 时的 valid rate: 原来 16% → 目标 >40%

---

## 实验2：Diffusion Topology Attention Bias

### 动机

Diffusion 的 24 层 TransformerEncoder 对所有 30 个 face token 一视同仁（全 self-attention）。
不知道哪些面应该相邻 → 生成的 latent 之间可能不满足拓扑一致性。
给 self-attention 加 topology bias → 相邻面互相关注更多 → 生成更结构化。

### Step 1: 修改 Dataset，返回 GT adjacency

```python
# Diffusion_dataset.__getitem__ 中，额外加载 face_adj

# 原来只加载 features.npy
data_npz = np.load(self.latent_root / (folder_path + f"_{id_aug}") / "features.npy")
face_features = torch.from_numpy(data_npz)

# 新增: 加载 GT topology
brep_data_path = Path(self.conf["data_root"]) / folder_path / "data.npz"
if brep_data_path.exists():
    brep_data = np.load(brep_data_path)
    face_adj = torch.from_numpy(brep_data["face_adj"]).float()
    num_faces = min(face_adj.shape[0], self.max_faces)
    padded_adj = torch.zeros(self.max_faces, self.max_faces)
    padded_adj[:num_faces, :num_faces] = face_adj[:num_faces, :num_faces]
else:
    padded_adj = torch.zeros(self.max_faces, self.max_faces)

# return 时加上 padded_adj
return (folder_path, padded_face_features, condition, id_aug, padded_adj)

# collate_fn 里 stack adjacency:
# face_adj_batch = torch.stack([item[4] for item in batch], dim=0)  # [B, 30, 30]
```

### Step 2: 修改 Diffusion backbone，接受 topology bias

```python
class DiffusionCrossAttn(nn.Module):
    def __init__(self, cfg):
        ...
        # 新增: 可学习的 topology bias 强度
        self.topo_scale = nn.Parameter(torch.tensor(1.0))
    
    def apply_denoising_backbone(self, noise_features, img_memory, topo_bias=None):
        """
        topo_bias: [B, 30, 30] adjacency matrix (0~1)
        """
        if topo_bias is None or not hasattr(self, 'topo_scale'):
            return self.net1(noise_features)
        
        # 转成 additive attention bias
        # 相邻 → 正值(鼓励attention), 不相邻 → 0
        attn_mask = topo_bias * self.topo_scale  # [B, 30, 30]
        
        # PyTorch TransformerEncoder 的 mask 需要是 [seq, seq] 或 [B*nhead, seq, seq]
        # 简化版: 取 batch mean (或逐样本处理)
        # 注意: nn.TransformerEncoderLayer 的 src_mask 是 additive (加到attn logits上)
        # 正值 = 鼓励attention, 负值/极大负值 = 抑制
        
        output = noise_features
        for layer in self.net1.layers:
            output = layer(output, src_mask=attn_mask)
        if self.net1.norm is not None:
            output = self.net1.norm(output)
        return output
```

### Step 3: Forward 中加入 timestep-dependent 衰减

```python
def diffuse(self, v_feature, v_timesteps, v_condition, v_align_feature, topo_bias=None):
    ...
    # Topology bias 在高噪声时强，低噪声时弱
    # 直觉: 高噪声阶段决定结构，低噪声阶段精修几何
    if topo_bias is not None:
        t_normalized = v_timesteps.float() / 1000.0  # [B], 范围0~1
        # 高t → 权重大, 低t → 权重小
        topo_weight = t_normalized  # 线性衰减, 或用 cosine
        weighted_topo = topo_bias * topo_weight[:, None, None]  # [B, 30, 30]
    else:
        weighted_topo = None
    
    ...
    pred_x0 = self.apply_denoising_backbone(noise_features, img_memory, weighted_topo)
    ...
```

### Step 4: 推理时的 topology 来源

```
实验2a (oracle验证): 直接用 GT topology → 验证方向是否有效
  → 如果 F1 大幅提升 → 方向对，继续2b/2c

实验2b (self-derived): 先跑500步无bias → 用中间face_z的cosine similarity估计topology → 后500步用它
  → 不需要额外模型

实验2c (predictor): 训一个小网络 image_features → adj_matrix
  → 需要额外训练，但更准确
```

### 实验2a 的推理代码（oracle上限测试）

```python
@torch.no_grad()
def inference_with_gt_topology(model, batch, device):
    """用GT topology做推理，测试方向有效性上限"""
    gt_adj = batch["face_adj"].to(device)  # [B, 30, 30]
    face_features = torch.randn((B, 30, 32), device=device)
    condition = extract_condition(batch)
    
    for t in tqdm(scheduler.timesteps):
        # Timestep-dependent weight
        t_weight = t.float() / 1000.0
        topo_bias = gt_adj * t_weight
        
        timesteps = t.reshape(-1).to(device)
        pred, _ = model.diffuse(face_features, timesteps, condition, 
                                v_align_feature=face_features,
                                topo_bias=topo_bias)
        face_features = scheduler.step(pred, t, face_features).prev_sample
    
    return face_features
```

### 注意: src_mask 的形状问题

PyTorch 的 `nn.TransformerEncoderLayer` 的 `src_mask` 参数:
- 形状: `[seq_len, seq_len]` 或 `[B * num_heads, seq_len, seq_len]`
- 类型: additive float mask (加到 attention logits 上)
- 如果 batch 内每个样本的 adj 不同，需要展开为 `[B * num_heads, 30, 30]`

```python
# 处理 batch-wise mask
B = noise_features.shape[0]
num_heads = self.dim_total // 64  # e.g., 12
# [B, 30, 30] → [B*num_heads, 30, 30]
attn_mask_expanded = topo_bias.unsqueeze(1).repeat(1, num_heads, 1, 1)
attn_mask_expanded = attn_mask_expanded.reshape(B * num_heads, 30, 30)
```

---

## 执行顺序

```
今天回去:
  1. 下载 StableNormal / DSINE
  2. 对测试集100张真实照片提取 normal map
  3. 用 normal map 作为 condition 输入，直接推理（不重训）→ 看 CD/F1
  4. 同时: 改3行代码做 VAE adversarial topo training, 开始fine-tune

这周:
  5. 如果 normal map 有效 → 对整个训练集提取 normal map → 重训 diffusion
  6. 如果 normal map 不够 → 做 Feature Mapper
  7. 同时: 改 dataset + backbone 做 topology bias 实验
```
