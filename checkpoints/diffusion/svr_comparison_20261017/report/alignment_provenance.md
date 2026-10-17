# 历史 alignment 候选的身份与边界

本次找到并严格恢复的是 `0413_deepcad_natural_blender_align` 的历史模型。
它是使用 DINOv2 的前端 cross-attention 加 CLIP 式对称 InfoNCE 对齐，**不是 OpenAI CLIP 图像编码器**。
权重中的实验名、callback 保存路径与本地 W&B run `ya2o4xi1` 相互对应；判断并非仅来自文件名。

## 权重选择

| 角色 | 实际 metadata step / epoch | 位置 |
|---|---:|---|
| 已排队的历史候选 | 150,000 / 513 | `/mnt/d/data/new_cond_ckpt/0423_deepcad_natural_blender_align.ckpt` |
| 此 run 的 callback-best | 250,000 / 856 | `outputs_brepnet_diffusion/Diffusion_0413_deepcad_natural_blender_align/img2brep/ya2o4xi1/checkpoints/epoch=856-step=250000.ckpt` |
| 此 run 最后保存 | 352,152 / 1205 | 同目录 `last.ckpt` |

原始 `last.ckpt` 的 callback 明确将 250,000 步设为最佳，记录 `Validation_Loss=0.0285445675`。
该选择发生于历史训练，独立于此次固定 100 对象的评估结果。150k 文件只是另存的中间权重，不能称为此 run 的最终最佳。
这里的“最佳”仅指这个 run 的历史 callback，**尚未确认它就是用户记忆中的全历史最佳 CLIP 权重**。

## 数据协议限制

保存配置的 train、validation、test 路径都指向同一份 `src/brepnet/data/DataGeneration/test_list/train.txt`。
当前可恢复的文件包含 750 个不同 CAD；因此历史验证是在训练同集上计算，分数不能与当前独立 validation loss 直接比较。
当前主线使用更大的训练集；这个历史模型也不能作为相同数据和训练预算的架构消融。

当前这 750 个 ID 与本次冻结的 100 个 validation ID、当前完整 3,505 个 validation ID，精确字符串交集均为零。
旧列表未被历史 Git 版本化，因此无法证明当前文件的每个字节就是训练时内容；精确 ID 无交集也不证明几何去重组绝无重叠。

本地 run 记录源码 commit `be94bbea8a6eadc49ef7539e2ea0c13525b71a72`。
该源码的 `single_img` 明确读取 `imgs.npz["svr_imgs"]`；保存配置指向 `deepcad_v6_cond`，并开启旧视角增强。
旧实现使用 `np.random.randint(0, 63)`，即索引 0–62，而当前主线训练固定 rotation 0。
此次比较统一输入固定 validation 清单的灰模 view 0，不使用照片或 GT topology。

多个不同外观的历史 run 记录了同一 Git commit，且没有完整保存当时未提交的源码差异。
所以源码和实验关联支持此候选的灰模来源，但不能完全排除训练时工作树差异；不能用 `single_img` 或文件名独立证明数据身份。

## 运行恢复与核验

`alignment_runtime.py` 从固定 Git commit 提取网络、dataset 辅助函数和原生 VAE 源码，隔离旧包名后构造模型。
150k 和 250k 都已在 CPU 上完成 `strict=True, assign=True` 完整加载：1,556 个 state key，缺失和额外 key 均为零。
没有删除 alignment 参数、没有宽松加载，也没有改动原生采样、VAE 解码或去重行为。
模型参数、buffer 和重建后的 scheduler 均已落到 CPU；这部分验证没有运行 GPU 推理。

完整 checkpoint 内的 DINO 和 AE 权重直接恢复，不下载权重，不强换其他 AE。
另对 150k 中的 704 个 AE tensor 与当前 `1119_deepcad_aug1_11k.ckpt` 做了逐 tensor 比较，key、shape、值全部相同。
旧模型保留自己的原生 decoder 行为；权重相同不意味着可以忽略实现版本。

归档网络包含 4 层 1024 维前端 TransformerDecoder、24 层 768 维去噪主干，以及 8 个 alignment tensor。
对齐目标是 clean CAD latent 与图像 token 的全局嵌入，采用温度 0.07、权重 1 的双向 InfoNCE。
原生推理没有读取 GT latent、face count、ordering 或 topology；对齐支路在推理中使用当前 noisy latent，返回的对齐 loss 不参与生成更新。

详细来源、源码哈希、严格加载结果与 split 哈希见 [alignment_checkpoint_audit.json](audits/alignment_checkpoint_audit.json)。
推理与几何评估结果另行记录；本说明不据此宣称重建质量改善。

## 其他候选核验

另外检查了被忽略的旧 run 目录、`/mnt/d/ckpt/HoLa`、`/mnt/d/data/new_cond_ckpt` 等权重位置，
并沿 `img2brep/<wandbid>/checkpoints` 的固定层级找到 113 个旧 checkpoint 路径，没有遍历几何输出内容。
0408 的三个灰模候选实际为 42、2,920、5,256 步，仍使用 750 对象训练列表；
0411 的四个候选实际为 292、2,044、22,776、8,760 步，使用相同的小集。
已检查路径中，尚未发现来源明确且训练更充分的大训练集灰模 alignment 最佳权重。

其他充分训练的候选包含 FLUX 条件、额外 condition-margin 目标或 GT adjacency；不能把它们直接并入无 GT 的灰模 SVR 行。
原始 HoLa SVR 1500k 权重则没有这些 alignment 参数，属于另一种历史架构。
检索范围和逐文件实际 metadata 见 [alignment_search.json](audits/alignment_search.json)；这一检索结论不代表用户全部历史资产都已穷尽。
