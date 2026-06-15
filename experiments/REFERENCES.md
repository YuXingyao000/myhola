# Reading List / References

> 每天规划实验前先查这里。如果今天的实验方向在这里找不到足够支撑，再用 WebSearch 检索
> 近 3 年的相关论文（或奠基级理论），读完后把新条目回写到这里。维护规则见 [brepnet-experiment-planning](../skills/brepnet-experiment-planning/SKILL.md)。
>
> 规则：
> - 每条尽量写清 **为什么和我们相关**（one line），方便日后规划直接引用。
> - 不要编造 arXiv 号或链接；不确定就写 `（链接待补）`。
> - 优先：近 3 年同方向工作 + 奠基级理论（DDPM / VAE / LDM 等）。

## 奠基理论（Foundational）

| 论文 | 链接 | 为什么和我们相关 |
|------|------|------------------|
| Auto-Encoding Variational Bayes (VAE) | [arXiv:1312.6114](https://arxiv.org/abs/1312.6114) | topology VAE / CVAE 的 posterior+prior+KL 基础 |
| Learning Structured Output Representation using Deep Conditional Generative Models (CVAE) | [NeurIPS 2015](https://papers.nips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models) | `image -> topology` 是条件 structured output；CVAE 正对应 `q(z\|x,y)`、`p(z\|x)`、`p(y\|x,z)` 的训练/推理拆分 |
| A Probabilistic U-Net for Segmentation of Ambiguous Images | [arXiv:1806.05034](https://arxiv.org/abs/1806.05034) | 单图到 topology mask 和 ambiguous segmentation 类似，都是一个条件输入对应多个 plausible structured outputs，应使用 sample@K / diversity 视角 |
| Denoising Diffusion Probabilistic Models (DDPM) | [arXiv:2006.11239](https://arxiv.org/abs/2006.11239) | B-Rep latent diffusion 的核心生成框架 |
| High-Resolution Image Synthesis with Latent Diffusion (LDM) | [arXiv:2112.10752](https://arxiv.org/abs/2112.10752) | 在 AE latent 上做 diffusion 的范式，和我们的 latent diffusion 对齐 |
| Classifier-Free Diffusion Guidance | [arXiv:2207.12598](https://arxiv.org/abs/2207.12598) | image-conditioned diffusion 的条件强度控制 |
| Scalable Diffusion Models with Transformers (DiT) | [arXiv:2212.09748](https://arxiv.org/abs/2212.09748) | transformer denoiser 设计参考 |
| Elucidating the Design Space of Diffusion Models (EDM) | [arXiv:2206.00364](https://arxiv.org/abs/2206.00364) | 噪声调度 / 采样器设计参考 |

## 表示与条件编码（Representation & Conditioning）

| 论文 | 链接 | 为什么和我们相关 |
|------|------|------------------|
| DINOv2: Learning Robust Visual Features without Supervision | [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | 当前 CVAE/diffusion 的 image encoder backbone（real photo 条件）|
| Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet) | [arXiv:2302.05543](https://arxiv.org/abs/2302.05543) | 冻结强 backbone，用 zero-initialized / trainable control branch 接入外部条件；支撑 06-14 的 topology decoder adapter 思路 |
| T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models | [arXiv:2302.08453](https://arxiv.org/abs/2302.08453) | 训练轻量 adapter 对齐外部结构条件，同时保留原模型能力；对应“不要破坏 06-08 topology decoder，但允许 image tokens 影响生成”的方向 |
| Gated Linear Attention Transformers with Hardware-Efficient Training | [arXiv:2312.06635](https://arxiv.org/abs/2312.06635) | 近期 gated attention / gated memory update 思路的代表；与 06-14 的 gate 直觉相关，但当前实验只借用 gate 控制 residual 条件写入，不替换 Transformer attention |

## CAD / B-Rep 生成（Domain，近 3 年优先）

| 论文 | 链接 | 为什么和我们相关 |
|------|------|------------------|
| DeepCAD: A Deep Generative Network for CAD Models | [arXiv:2105.09492](https://arxiv.org/abs/2105.09492) | 当前数据集（deepcad_v7）来源与 CAD 序列生成参考 |
| HoLa-BRep（本项目主线方法） | （链接待补） | 我们的 diffusion 主线方法；topology mask 接入对象 |
| BrepGen / SolidGen 等 B-Rep 生成工作 | （链接待补） | B-Rep 拓扑+几何联合生成的对照方法 |

## 待读 / 本周新增（Queue）

> 每天 WebSearch 找到的新论文先放这里，读完后归类到上面对应分区。

- [ ] （示例）image-to-3D / single-view CAD reconstruction 近 1 年工作 —（链接待补）
