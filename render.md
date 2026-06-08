好，画像清晰了。我给你的判断和路线如下。

---

# 一、先校准两个认知

### 1. 你最大的问题不是"知识不够"，是"产出物不够"

你已经看了 BRDF/AO/IBL 理论，看了 Ben Cloward。下一道槛**不是再看更多视频**，而是把你看过的东西**重新写一遍出来**。3 个月时间窗口里，"看更多内容"的 ROI 在快速递减；"做出能给面试官看的东西"的 ROI 在快速递增。

具体目标：**3 个月后你应该有 4-6 个 GitHub 上的 UE 渲染小项目**，每个项目带：
- 一个 30s 的 GIF 或视频
- 一段说明文档（解释你做了什么、为什么这么做、性能数据、踩过的坑）

这个产出物**远比"我学过 RTR4 第 9 章"有说服力**。面试官看 GIF + README 30 秒就知道你水深。

### 2. "客户端+懂渲染"和"渲染向客户端/引擎"的分水岭

我先按"渲染向客户端/引擎"的标准给你规划，因为：
- 你周末整段 + 工作日零碎，时间总量够（每周 ~15-20h 渲染实践）
- 路线**做满了能投渲染岗，做一半能投客户端+懂渲染岗**，向下兼容
- 反过来不行：按客户端+懂渲染规划做满了，去投渲染岗时手里没硬货

---

# 二、知识补丁清单（先看你缺什么）

你已经懂的我不重复。你**应该缺**这些，按重要性排序：

| 主题                                               | 你的现状       | 必须补的程度                   |
| -------------------------------------------------- | -------------- | ------------------------------ |
| **GPU 管线 / Vertex-Pixel 数据流**                 | 听过           | 必补，这是面试基础题           |
| **空间变换 / MVP 矩阵 / 齐次裁剪**                 | 大概知道       | 必补，每个 vertex shader 都用  |
| **延迟渲染 vs 前向渲染 / G-Buffer**                | 听过           | 必补，UE 默认延迟，面试必问    |
| **PBR 落地（不只是 BRDF 公式）**                   | 公式懂，没写过 | **重点补**，做 demo 的核心     |
| **后处理管线（Bloom/Tonemapping/...）**            | 没碰过         | 必补，UE 里写得最多的就这个    |
| **阴影（Shadow Map / PCF / CSM）**                 | 没碰过         | 重点补，性能瓶颈常见来源       |
| **GPU 性能分析（RenderDoc）**                      | 没碰过         | **极其重要**，这是渲染岗加分项 |
| **UE 的渲染分层（Material → Shader → RDG → RHI）** | 不清楚         | 必补，面 UE 渲染岗必问         |

---

# 三、3 个月路线（每周一段）

我按"周末整段产出 demo + 工作日零碎读资料"切。每周 demo 都建议丢 GitHub，渐进堆出作品集。

### 阶段 A：基础地基（第 1-3 周）

> **目标：把 GPU 管线、空间变换、UE 材质编辑器打通。能用材质编辑器做出"非业务美术"的效果。**

**第 1 周**
- 工作日读：闫令琪 GAMES101 的第 7-10 课（光栅化、变换、着色），**只看不做**，每节 1 小时。
- 周末做：UE 里建空场景，用材质编辑器手动复现：
  - **菲涅尔效果**（边缘高亮的护盾感）
  - **法线贴图 + 简单光照**（用 `LightVector` 拿光方向，自己做 N·L 不要用默认）
  - **UV 流动效果**（水流、能量线条）
- 关键：**全用材质节点，但每个节点你要能说出它干什么**。不要"找教程抄节点连法"，要"我想算 N·L，所以我需要 NormalVector 节点和 ReflectionVector 节点"。

**第 2 周**
- 工作日读：[Real-Time Rendering 4](https://www.realtimerendering.com/) 第 5 章（Shading Basics）—— 不要从头读，只读这一章。读完你应该能向人解释什么是 BRDF。
- 周末做：在 UE 里做一个 **"完全自己写的" Lambert + Blinn-Phong 球体**：
  - 找一个简单 mesh
  - 不用 UE 的 PBR 默认输出，把 Material 的 Shading Model 设为 `Unlit`
  - 在 Emissive 里完全手写公式：拿光方向、法线、视方向、自己算 N·L 和 (H·N)^p
  - 做对比图：UE 默认 PBR 球 vs 你的 Blinn-Phong 球
- **这一步是分水岭**。能做出来你就有了 shader 直觉。

**第 3 周**
- 工作日读：UE 官方 [Shading Models 文档]，但**只翻一遍记结构**，不背细节。重点理解 UE 的 Shading Model 是个"分支选择器"。
- 周末做：写一个 **Custom 节点贴 HLSL** 的小例子。
  - 用 UE 材质里的 `Custom` 节点贴 10 行 HLSL 代码
  - 做点稍微复杂的：比如自己实现 PBR 里的 GGX D 项
  - 把这个 D 项的可视化输出到屏幕（用 SceneColor debug）
- **目的**：从"连节点"过渡到"写 HLSL"。Custom 节点是个桥。

第 1-3 周末你应该能自信说："我在 UE 材质编辑器里能复现常见效果，会用 Custom 节点写 HLSL，理解 PBR 的 D/F/G 项是什么。"

### 阶段 B：手撕一遍渲染原理（第 4-6 周）

> **目标：从零写一个迷你渲染器，让"理论"变"肌肉记忆"。这一段是 ShaderToy / 小框架时间。**

⚠️ 这一阶段**有意离开 UE**，因为 UE 把太多东西包好了，你看不到全貌。

**第 4 周：LearnOpenGL 速通**
- 跟 [LearnOpenGL.com](https://learnopengl.com/) 走到 **"Lighting" 章节末尾**（包括基础光照、材质、贴图、多光源）。
- 不要全做，**关键章节自己敲一遍**：Coordinate Systems、Basic Lighting、Materials、Lighting maps。
- 这一周你会写出一个能旋转的 phong 立方体。**这就是你的 GPU 管线肌肉记忆**。

**第 5 周：LearnOpenGL PBR + IBL**
- 直接跳到 [PBR](https://learnopengl.com/PBR/Theory) 那几章。
- **必做**：PBR Theory → Lighting → IBL Diffuse → IBL Specular。
- 周末末出一个 PBR 球阵（不同 metallic/roughness），加 IBL 环境光。这就是 Substance Painter / Marmoset 那种球阵预览。
- **这一步你之前看过的 BRDF/IBL 理论会突然全部连起来**。

**第 6 周：ShaderToy 周**
- 从 Inigo Quilez 的 [Articles](https://iquilezles.org/articles/) 选 1-2 篇看（强推 "raymarching distance fields" 和 "smooth minimum"）。
- 在 ShaderToy 上写：
  - 一个程序化噪声（Perlin / Worley 任选）
  - 一个 raymarching SDF 球，加简单光照
  - 一个体积光（如果有时间）
- 这周的目的是**让你习惯纯 fragment shader 数学思维**，将来读 UE 的 PostProcess shader 不会怕。

### 阶段 C：UE 渲染深入（第 7-10 周）

> **目标：从"会用材质编辑器"升级到"懂 UE 渲染管线，能改、能扩展"。**

**第 7 周：RenderDoc + UE 渲染抓帧**
- **必看**：[RenderDoc 官方教程](https://renderdoc.org/docs/getting_started/quick_start.html) 跟一遍，1 小时搞定。
- 在 UE 里 capture 一帧，**逐 pass 看**：
  - Base Pass 在画什么？
  - Lighting Pass 怎么读 G-Buffer？
  - PostProcess 有哪些 pass？
  - Bloom 是怎么做的（多次降采样）？
- 写一篇笔记：**"我抓了 UE 一帧，发现它做了 X 个 pass，每个 pass 干什么"**。这篇笔记如果你写得清楚，**直接是简历项目**。
- 推荐看：B 站 / YouTube 搜 "UE5 抓帧 RenderDoc"，有几个国人做的视频很好。

**第 8 周：自定义后处理**
- 在 UE 里做一个 **PostProcessMaterial**：
  - 读 SceneColor、SceneDepth、WorldNormal
  - 做边缘检测（Sobel / Roberts）
  - 做轮廓描边、卡通光照
- 进阶：做一个**屏幕空间扫描特效**（沿 Z 平面扫描，扫到的物体高亮）。这个 demo 会让你彻底理解 SceneDepth 和 WorldPos 重建。
- 产出物：**一段视频，展示你的描边/扫描效果，文档里写"我用了 SceneDepth 重建世界坐标，公式是..."**

**第 9 周：自定义 Shader 文件 (.usf)**
- 读 [UE 官方的 ShaderDevelopmentMode 文档]（`r.ShaderDevelopmentMode 1`）。
- 在你的项目里：
  - 创建一个自定义 `.usf` 文件
  - 在 Material 里通过 `Custom` 节点 `#include` 它
  - 写一个**自己的 BRDF**（比如 Disney Diffuse 或 Oren-Nayar）替换默认 Lambert
- 这一步是面试官最爱问的"你写过 shader 吗"的硬证据。

**第 10 周：性能分析**
- 学 `stat unit / stat gpu / stat scenerendering` 这些命令。
- 学 [Unreal Insights](https://dev.epicgames.com/documentation/en-us/unreal-engine/unreal-insights-in-unreal-engine) 抓性能。
- 找一个有点复杂的场景，**故意做一个性能问题**（比如几百个 dynamic light），用工具定位它，写优化前后对比报告。
- 产出物：**一份性能优化案例文档**。这是渲染岗简历最缺的东西。

### 阶段 D：综合项目（第 11-12 周）

> **目标：做一个能放在简历最前面的"压舱石"项目。**

挑一个你感兴趣的方向，做一个**完整的小项目**，不是 demo 是项目（有目标、有约束、有性能数据）：

候选（任选其一）：
1. **可交互体积云**：在 UE 里实现简化的 raymarching 体积云，能随时间漂移、能被光照
2. **卡通渲染管线**：完整的 NPR pipeline——卡通光照 + 描边 + 阴影硬切 + 角色渲染细节
3. **风格化水体**：自交互（有水波/反射/折射/泡沫），能跟玩家角色互动产生涟漪
4. **大世界材质优化**：复刻一个 Megascans 场景，做 LOD/距离衰减/性能优化案例

挑 1 个做，写完整文档，录视频。**这个就是你简历的封面项目**。

---

# 四、面试时怎么用这些产出物

每个 demo 的 README 写成这种结构（这是关键）：

```
# Demo: XXX 效果（一句话描述）

[GIF/Video]

## 实现要点
- 用了 UE 的 SceneTexture:CustomDepth 拿到深度
- 自己写了 GGX BRDF 替换默认（在 PixelShader.usf 里加了 #include MyShading.ush）
- 阴影用 PCF 5x5

## 性能
- 1080p RTX 3060 上 0.34ms
- 主要瓶颈在 X，做了 Y 优化降到 0.18ms

## 踩过的坑
- 一开始误把 UV 的 (0,0) 当作左上角，结果...

## 不足
- 还没处理 MSAA 兼容
- 半透明叠加顺序有 bug
```

**面试官最爱"踩过的坑"和"不足"这两节**。这两节告诉他你真做过、不是抄的。

---

# 五、资料推荐（精简版，不堆书）

不堆书单。你 3 个月，**只读以下**：

**视频/课程**（按顺序）：
1. **GAMES101**（闫令琪）—— B 站免费，第 7-10 课优先，其他选看
2. **LearnOpenGL.com**（在线免费）—— 你的 ShaderToy 周打底
3. **B 站搜"UE5 渲染管线"**—— 国人做的几个系列质量参差，挑剧本清晰的看 1-2 个就行

**书**（**只买/借 1 本，啃一年**）：
- **Real-Time Rendering 4th Edition** —— 不从头读，作为查阅手册。"我要做 Bloom"→ 翻第 12 章。

**博客 / 单篇**（按需查）：
- [Inigo Quilez 的文章](https://iquilezles.org/articles/) —— ShaderToy 圣经
- [Catlike Coding](https://catlikecoding.com/unity/tutorials/) —— Unity 的，但渲染原理通用，**讲解清晰胜过 UE 任何文档**
- [Shahriar Shahrabi](https://shahriyarshahrabi.medium.com/) —— 不少风格化效果原理
- [谢光辉的 UE Blog](https://zhuanlan.zhihu.com/p/116356381) 系列 —— UE 源码层级的中文资料

**不要看**：
- 不要花时间在 SIGGRAPH 论文上，**3 个月内用不上**
- 不要追每一个新 demo / Twitter 热点，会把节奏打乱
- 不要把 RTR4 当小说从头读，会读不完

---

# 六、几个关键决策

1. **不要"补完所有理论再开始实践"**。你已经掉进过这个坑。从第 1 周开始就动手。
2. **做的每个 demo 都丢 GitHub**，README 写好。3 个月后你简历有 ~10 个项目，是质变。
3. **每个周末必出一个产出**（哪怕是个小到不行的 ShaderToy）。**节奏比单点完美重要**。3 个月 12 周 = 12 个产出物。
4. **工作日的"实习+1-2 小时"**用来读、看视频、计划下周末做什么。**不要工作日做 demo**——会做不完，挫败感连锁。
5. **ShaderToy 是你的"周末快速验证器"**。某个原理没搞懂？周六上午写 30 行 ShaderToy 验证一下，比花一周读论文管用。

---

# 七、3 个月之后的"渲染岗简历样子"

你应该能写出这种描述（这是大厂渲染岗实习生 JD 上的真实表述）：

> - 在个人 UE5 项目中实现 X/Y/Z 效果（附 GitHub）
> - 使用 RenderDoc 抓帧分析 UE 渲染管线，输出技术总结
> - 通过自定义 .usf shader 实现自定义 BRDF / 后处理 X
> - 熟悉 PBR 渲染原理（D/F/G 项推导），并在 ShaderToy 实现
> - 使用 Unreal Insights 完成场景性能优化，[case 描述] 从 Yms 优化到 Zms

每一条**背后都有一个 GitHub 项目和文档支撑**。这种简历在客户端岗里是顶配，在渲染岗里也站得住。

---

# 八、还有哪些选项

如果你某个阶段觉得节奏吃不消，**砍掉 ShaderToy 周（第 6 周）**，把时间挪到 UE 渲染深入。如果你觉得游刃有余，**第 11-12 周的综合项目挑 2 个**。

如果你目标公司明确了（之后告诉我具体公司或岗位 JD），路线可以再调——比如某些公司更看中 Mobile 渲染（那 LWRP/URP 反而比 UE 桌面延迟管线更对口）、某些公司只看引擎源码贡献（那阶段 C 后期要直接改引擎）。

---

要不要我下一步帮你**把第 1 周的具体任务拆出来**？包括：要在 UE 哪个版本、新建什么类型的项目、第一个材质球该怎么开始连节点。这个我可以给你一个很具体的"第 1 天该干什么"的清单。