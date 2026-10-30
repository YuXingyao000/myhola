# 看图评议

逐 region_id 读取数值报告，并实际查看 images 中的 before 和 after；有 visibility 时
一并查看指定组的掩码。before 没有方体，after 包含整理后的物件和方体，两图机位相同。

检查：
1. 灰方体是否像 20 cm 物品，处于画面重心，周围比例、背景和拍摄视角是否自然。
2. 被挪动的物件是否完整、仍可辨认、像临时推到旁边；不是消失在镜头外或整齐排成边框。
3. 环境的使用状态是否保留，照明与景深层次是否支持拍摄，而非空洞的产品棚。

给每个候选 keep / revise / drop、reason、suggested_change 和真正查看过的完整
inspected_images 路径。numeric_pass=false 或没有前后图片时不能 keep；对这种项
说明可修正的原因或放弃理由。没有看到图片就不要声称已经看图。keep 不是人工通过。
