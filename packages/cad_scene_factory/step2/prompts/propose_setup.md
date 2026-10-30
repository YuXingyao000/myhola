# 提议本轮完整摆放方案

为输入中的每个 region_id 返回一个 setup。读取测量摘要，必要时打开 full_measurements；
free 表示当前空位，movable 表示占用者都在你给出的白名单中，fixed_or_unknown 不能当成
已经腾出的空位。采样不是最终通过，外围会对你选择的精确落点重新检查。

选择 support_point_world、camera_position_world、look_at_world，给出相对基础版本的
完整 moves。每项包含 group_id、pivot_world、translation_xy_m、yaw_degrees。
该轮不需要移动的组不写入 moves。retain_in_frame 必须包含所有移动组，可以再包含
希望保留在照片里的未移动组。不得靠把物件移到镜头外达到“整洁”。

利用对象边界和测量高度安排终点：同一支撑面、边界以内、避开其他物件及主体拍摄通道。
遵守 config 的移动数、距离、角度和摄影范围。优先保持少量挪动、自然背景和斜侧视角，
不要让看向方向与世界 Z 平行。灰方体底面中心和相机目标点不是同一点。

若有 previous_attempt，先读数值失败、对照图及上轮视觉意见，再给完整替换方案；
不能沿用上轮已经位移的场景。reason 简短说明这一轮如何选位、构图或修正。
