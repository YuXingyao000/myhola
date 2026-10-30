# AgX成像对比度诊断

上一轮材质消融确认：程序条纹可见并不代表更像照片。本轮排查另一项有限假设：当前AgX额外对比度是否压平了可用的金属反射。

使用既有00301161零件在办公室、机加工车间、装配车间的Combined denoised线性EXR；同一张EXR只换AgX look，由Medium High Contrast变成Base Contrast。安装版OCIO中对应log域contrast为1.2和1.0，不改变曝光、白平衡或饱和度。装配场景保留原曝光+0.2，其余为0。

执行6张CPU纯合成输出，没有重新渲染、降噪、几何操作或模型调用。首次Image.save_render路径相对先前Compositor输出出现最大1/255量化差异，完整保留于`camera_response_20261029_001`；改用相同Compositor文件输出后，三场景baseline均与先前PNG逐像素完全一致。正式对照位于`/mnt/d/data/cad_scene_enhancements/camera_response_20261029_002`。

实际观察：候选画面整体更淡，暗部稍抬高，但没有补出缺失的表面细节，也没有明确的跨场景照片感优势。此处只排查一对look、一个形状，不是证明所有成像模型无效，也不是证明所有光照均正确。默认保持原look，用户评价待定。

`spec.json`为输入/比较协议；`convert.py`为CPU转换；`result.json`记录6张产物与baseline重建；`research.json`记录官方OCIO来源及历史检查。成像层不是本轮优先继续加复杂度的位置。
