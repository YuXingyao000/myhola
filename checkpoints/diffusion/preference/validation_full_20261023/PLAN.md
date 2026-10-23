# 原 SVR 与 DPO：补全 validation 对照

## 1. Task
用户要求利用空闲GPU完成剩余validation样本，最终汇总完整验证集的双模型配对评估。

## 2. Assumptions
延续已完成100件的灰模单图view0、原HoLa VAE、K=1与固定最终DPO 5,000步权重。
只扩大样本清单；不训练、挑checkpoint、重抽候选、修改模型/指标/后处理或使用test。
8张RTX4090当前空闲，全部用于生成；CPU后处理并发仍为16。

## 3. Inputs
- 完整validation：`src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt`。
- 旧100件run：`checkpoints/diffusion/preference_eval/20261023T062400Z`。
- 原权重：`/mnt/d/ckpt/HoLa/Diffusion_svr_sq30_1500k.ckpt`。
- DPO：`checkpoints/diffusion_preference_svr/20261023T030527Z/weights/steps/step=00005000.ckpt`。
- 图片：`/mnt/d/data/deepcad_v7_cond/{id}/imgs.npz` 的 `svr_imgs[0]`。
- GT：`/mnt/d/data/deepcad_v7/{id}/data.npz` 与 `normalized_shape.step`。

## 4. Outputs
独立run中冻结完整清单、checkpoint及源码哈希、命令与Git状态；保存每件原始latent、
NPZ、STEP、状态和指标，结束自动生成完整分母的CSV/JSON/对照图。
旧100件通过只读引用复用，记录旧run协议与被复用ID，不复制或覆盖历史输出。

## 5. Processing Protocol
先核对清单唯一性、旧100件包含关系、train重叠与输入可用性。
复用前核对双checkpoint哈希、数据根、seed、采样及后处理协议，所有影响输出的
源码哈希相同；本轮只允许编排与汇总源文件变化。复用项必须已有完整终态。
其余每图每模型仅c00：seed=candidate_seed(42,id,0)，DDPM1000、FP32、TF32关闭、
batch1、原AE解码去重；两模型各分配4张GPU。
后处理recover_v1、drop0、200/自适应600、CPU、单候选总预算180秒、3次几何采样。
新样本两模型共用GT采样缓存及seed；复用项不再调用后处理或指标计算。

## 6. Metrics
与旧100件完全相同：推理/STEP覆盖率、合法solid、invalid/unknown及失败数量；
后处理前面/边双向平方Chamfer、Hungarian面F1、面数绝对误差；后处理后面/边
平方Chamfer及面F1（三次采样先在件内平均）。
各自可测集合与共同可测配对集合分别报告，全部失败保留在完整预定分母。
CAD配对bootstrap仍为seed42、10,000次，分批计算只降低内存，不改变抽样序列。

## 7. Decision Rule
固定原模型和最终5,000步DPO，报告合法性救回/退步与配对几何变化；不据结果自动调参。

## 8. Edge Cases
重复ID/错误split/协议不一致在运行前报错，不能混合不同来源结果。
缺输入、推理异常、OCC失败、超时或不可测几何记录为对应终态；不静默删除、不填零。
合法性已完成但几何采样失败仍保留合法标签。不重试失败候选或重抽以改善指标。
原100件只读引用，不把新cache路径传给旧评估以免改写历史。

## 9. Validation / Sanity Checks
验证完整清单和原100件关系、输入可用性及未使用test；检查显式样本量和目录约定。
针对复用协议与非100件汇总做必要检查；旧100汇总值及bootstrap区间保持不变。
启动后确认8个GPU进程、两模型原100件跳过、新ID输出及日志正常；后台继续运行。

## 10. Implementation Constraints
仅扩展现有`eval/preference_compare.py`及`preference_report.py`的固定清单支持，
保留旧100件接口兼容。显式路径和参数、简短注释，不引入新的模型或评估逻辑。

## 11. Deliverables
最小编排/汇总改动、检查记录、完整validation后台任务及可查询日志/状态路径。
完成后自动汇总，不新增训练或无限监控。
