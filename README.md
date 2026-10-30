# Img2Brep

从单张图片重建 B-Rep，并构建来源可追溯的 CAD–图像配对数据。研究记录、数据、权重与逐对象结果保存在代码仓库外。

当前主线：冻结 DINOv2 → 图像 token 投影 → 24 层逐层 image cross-attention 的 face-latent DDPM → 冻结 VAE → OCC 后处理。photo baseline / DPO 使用固定历史 HoLa SVR 后端，权重分别加载。

## 环境安装

RTX 4090 本机已验证 Python 3.10、PyTorch 2.4.1 / CUDA 12.4、PythonOCC 7.8.1 API 系列。`environment.yml` 固定基础环境，Python 依赖统一在 `pyproject.toml` 声明。

```bash
git clone https://github.com/YuXingyao000/myhola.git Img2Brep
cd Img2Brep
conda env create -f environment.yml
conda activate img2brep
python -m pip install --no-build-isolation -e '.[dev]'
pre-commit install
```

已有环境可继续使用原名称；本轮没有重装，新建环境流程尚未从空环境复验。CUDA 扩展需要与 PyTorch 匹配的 CUDA toolkit 和 C++ 编译器。

RTX 5090 服务器已验收 PyTorch 2.8.0 / CUDA 12.8、PythonOCC 7.8.1 API 系列。重建时使用独立步骤：

```bash
conda create -n img2brep -c conda-forge python=3.10 pythonocc-core=7.8.1 pip
conda activate img2brep
python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.8.0+cu128.html
python -m pip install --no-build-isolation -e '.[dev]'
pre-commit install
```

### 历史 SVR / photo / DPO 后端

后端从固定 Git 提交 `c2660c00` 提取原网络，需要包含该提交的 Git 历史；浅克隆或单独的 wheel 不提供此后端。Pointnet2 是环境依赖，仓库中没有它的子模块或安装源码。需要构建时安装固定上游版本：

```bash
python -m pip install --no-build-isolation --no-deps 'git+https://github.com/erikwijmans/Pointnet2_PyTorch.git@b5ceb6d9ca0467ea34beb81023f96ee82228f626#subdirectory=pointnet2_ops_lib'
python -c 'from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG'
```

历史源码仅将 Pointnet2 导入改为环境包路径，网络定义和参数键保留。DINOv2 本地 torch.hub 源码也必须可用，可用 `dino_repo` 指定。

## 数据与训练

数据路径通过 Hydra overrides / CLI 显式指定。固定 split 在 `src/preprocess/splits/`，按 CAD identity 划分。

| 资产 | 接口 |
|---|---|
| 原始 B-Rep | `<raw_root>/<model_id>/data.npz` |
| 灰模图 | `<condition_root>/<model_id>/imgs.npz` 的 `svr_imgs` |
| 合成照片风格图 | 同目录 `real_photo.npz` 的 `flux`；历史文件名不表示实拍 |
| latent cache | `<cache_root>/<model_id>_<rotation_id>/features.npy`；32D VAE 常见为 mean(32) + std(32) |
| 边界 / UV 监督 | 按具体 VAE 配置准备对应 sidecar |

图像、latent、face 顺序、相机与 cube24 旋转必须一致。单张 `flux[H,W,3]` 仅支持 identity。当前灰模入口关闭 overfit，训练和验证 photo ratio 均为 0：

```bash
python -m src.train --config-name train_diffusion_layerwise_svr \
  dataset.raw_data_root="${RAW_DATA_ROOT:?}" \
  dataset.cached_latent_root="${LATENT_CACHE_ROOT:?}" \
  dataset.condition_data_root="${CONDITION_DATA_ROOT:?}" \
  dataset.train_dataset=src/preprocess/splits/deduplicated_deepcad_training_7_30.txt \
  dataset.val_dataset=src/preprocess/splits/deduplicated_deepcad_validation_7_30.txt \
  dataset.test_dataset=src/preprocess/splits/deduplicated_deepcad_testing_7_30.txt \
  model.autoencoder.checkpoint="${AE_CHECKPOINT:?}" \
  trainer.devices=1 trainer.batch_size=4 trainer.seed=42
```

| 入口 | 用途 |
|---|---|
| `train_diffusion_layerwise_real` | FLUX 合成图条件，训练和验证 photo ratio 均为 1 |
| `train_diffusion_photo_baseline` | 历史 SVR 初始化的照片 DDPM，读取已核验冻结特征缓存 |
| `train_diffusion_preference` / `train_diffusion_preference_photo` | SVR / photo DPO，需对应来源和偏好协议 |
| `train_vae_shared_curve_uv` | 32D shared curve + UV，需对应 sidecar 与初始化来源 |
| `train_vae_uv_intersection*` | UV intersection 实验，按各自架构快照使用 |

历史 `train_diffusion_white` / `train_diffusion_real` 的 `is_overfit=true`，后者验证 photo ratio 仍为 0，使用前核对配置；`baseline` 文件名也不自动代表旧网络。AE checkpoint 必须与 cache 同源。

新运行默认写入 `${IMG2BREP_RUNS_ROOT:-/mnt/d/hola_runs}/<exp_name>/<UTC timestamp>/`。已有 `hola_runs`、`hola_research`、`hola_archive` 目录保留；新启动使用 `IMG2BREP_RUNS_ROOT`，原 `HOLA_RUNS_ROOT` 需更新。`trainer.output_dir` 是根目录，`trainer.run_dir` 是完整 run 路径。

run 分设 `weights/`、`logs/`、`wandb/`、`raw_predictions/`、`visualizations/`、`metadata/`。用 `trainer.resume_from_checkpoint` 恢复优化器和步数；权重初始化创建新 run。新 W&B project 为 `img2brep`，旧 run 若继续使用原 project / ID，显式 override。

2026-10-30 调整过包和资产路径，复用旧记录时先确认路径；需要恢复旧配置时可先运行 `python tools/migrate_checkpoint.py OLD.ckpt NEW.ckpt --dry-run`，保留原权重。

## 单图推理

此 CLI 对应当前逐层条件 `Diffusion`，输入为图片文件列表。手写模型设置需与训练配置核对：

```bash
python -m src.inference --diffusion-weights "${DIFFUSION_CHECKPOINT:?}" \
  --autoencoder-weights "${AE_CHECKPOINT:?}" --condition single_img \
  --input /path/to/image.png --num-samples 1 --output-dir /path/to/new_inference_run
```

历史模型严格加载接口：`src.models.diffusion.legacy.load_legacy_svr` / `load_preference_svr`、`src.models.diffusion.photo_baseline.load_photo_svr`、`src.eval.photo_model_loading.load_photo_dpo`。photo DPO 的 reference 必须是对应 photo baseline。

## 评估口径

```bash
python -m src.evaluate --pred-root /path/to/postprocessed_predictions \
  --gt-root /path/to/ground_truth --split-list /path/to/frozen_ids.txt --metrics condition,validity
```

此入口的 `validity` 沿用历史 HoLa 检查：读取 STEP，对 solid 设置 `tolerance=0.1` 后执行 BRepCheck。原生检查由 `src/eval/step_metrics.py:native_validity` 提供：mode 0 重读后直接检查 as-read solid。两种口径分别报告，当前口径选择仍在讨论。

默认汇总有历史限制：condition 的 `num_total` 使用 split 与已有目录的交集；validity 的 worker 异常可能被跳过。baseline 另记固定预定总数、目录数、STEP 数、失败／超时数，并列固定分母和已有目录分母的比例。不能仅凭单个汇总字段确定论文当年的统计协议。

## 预处理与维护

`python -m src.preprocess.extract_brep --help` 查看 STEP 提取接口；[datagen README](src/datagen/README.md) 描述场景制作、Blender 拍摄与修图。

审计与筛单工具显式传入数据、缓存和仓库外的报告目录：

```bash
python tools/audit_deepcad_v7_dataset.py --data-root "${RAW_DATA_ROOT:?}" \
  --condition-root "${CONDITION_DATA_ROOT:?}" --latent-root "${LATENT_CACHE_ROOT:?}" \
  --out-dir /path/to/research/topic/asset_audit --max-models 100
python tools/filter_deepcad_v7_lists.py --data-root "${RAW_DATA_ROOT:?}" \
  --condition-root "${CONDITION_DATA_ROOT:?}" --latent-root "${LATENT_CACHE_ROOT:?}" \
  --output-dir /path/to/derived_splits --report-dir /path/to/research/topic/list_filter \
  --no-require-real-photo --dry-run
```

`tools/prepare_photo_baseline.py` 的 `--checkpoint`、`--conditions`、`--source-run`、`--output` 同样显式指定，源 run 使用实际新位置。历史产物与哈希保持原样，路径迁移不代表旧生产任务可原地续跑。

## 结构与检查

```text
src/
  train.py  inference.py  evaluate.py
  common/        运行溯源、IO、OCC 公共工具
  preprocess/    原始 CAD 预处理、旋转、监督目标和固定 split
  datagen/       场景制作、Blender 拍摄与合成真实感图像
  datasets/      训练读取
  preference/    DPO 偏好数据生成与选择
  models/        vae/、diffusion/ 和共享算子
  postprocess/   拟合、裁剪、缝合与 STEP 构建
  eval/          指标、评估与报告
configs/  tests/  tools/
```

规则见 [AGENTS.md](AGENTS.md)，代码经短期 PR 合入 `main`。GitHub Actions 执行布局、禁止数据／子模块、可达性、import 分层、Ruff 和无 OCC/CUDA 依赖的维护测试。模型、OCC、真实小批量与 GPU 测试在已验收的本机／服务器环境执行并记录命令；本轮没有注册自托管 runner 或部署任务。

```bash
pre-commit run --all-files
python -m pytest tests/test_maintenance_tools.py
```

## 历史来源

原始方法与权重来源：[HoLa: B-Rep Generation using a Holistic Latent Representation](https://arxiv.org/abs/2504.14257)，ACM TOG / SIGGRAPH 2025。引用原始方法时保留论文名称和作者；Img2Brep 的研究改动以各次实现、checkpoint 和评估协议为准。
