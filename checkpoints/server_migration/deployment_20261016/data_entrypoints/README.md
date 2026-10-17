# 数据入口修复与验证（2026-10-16）

本轮仅修复迁移时可直接处理的参数、路径与配置入口；未进行 ABC 全库预处理、ABC 评估或 FLUX 正式生成。所有验证使用临时合成文件和 CPU，源机训练未被中断。

## 已修复

- `configs/datagen/` 各配置组显式声明 Hydra package，使 `cfg.paths`、`cfg.blender`、`cfg.flux`、`cfg.prompt` 与入口读取一致；保留原 `datagen/prompt=fixed` / `dynamic` 切换语法，以及 `paths.*`、`blender.*`、`flux.*` 覆盖方式。
- `run_flux` 将完整 prompt 配置传给两种 worker；单视图接受原上游 `--prompt-base`，并实际应用负提示和 `pre_encode_text`。动态模式按 CAD ID / seed 使用 YAML 中的材质、加工、表面和状态配置，替换 `{material}`；固定模式保持固定文字与预编码。动态 prompt 配合预编码会明确报错，防止复用过期文本嵌入。
- 底层 FLUX 模型与 Ray legacy 入口改为按需导入；`--help` 不再要求先成功导入整个 Diffusers pipeline，空分片也不会加载模型。现有直接 worker 调用不传完整配置时仍保留其历史 prompt 逻辑。
- 新增 `python -m src.brepnet.data.extract_brep`，要求显式 `--data-root`、`--output-root`、`--model-list`，可重复传 `--exclude-list`，不再引用失效机器路径或隐式例外文件。`--dry-run` 仅检查清单和路径，不读取 CAD / 不导入 OCC。实际执行拒绝非空输出目录和输入输出重叠；旧入口也委派到新入口，移除无条件删除输出根的行为。
- ABC 几何算法、采样分辨率和阈值保持原实现。新入口默认单 CPU worker；多 worker 时限制待执行 Ray 任务数量，保存源清单路径与 SHA256、计数、逐件成功/失败状态。它不创建研究 split，也不生成条件图或 AE latent。

## 验证结果

```bash
CUDA_VISIBLE_DEVICES='' /mnt/d/anaconda3/envs/torch/bin/python -m pytest -q src/brepnet/test/test_data_entrypoints.py
```

结果：**14 passed in 2.58s**。覆盖：固定/动态 × 单视图/cube24 的 launcher→CLI 参数合同；合成 4×4 PNG 与替代 pipeline 的实际 prompt/negative/pre-encode/seed 传递；`run_blender`、`run_pack` 的共享配置与 `${oc.env:...}` 路径合成；ABC 仅使用虚构 ID 的 dry-run、显式排除、既有输出保护与路径重叠拒绝。`run_pack` 使用替代 subprocess 验证命令组装，没有宣称其底层 pack 已可用。

另外通过：新 ABC CLI `--help`、两个 FLUX worker `--help`、真实 Hydra `run_flux --cfg job datagen/prompt=fixed mode=single-view paths.model_root=/tmp/fixture flux.num_gpus=1`、源 torch 环境中原 OCC 提取模块的纯导入、`git diff --check`。

## 未完成与使用边界

- ABC 仍缺完整 `data.npz` 几何库和非空 `1225_abc_aug1` cache；本轮入口修复不会补齐这些大量派生资产。尚未选择、查看或评估 ABC 最终评估样本，没有创建或改变 ABC split。实际处理需先确定开发清单、去重/重叠处理和隔离协议，使用新输出目录，先验证 PythonOCC 7.8.1 与代表性开发样本的提取合同。
- 本轮没有加载 FLUX 权重或执行真实生成。源 torch 环境中，Diffusers 0.37.1 与 Torch 2.4.1 的导入曾在 `attention_dispatch` 触发 `infer_schema` 类型注解错误；没有修改源环境。目标 5090 环境须实际导入/加载 pipeline 验证，不能以 CLI 帮助或 pip 安装完成替代。
- `run_pack` 的单视图旧 `PATHS/RUNTIME` 导入、cube24 仍写 `natural.npz`、Blender raw cube24 与运行时 identity-first 顺序映射、mask 阶段和逐图来源元数据仍是独立缺口。`run_pipeline` 仍未完整转发配置。本轮不把这些入口称为端到端可复现管线。
- 数据局部 `AGENTS.md` 的旧问题清单是 2026-10-12 审阅记录；其中 Hydra package、FLUX prompt 转发与单视图 CLI 不兼容问题以本记录更新，其余缺口继续有效。

新 ABC 入口的使用形式（占位路径，未执行）：

```bash
python -m src.brepnet.data.extract_brep \
  --data-root /path/to/abc_step \
  --output-root /path/to/new_abc_geometry \
  --model-list /path/to/approved_development_ids.txt \
  --num-workers 1 --dry-run
```

`--dry-run` 不验证每个 STEP 内容或输出数组；移除该参数才会执行几何提取。图像与 latent / face order / rotation 的对应关系仍需独立验证。
