# Claude Code 指南 — 预处理

本目录的权威说明是 `AGENTS.md`；改动前先读。仓库入口为根目录 `CLAUDE.md`。

- 范围：DeepCAD／ABC 原始 CAD → `data.npz`、可见性缓存、cube24 旋转、共享边界与 UV 监督目标、`splits/` 固定清单。图片数据集生成在 `src/datagen/`，训练读取在 `src/datasets/`。
- 入口：`python -m src.preprocess.extract_brep`（显式路径与清单，拒绝覆盖，先 `--dry-run`）；`tools/filter_deepcad_v7_lists.py --dry-run`；报告目录显式传入并写到 `/mnt/d/hola_research/`。
- 契约：`rotation_id=0` 是 identity；latent、条件图、adjacency、visibility 同一对象／旋转／面索引。latent cache 是 mean+std。训练读 `imgs.npz["svr_imgs"]` 与 `real_photo.npz["flux"]`；拒绝旧 `natural.npz`。
- 共享边界：每对不同面恰两条互反行；目标逐对象保存，零边界合法，缺失／不一致带 ID 报错，不静默过滤。训练用 `splits/deepcad_v7_contract_*.txt`。
- 纪律：按 CAD identity 划分再派生视图／外观；不按测试成绩清洗测试集；派生数据写新位置并记录来源与参数。
