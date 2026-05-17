      1 +# Training Data Inventory
      2 +
      3 +Date: 2026-05-14
      4 +
      5 +This file records the data, split lists, generated artifacts, and external data roots currently used or referenced by the HoLa-BRep conditional d
         iffusion work. It is meant to help future repository cleanup without accidentally deleting training-critical files.
      6 +
      7 +## Current Core Training Command Inputs
      8 +
      9 +The recent conditional diffusion training commands use these external data roots:
     10 +
     11 +| Purpose | Path | Notes |
     12 +|---|---|---|
     13 +| Cached HoLa face latent features | `/mnt/d/data/ae_cache/1119_deepcad_aug1_11k` | Used as `dataset.face_z`; each item is loaded as `<prefix>_<a
         ug_id>/features.npy`. For `stored_z=true`, this contains mean/std latent features. |
     14 +| Conditional image data | `/mnt/d/data/deepcad_v6_cond` | Used as `dataset.cond_root`; contains `imgs.npz`, `single_view.npz`, and possibly cach
         ed image features per model. Current `single_img` training samples 20% FLUX images from `single_view.npz` and 80% rendered `svr_imgs` from `imgs.
         npz`. |
     15 +| Evaluation GT BRep/STEP root | `/mnt/d/data/deepcad_v6` | Used by post-eval scripts as `GT_ROOT`; expected to contain `<prefix>/normalized_shap
         e.step`. |
     16 +| Autoencoder checkpoint | `/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt` | Used as `model.autoencoder_weights`. |
     17 +| Conditional diffusion checkpoints | `/mnt/d/data/new_cond_ckpt` | Stores checkpoints such as 0504, 0511, 0513, and future 0514 addition-tag run
         s. |
     18 +| Conditional diffusion outputs | `/mnt/d/data/new_cond_results` | Stores raw generated outputs and post-eval summaries. Should stay outside git.
          |
     19 +
     20 +## Split Lists Kept In Repo
     21 +
     22 +These are small text files and should remain version-controlled. They define train/validation/test membership and are part of experiment reproduc
         ibility.
     23 +
     24 +### Main DeepCAD Splits
     25 +
     26 +```text
     27 +src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt
     28 +src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt
     29 +src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt
     30 +src/brepnet/data/list/deduplicated_deepcad_testing_7_30_sample_100.txt
     31 +src/brepnet/data/list/deduplicated_deepcad_testing_7_30_nvdnet_filtered.txt
     32 +```
     33 +
     34 +Also present:
     35 +
     36 +```text
     37 +src/brepnet/data/list/deepcad_training.txt
     38 +src/brepnet/data/list/deepcad_validation.txt
     39 +src/brepnet/data/list/deepcad_testing.txt
     40 +src/brepnet/data/list/deepcad_total.txt
     41 +```
     42 +
     43 +### ABC Splits
     44 +
     45 +```text
     46 +src/brepnet/data/list/abc_training.txt
     47 +src/brepnet/data/list/abc_validation.txt
     48 +src/brepnet/data/list/abc_testing.txt
     49 +src/brepnet/data/list/abc_total.txt
     50 +```
     51 +
     52 +### Backup/Legacy Split Lists
     53 +
     54 +These are small and useful for provenance, but should probably be moved under a clearer `data/splits/legacy/` directory later:
     55 +
     56 +```text
     57 +src/brepnet/data/list/bak/
     58 +src/brepnet/bak/list_bak/
     59 +```
     60 +
     61 +## Dataset Loading Code
     62 +
     63 +The main dataset code is:
     64 +
     65 +```text
     66 +src/brepnet/dataset.py
     67 +```
     68 +
     69 +Important behavior:
     70 +
     71 +- `Diffusion_dataset` loads `features.npy` from `dataset.face_z`.
     72 +- `prepare_condition()` loads image condition files from `dataset.cond_root`.
     73 +- With `pad_method=random`, face slots are padded to `num_max_faces` by repeating real face latents.
     74 +- The current addition-tag experiment appends a keep/drop tag to distinguish mandatory slots from repeated padding slots.
     75 +
     76 +Related data utility scripts:
     77 +
     78 +```text
     79 +src/brepnet/data/prepare_img_feature.py
     80 +src/brepnet/data/feature_similarity.py
     81 +src/brepnet/data/feature_similarity_lightning.py
     82 +src/brepnet/data/filter_out_faces.py
     83 +src/brepnet/data/convert_split_from_deepcad.py
     84 +src/brepnet/data/pack_conditional_data.py
     85 +src/brepnet/data/combine_data.py
     86 +src/brepnet/data/compress_natural_img.py
     87 +```
     88 +
     89 +## In-Repo Data Generation Material
     90 +
     91 +These paths currently contain data-generation code mixed with local artifacts. They should be cleaned carefully.
     92 +
     93 +### `src/brepnet/data/DataGeneration/`
     94 +
     95 +Current status: untracked in the clean `VAE` branch at the time of writing.
     96 +
     97 +Observed contents include:
     98 +
     99 +```text
    100 +src/brepnet/data/DataGeneration/01_sample_lists.sh
    101 +src/brepnet/data/DataGeneration/02_build_render_lists.sh
    102 +src/brepnet/data/DataGeneration/03_render_blender.sh
    103 +src/brepnet/data/DataGeneration/03_render_blender_cube24.sh
    104 +src/brepnet/data/DataGeneration/04_generate_flux.sh
    105 +src/brepnet/data/DataGeneration/05_pack_cond.sh
    106 +src/brepnet/data/DataGeneration/run_all_stages.sh
    107 +src/brepnet/data/DataGeneration/list/
    108 +src/brepnet/data/DataGeneration/render_lists/
    109 +src/brepnet/data/DataGeneration/test_list/
    110 +```
    111 +
    112 +High-risk non-code artifacts found in this directory:
    113 +
    114 +```text
    115 +src/brepnet/data/DataGeneration/blender-4.2.0-linux-x64/
    116 +src/brepnet/data/DataGeneration/materials/*.zip
    117 +src/brepnet/data/DataGeneration/Img2Brep-Dataset/model.pth
    118 +src/brepnet/data/DataGeneration/Img2Brep-Dataset/model2.pth
    119 +src/brepnet/data/DataGeneration/collected_samples.zip
    120 +src/brepnet/data/DataGeneration/logs_*/
    121 +```
    122 +
    123 +Recommendation:
    124 +
    125 +- Keep scripts and README in repo only after moving them to a clean tools directory.
    126 +- Do not commit Blender binaries, generated logs, material archives, or model weights.
    127 +- Move large artifacts to external storage, for example `/mnt/d/data/hola_artifacts/data_generation/`.
    128 +
    129 +### `src/brepnet/data/DataGenerationRefactored/`
    130 +
    131 +Current status: untracked in the clean `VAE` branch at the time of writing.
    132 +
    133 +Observed code/scripts:
    134 +
    135 +```text
    136 +src/brepnet/data/DataGenerationRefactored/run_pipeline.py
    137 +src/brepnet/data/DataGenerationRefactored/run_lists.py
    138 +src/brepnet/data/DataGenerationRefactored/run_blender.py
    139 +src/brepnet/data/DataGenerationRefactored/run_flux.py
    140 +src/brepnet/data/DataGenerationRefactored/run_pack.py
    141 +src/brepnet/data/DataGenerationRefactored/config.py
    142 +src/brepnet/data/DataGenerationRefactored/launcher.py
    143 +src/brepnet/data/DataGenerationRefactored/blender_scripts/
    144 +src/brepnet/data/DataGenerationRefactored/flux_scripts/
    145 +src/brepnet/data/DataGenerationRefactored/tools/
    146 +```
    147 +
    148 +High-risk non-code artifacts:
    149 +
    150 +```text
    151 +src/brepnet/data/DataGenerationRefactored/materials/*.zip
    152 +src/brepnet/data/DataGenerationRefactored/real_photo/*.jpg
    153 +src/brepnet/data/DataGenerationRefactored/real_photo/prepared/
    154 +src/brepnet/data/DataGenerationRefactored/runtime_lists/
    155 +src/brepnet/data/DataGenerationRefactored/machine_lists/
    156 +```
    157 +
    158 +Recommendation:
    159 +
    160 +- Keep the refactored generation pipeline code.
    161 +- Move real photos, prepared condition roots, runtime lists, material archives, and generated outputs outside git.
    162 +- If tiny test fixtures are needed, keep only a minimal anonymized sample under a dedicated `tests/fixtures/` directory.
    163 +
    164 +## Experiment Harness and Records
    165 +
    166 +The current condition-debug harness is tracked and should stay in repo:
    167 +
    168 +```text
    169 +src/brepnet/experiments/condition_debug/
    170 +EXPERIMENT_RECORD.md
    171 +```
    172 +
    173 +It contains:
    174 +
    175 +- E0 baseline sampling/eval script.
    176 +- E2 condition sensitivity script.
    177 +- E3 denoise probe script.
    178 +- E5 face-count summary script.
    179 +- Summarization utilities.
    180 +- Markdown reports and small report figures.
    181 +
    182 +External results referenced by the harness remain outside git:
    183 +
    184 +```text
    185 +/mnt/d/data/new_cond_results/exp_plan_0504_dino_baseline
    186 +/mnt/d/data/new_cond_results/exp_plan_0511_dino_cond_margin_s0p1_m0p01_shuffle
    187 +/mnt/d/data/new_cond_results/exp_plan_0513_dino_cond_margin_s0p1_m0p01_shuffle_t300
    188 +```
    189 +
    190 +## Real Photo Demo Data
    191 +
    192 +Observed local/untracked demo-related files:
    193 +
    194 +```text
    195 +src/brepnet/data/prepare_custom_6_demo.py
    196 +src/brepnet/scripts/eval_real_photo_00104204_cached_z.sh
    197 +src/brepnet/scripts/qualitative_custom_demo.sh
    198 +src/brepnet/results/
    199 +```
    200 +
    201 +Recommendation:
    202 +
    203 +- Keep only reusable scripts if needed.
    204 +- Move real photo inputs and generated results outside git.
    205 +- Keep a README describing where external demo assets should be placed.
    206 +
    207 +## Model/Third-Party Weights
    208 +
    209 +Observed or referenced weight/checkpoint locations:
    210 +
    211 +```text
    212 +/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt
    213 +/mnt/d/data/new_cond_ckpt/
    214 +thirdparty/Depth_Anything_V2/
    215 +src/brepnet/data/DataGeneration/Img2Brep-Dataset/model.pth
    216 +src/brepnet/data/DataGeneration/Img2Brep-Dataset/model2.pth
    217 +```
    218 +
    219 +Recommendation:
    220 +
    221 +- Do not store large model weights in the git repo.
    222 +- Keep download/setup instructions.
    223 +- If a third-party repository is needed, use a submodule or documented install script rather than copying full binaries/checkpoints into the code
          tree.
    224 +
    225 +## Files/Directories To Avoid Committing
    226 +
    227 +These should generally be gitignored or externalized:
    228 +
    229 +```text
    230 +wandb/
    231 +src/brepnet/results/
    232 +src/brepnet/data/DataGeneration/blender-4.2.0-linux-x64/
    233 +src/brepnet/data/DataGeneration/logs_*/
    234 +src/brepnet/data/DataGeneration/materials/*.zip
    235 +src/brepnet/data/DataGeneration/collected_samples.zip
    236 +src/brepnet/data/DataGenerationRefactored/materials/*.zip
    237 +src/brepnet/data/DataGenerationRefactored/real_photo/prepared/
    238 +src/brepnet/data/DataGenerationRefactored/runtime_lists/
    239 +thirdparty/Depth_Anything_V2/checkpoints/
    240 +*.ckpt
    241 +*.pth
    242 +*.pt
    243 +*.npz
    244 +*.npy
    245 +*.ply
    246 +*.step
    247 +*.stl
    248 +```
    249 +
    250 +Important exception:
    251 +
    252 +- Small split-list `.txt` files under `src/brepnet/data/list/` should remain tracked.
    253 +- Small report assets under `src/brepnet/experiments/condition_debug/report_assets/` are acceptable because they document the debugging process.
    254 +
    255 +## Suggested Future Layout
    256 +
    257 +Recommended cleanup target:
    258 +
    259 +```text
    260 +data/splits/
    261 +  deepcad/
    262 +  abc/
    263 +  legacy/
    264 +
    265 +tools/data_generation/
    266 +  blender_render/
    267 +  flux_generation/
    268 +  pack_condition/
    269 +
    270 +experiments/condition_debug/
    273 +  DATA_INVENTORY.md
    274 +  EXPERIMENT_RECORD.md
    275 +```
    276 +
    277 +External artifact root:
    278 +
    279 +```text
    280 +/mnt/d/data/hola/
    281 +  ae_cache/
    282 +  ae_checkpoints/
    283 +  cond_root/
    284 +  gt_step/
    285 +  diffusion_ckpt/
    286 +  diffusion_results/
    287 +  generation_artifacts/
    288 +```
    289 +
    290 +## Cleanup Warnings
    291 +
    292 +Before deleting or moving anything, verify these still resolve:
    293 +
    294 +```text
    295 +dataset.face_z=/mnt/d/data/ae_cache/1119_deepcad_aug1_11k
    296 +dataset.cond_root=/mnt/d/data/deepcad_v6_cond
    297 +dataset.train_dataset=src/brepnet/data/list/deduplicated_deepcad_training_7_30.txt
    298 +dataset.val_dataset=src/brepnet/data/list/deduplicated_deepcad_validation_7_30.txt
    299 +dataset.test_dataset=src/brepnet/data/list/deduplicated_deepcad_testing_7_30.txt
    300 +model.autoencoder_weights=/mnt/d/data/ae_checkpoints/1119_deepcad_aug1_11k.ckpt
    301 +GT_ROOT=/mnt/d/data/deepcad_v6
    302 +```
    303 +
    304 +These paths are the minimum required for reproducing the current conditional diffusion training/evaluation loop.