# 2026-06-01 Topology Predictor Notes

## Data Integrity

- Found corrupted condition files after the dataset migration to `deepcad_v7_cond`.
- Symptom: training can fail inside a PyTorch `DataLoader` worker when NumPy reads an existing `.npz` file, for example:

```text
zlib.error: Error -3 while decompressing data: invalid code lengths set
```

- Observed path type: `svr.npz` while reading the `images` array for the single gray-image topology predictor.
- This means existence-only filtering, such as `has_required_condition_files`, is not sufficient to detect every bad sample. Some files exist but cannot be decompressed.
- The toy topology scripts now catch common `.npz` read/decompression errors in `__getitem__`, print the bad `model_id`, skip that sample, and continue training.
- Follow-up: after the current feasibility runs, scan `deepcad_v7_cond` for corrupted `.npz` files and either regenerate or remove the affected migrated condition folders.

## 24-View Conditioning

- Initial `svr24` and `sketch24` toy models averaged 24 DINOv2 CLS features directly.
- This lost explicit view/camera information.
- Updated `svr24` and `sketch24` scripts now attach cube24 `view_ids`, learn a view embedding, project the cube24 rotation matrix, and use attention pooling over the 24 image features.
