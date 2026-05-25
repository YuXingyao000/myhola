#!/usr/bin/env bash
# ============================================================================
# 2026-05-25 Data Generation: Blender + FLUX (Hydra-based)
# ============================================================================
#
# Uses the new Hydra config system (configs/datagen/).
# All parameters can be overridden via CLI: key=value
#
# Camera alignment: FOV=45°, normalize scale=0.9 (matches OCC renderer)
#
# ============================================================================

set -euo pipefail
cd /mnt/d/python

case "${1:-help}" in

  # --- Quick test (1 GPU, verify render works) ---
  test)
    echo "=== Quick test: Blender render, 1 GPU ==="
    python -m src.brepnet.data.datagen.run_blender \
      blender.num_gpus=1
    ;;

  # --- Blender cube24 render ---
  cube24)
    echo "=== Blender cube24 render (8 GPUs) ==="
    python -m src.brepnet.data.datagen.run_blender \
      mode=cube24
    ;;

  single-view)
    echo "=== Blender single-view render (8 GPUs) ==="
    python -m src.brepnet.data.datagen.run_blender \
      mode=single-view
    ;;

  # --- FLUX generation ---
  flux-cube24)
    echo "=== FLUX cube24-dynamic generation ==="
    python -m src.brepnet.data.datagen.run_flux \
      mode=cube24
    ;;

  flux-single-view)
    echo "=== FLUX single-view generation ==="
    python -m src.brepnet.data.datagen.run_flux \
      mode=single-view
    ;;

  flux-fixed-prompt)
    echo "=== FLUX with fixed prompt (pre_encode_text=True) ==="
    python -m src.brepnet.data.datagen.run_flux \
      mode=cube24 datagen/prompt=fixed
    ;;

  # --- Pack ---
  pack-cube24)
    echo "=== Pack cube24 FLUX outputs into npz ==="
    python -m src.brepnet.data.datagen.run_pack \
      mode=cube24
    ;;

  pack-single-view)
    echo "=== Pack single-view FLUX outputs into npz ==="
    python -m src.brepnet.data.datagen.run_pack \
      mode=single-view
    ;;

  # --- Full pipeline ---
  full)
    echo "=== Full pipeline: Blender → FLUX → Pack ==="
    python -m src.brepnet.data.datagen.run_pipeline \
      mode=cube24
    ;;

  # --- Mask (apply Blender silhouette to FLUX output) ---
  mask)
    echo "=== Apply Blender mask to FLUX outputs ==="
    python -m src.brepnet.data.datagen.mask_flux_with_blender
    ;;

  help|*)
    echo "Usage: bash experiments/2026-05-25/run_data_generation.sh <command>"
    echo ""
    echo "Blender render:"
    echo "  test              - Quick test (1 GPU)"
    echo "  cube24            - Full cube24 render (8 GPUs)"
    echo "  single-view       - Single-view render (8 GPUs)"
    echo ""
    echo "FLUX generation:"
    echo "  flux-cube24       - FLUX cube24-dynamic"
    echo "  flux-single-view  - FLUX single-view"
    echo "  flux-fixed-prompt - FLUX with fixed prompt mode"
    echo ""
    echo "Pack & post:"
    echo "  pack-cube24       - Pack into training npz"
    echo "  pack-single-view  - Pack single-view into npz"
    echo "  mask              - Apply Blender silhouette mask"
    echo ""
    echo "Full pipeline:"
    echo "  full              - Blender → FLUX → Pack (all-in-one)"
    echo ""
    echo "Override any param: bash ... cube24  # then manually add Hydra overrides"
    echo "  e.g. python -m src.brepnet.data.datagen.run_blender mode=cube24 blender.num_gpus=4"
    ;;
esac
