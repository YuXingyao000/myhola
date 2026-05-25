#!/usr/bin/env bash
# Simple command reference for DataGenerationRefactored.
# No shared variables on purpose; commands are written out explicitly.
#
# Run from repository root:
#   cd /mnt/d/python
#
# Usage examples:
#   bash src/brepnet/data/DataGenerationRefactored/commands_reference.sh lists
#   bash src/brepnet/data/DataGenerationRefactored/commands_reference.sh split-machines
#   bash src/brepnet/data/DataGenerationRefactored/commands_reference.sh blender-cube24-machine0

set -euo pipefail

case "${1:-help}" in
  lists)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_lists \
      --num-gpus 8
    ;;

  split-machines)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.split_model_lists \
      /mnt/d/python/src/brepnet/data/DataGenerationRefactored/render_lists/all.txt \
      --output-dir /mnt/d/python/src/brepnet/data/DataGenerationRefactored/machine_lists \
      --num-machines 2 \
      --strategy round-robin
    ;;

  split-machines-check-mesh)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.split_model_lists \
      /mnt/d/python/src/brepnet/data/DataGenerationRefactored/render_lists/all.txt \
      --output-dir /mnt/d/python/src/brepnet/data/DataGenerationRefactored/machine_lists \
      --num-machines 2 \
      --strategy round-robin \
      --check-mesh \
      --model-root /mnt/d/data/deepcad_v6 \
      --mesh-name mesh.stl
    ;;

  blender-cube24-machine0)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_blender cube24 \
      --machine-list-dir /mnt/d/python/src/brepnet/data/DataGenerationRefactored/machine_lists \
      --machine-index 0 \
      --num-gpus 8
    ;;

  blender-cube24-machine1)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_blender cube24 \
      --machine-list-dir /mnt/d/python/src/brepnet/data/DataGenerationRefactored/machine_lists \
      --machine-index 1 \
      --num-gpus 8
    ;;

  blender-single-view)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_blender single-view \
      --num-gpus 8
    ;;

  blender-single-view-metal010)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_blender single-view-metal010 \
      --num-gpus 8
    ;;

  blender-cube24-small-test)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_blender cube24 \
      --num-gpus 1 \
      --max-models 2
    ;;

  flux-cube24)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_flux cube24-dynamic \
      --num-gpus 8
    ;;

  flux-single-view)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_flux single-view \
      --num-gpus 8
    ;;

  flux-single-view-machine0)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_flux single-view \
      --machine-list-dir /mnt/d/python/src/brepnet/data/DataGenerationRefactored/machine_lists \
      --machine-index 0 \
      --render-root /mnt/d/python/src/brepnet/data/DataGenerationRefactored/outputs/blender_single_view \
      --output-root /mnt/d/python/src/brepnet/data/DataGenerationRefactored/outputs/flux_single_view \
      --num-gpus 8
    ;;

  flux-single-view-machine1)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_flux single-view \
      --machine-list-dir /mnt/d/python/src/brepnet/data/DataGenerationRefactored/machine_lists \
      --machine-index 1 \
      --render-root /mnt/d/python/src/brepnet/data/DataGenerationRefactored/outputs/blender_single_view \
      --output-root /mnt/d/python/src/brepnet/data/DataGenerationRefactored/outputs/flux_single_view \
      --num-gpus 8
    ;;

  pack-cube24)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_pack cube24
    ;;

  pack-single-view)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_pack single-view
    ;;

  pipeline-cube24-small-test)
    cd /mnt/d/python
    python -m src.brepnet.data.DataGenerationRefactored.run_pipeline cube24 \
      --num-gpus 1 \
      --max-models 2
    ;;

  help|*)
    echo "Available commands:"
    echo "  lists"
    echo "  split-machines"
    echo "  split-machines-check-mesh"
    echo "  blender-cube24-machine0"
    echo "  blender-cube24-machine1"
    echo "  blender-single-view"
    echo "  blender-single-view-metal010"
    echo "  blender-cube24-small-test"
    echo "  flux-cube24"
    echo "  flux-single-view"
    echo "  flux-single-view-machine0"
    echo "  flux-single-view-machine1"
    echo "  pack-cube24"
    echo "  pack-single-view"
    echo "  pipeline-cube24-small-test"
    ;;
esac
