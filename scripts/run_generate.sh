#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/../dataset"
CKPT_DIR="$SCRIPT_DIR/../checkpoints"

SAM2_CKPT="$CKPT_DIR/sam2.1_hiera_large.pt"
RAM_CKPT="$CKPT_DIR/ram_plus_swin_large_14m.pth"

# ---------------------------------------------------------------------------
# 실행할 작업 선택 (1=실행, 0=스킵)
# ---------------------------------------------------------------------------
RUN_EDGE=0
RUN_SEG_SEGFORMER=0
RUN_SEG_RAM=0
RUN_SEG_PKL=0
RUN_DEPTH=1

cd "$SCRIPT_DIR"

# ---------------------------------------------------------------------------
# Edge
# ---------------------------------------------------------------------------
if [ "$RUN_EDGE" -eq 1 ]; then
    echo "========================================="
    echo "[1/4] Edge generation"
    echo "========================================="
    python edge_generate.py \
        --input_dir  "$DATASET_DIR/CAM_F0" \
        --output_dir "$DATASET_DIR/CAM_F0_edge"
fi

# ---------------------------------------------------------------------------
# Seg: SegFormer
# ---------------------------------------------------------------------------
if [ "$RUN_SEG_SEGFORMER" -eq 1 ]; then
    echo "========================================="
    echo "[2/4] Seg: SegFormer (Cityscapes)"
    echo "========================================="
    python seg_generate.py \
        --input_dir  "$DATASET_DIR/CAM_F0" \
        --output_dir "$DATASET_DIR/CAM_F0_seg_segformer"
fi

# ---------------------------------------------------------------------------
# Seg: RAM++ → GroundingDINO → SAM2
# ---------------------------------------------------------------------------
if [ "$RUN_SEG_RAM" -eq 1 ]; then
    echo "========================================="
    echo "[3/4] Seg: RAM++ → GroundingDINO → SAM2"
    echo "========================================="
    python seg_generate_ram.py \
        --input_dir       "$DATASET_DIR/CAM_F0" \
        --output_dir      "$DATASET_DIR/CAM_F0_seg_ram" \
        --sam2_checkpoint "$SAM2_CKPT" \
        --ram_checkpoint  "$RAM_CKPT"
fi

# ---------------------------------------------------------------------------
# Seg: pkl → SAM2
# ---------------------------------------------------------------------------
if [ "$RUN_SEG_PKL" -eq 1 ]; then
    echo "========================================="
    echo "[4/4] Seg: pkl + SAM2"
    echo "========================================="
    python seg_generate_from_pkl.py \
        --pkl_dir         "$DATASET_DIR" \
        --img_root        "$DATASET_DIR/CAM_F0" \
        --output_dir      "$DATASET_DIR/CAM_F0_seg_pkl" \
        --sam2_checkpoint "$SAM2_CKPT"
fi

# ---------------------------------------------------------------------------
# Depth: Depth Anything V2
# ---------------------------------------------------------------------------
if [ "$RUN_DEPTH" -eq 1 ]; then
    echo "========================================="
    echo "[5/5] Depth: Depth Anything V2"
    echo "========================================="
    python depth_generate.py \
        --input_dir  "$DATASET_DIR/CAM_F0" \
        --output_dir "$DATASET_DIR/CAM_F0_depth"
fi

echo ""
echo "All done."
echo "  edge          → $DATASET_DIR/CAM_F0_edge"
echo "  seg_segformer → $DATASET_DIR/CAM_F0_seg_segformer"
echo "  seg_ram       → $DATASET_DIR/CAM_F0_seg_ram"
echo "  seg_pkl       → $DATASET_DIR/CAM_F0_seg_pkl"
echo "  depth         → $DATASET_DIR/CAM_F0_depth"
