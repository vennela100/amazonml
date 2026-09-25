#!/usr/bin/env bash
# run_pipeline.sh - Full training pipeline (Stages 1-6)
# Run from project root: bash run_pipeline.sh
#
# Prerequisites:
#   - Python 3.10+, packages in requirements.txt installed
#   - Dataset at: dataset/train/*.tsv, dataset/test/*.tsv
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$ROOT/src"
DATA_DIR="$ROOT/../../dataset"   # adjust if needed
ARTIFACTS="$ROOT/../../artifacts"
REPORTS="$ROOT/../../reports"

echo "========================================"
echo "  Entity Resolution Pipeline"
echo "========================================"

# Stage 1: Splits + scoring harness
echo ""
echo "[Stage 1] Building splits..."
python -X utf8 "$SRC/splits.py" \
  --data-dir "$DATA_DIR/train" \
  --splits-db "$REPORTS/stage1/splits.sqlite" \
  --report-dir "$REPORTS/stage1"

# Stage 2: Normalization
echo ""
echo "[Stage 2] Normalizing..."
python -X utf8 "$SRC/normalization.py" \
  --data-dir "$DATA_DIR/train" \
  --out-dir "$ARTIFACTS/stage2/train" \
  --report-dir "$REPORTS/stage2"

# Stage 3: Candidate generation (blocking)
echo ""
echo "[Stage 3] Generating candidates..."
python -X utf8 "$SRC/stage3.py" \
  --data-dir "$ARTIFACTS/stage2" \
  --split train \
  --sources s2,s3 \
  --routes name,address,rare \
  --top-k-name 50 \
  --top-k-address 30 \
  --max-features 131072 \
  --chunk-size 200000 \
  --threshold 0.05 \
  --report-dir "$REPORTS/stage3" \
  --splits-db "$REPORTS/stage1/splits.sqlite" \
  --eval-role validation \
  --overwrite

# Stage 4: Pairwise features
echo ""
echo "[Stage 4] Computing pairwise features..."
python -X utf8 "$SRC/stage4.py" \
  --data-dir "$ARTIFACTS/stage2" \
  --split train \
  --cand-db "$REPORTS/stage3/candidates.sqlite" \
  --splits-db "$REPORTS/stage1/splits.sqlite" \
  --report-dir "$REPORTS/stage4"

# Stage 5: Train classifier
echo ""
echo "[Stage 5] Training CatBoost classifier..."
python -X utf8 "$SRC/stage5.py" \
  --features-npz "$REPORTS/stage4/features.npz" \
  --splits-db "$REPORTS/stage1/splits.sqlite" \
  --report-dir "$REPORTS/stage5" \
  --n-folds 5 \
  --n-estimators 2000 \
  --depth 6 \
  --learning-rate 0.05 \
  --pos-weight 5.0

# Stage 6: Generate predictions
echo ""
echo "[Stage 6] Generating submission..."
python -X utf8 "$SRC/stage6.py" \
  --features-npz "$REPORTS/stage4/features.npz" \
  --model "$REPORTS/stage5/model.cbm" \
  --calibrator "$REPORTS/stage5/calibrator.pkl" \
  --cv-results "$REPORTS/stage5/cv_results.json" \
  --report-dir "$REPORTS/stage6"

echo ""
echo "========================================"
echo "  Pipeline complete!"
echo "  Submission: $REPORTS/stage6/submission.tsv"
echo "========================================"
