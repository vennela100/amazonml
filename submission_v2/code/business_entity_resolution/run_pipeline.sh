#!/usr/bin/env bash
# End-to-end: data -> normalization -> blocking -> matcher training -> test inference.
# Usage: bash run_pipeline.sh <dataset_dir> <work_dir> <output_dir>
#   dataset_dir contains train/ and test/ (the provided student_resource/dataset)
set -euo pipefail
DATA="${1:?dataset dir}"; WORK="${2:?work dir}"; OUT="${3:?output dir}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/src"
mkdir -p "$WORK" "$OUT"
cd "$SRC"
python -X utf8 prep.py  --data "$DATA" --split train --out "$WORK"
python -X utf8 prep.py  --data "$DATA" --split test  --out "$WORK"
python -X utf8 block.py --work "$WORK" --split train --k 10 --max-df 3000
python -X utf8 block.py --work "$WORK" --split test  --k 10 --max-df 3000
python -X utf8 train.py --work "$WORK" --gt "$DATA/train/train_ground_truth.tsv" --train-pool 900000 --trees 700 --out model
python -X utf8 predict.py --work "$WORK" --model "$WORK/model" --out "$OUT"
