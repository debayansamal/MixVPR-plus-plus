#!/usr/bin/env bash
set -euo pipefail

python train_cvusa.py \
  --aggregator "MixVPR++" \
  --root "/content/Dataset_CVUSA" \
  --train_csv "/content/TransGeo2022/data/custom_splits/train_80.csv" \
  --test_csv "/content/TransGeo2022/data/custom_splits/test_20.csv" \
  --save_dir "./LOGS/cvusa_80_20_mixvprpp_colab" \
  --batch_size 32 \
  --workers 2 \
  --epochs 20 \
  --lr 0.03 \
  --out_channels 1024 \
  --out_rows 4
