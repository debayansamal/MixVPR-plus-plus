#!/usr/bin/env bash
set -euo pipefail

python train_cvusa.py \
  --aggregator "MixVPR++" \
  --backbone "resnet18" \
  --root "/content/drive/MyDrive/Dataset_CVUSA" \
  --train_csv "/content/drive/MyDrive/TransGeo2022/data/custom_splits/train_80.csv" \
  --test_csv "/content/drive/MyDrive/TransGeo2022/data/custom_splits/test_20.csv" \
  --save_dir "/content/drive/MyDrive/MixVPR++_results/80_20_fast" \
  --batch_size 64 \
  --workers 2 \
  --epochs 5 \
  --lr 0.01 \
  --image_size 224 \
  --out_channels 256 \
  --out_rows 2 \
  --mix_depth 2 \
  --region_scales 1,2 \
  --gabor_orientations 2 \
  --max_train_pairs 2500 \
  --max_test_pairs 500 \
  --accelerator gpu \
  --precision 16-mixed
