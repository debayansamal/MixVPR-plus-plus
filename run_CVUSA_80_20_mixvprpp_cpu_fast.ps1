$ErrorActionPreference = "Stop"

$Python = "C:\Users\KIIT\AppData\Local\Programs\Python\Python313\python.exe"

& $Python train_cvusa.py `
  --aggregator "MixVPR++" `
  --root "E:\CVUSA Dataset\Dataset_CVUSA" `
  --train_csv "E:\Transgeo VPR\TransGeo2022\data\custom_splits\train_80.csv" `
  --test_csv "E:\Transgeo VPR\TransGeo2022\data\custom_splits\test_20.csv" `
  --save_dir ".\LOGS\cvusa_80_20_mixvprpp_cpu_fast" `
  --batch_size 4 `
  --workers 0 `
  --epochs 1 `
  --lr 0.005 `
  --out_channels 256 `
  --out_rows 2
