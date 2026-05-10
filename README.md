# MixVPR++ for CVUSA Cross-View Geo-Localization

This repository is a modified version of the official **MixVPR: Feature Mixing for Visual Place Recognition** codebase. It adds a **MixVPR++-style** aggregation module and CVUSA training/evaluation support for cross-view geo-localization experiments.

The goal of this project is to train and evaluate a MixVPR++ variant on a CVUSA-style dataset, then compare its recall performance with TransGeo.

## Important Note

There is currently no official public GitHub implementation of MixVPR++. The implementation in this repository is a practical research reproduction based on the main architectural ideas described for MixVPR++:

- Adaptive Gabor Texture Fuser
- Hierarchical-Region Feature-Mixer

So this repository should be described as a **MixVPR++-style implementation**, not official author code.

## Main Changes

### MixVPR++ Aggregator

Added in:

```text
models/aggregators/mixvpr.py
```

The new `MixVPRPlusPlus` module includes:

- learnable Gabor-style texture filters
- adaptive texture fusion
- global feature mixing
- multi-scale region feature mixing
- final normalized global descriptor output

It can be selected with:

```text
--aggregator "MixVPR++"
```

### CVUSA Dataset Support

Added:

```text
dataloaders/CVUSADataset.py
dataloaders/CVUSADataModule.py
train_cvusa.py
```

The loader supports the alternate CVUSA-style dataset layout used locally:

```text
E:\CVUSA Dataset\Dataset_CVUSA
```

It also handles CSV/file-name mismatches such as:

```text
streetview/input0000008.jpg
```

when the actual file is:

```text
streetview/0000008.jpg
```

### Custom CVUSA Splits

The training script supports custom train/test CSV files. The current local split files are expected at:

```text
E:\Transgeo VPR\TransGeo2022\data\custom_splits\train_80.csv
E:\Transgeo VPR\TransGeo2022\data\custom_splits\test_20.csv
E:\Transgeo VPR\TransGeo2022\data\custom_splits\train_70.csv
E:\Transgeo VPR\TransGeo2022\data\custom_splits\test_30.csv
```

Known split counts:

| Split | Train pairs | Test pairs |
|---|---:|---:|
| 80/20 | 7089 | 1773 |
| 70/30 | 6203 | 2659 |

## Repository Structure

```text
main.py                              Original Lightning VPR model, updated for CVUSA validation
train_cvusa.py                       CVUSA training entry point
dataloaders/CVUSADataset.py          CVUSA pair and validation datasets
dataloaders/CVUSADataModule.py       Lightning data module for CVUSA
models/aggregators/mixvpr.py         MixVPR and MixVPR++ aggregators
models/helper.py                     Aggregator/backbone factory
run_CVUSA_80_20_mixvprpp_cpu_fast.ps1
run_CVUSA_70_30_mixvprpp_cpu_fast.ps1
run_CVUSA_80_20_mixvprpp_colab.sh
```

## Environment

Development environment used:

```text
Windows laptop
Python 3.13
torch 2.11.0+cpu
torchvision 0.26.0+cpu
No NVIDIA CUDA GPU
Ryzen 7 8840HS
32 GB RAM
```

CPU training is expected to be slow. The local PowerShell scripts are intended only for quick pilot runs. Use Colab or another GPU machine for real training.

## Install Dependencies

The original MixVPR requirements were written for older Python/PyTorch versions, so installing `requirements.txt` directly may not work on Python 3.13.

At minimum, this project needs:

```powershell
pip install pytorch-lightning pytorch-metric-learning faiss-cpu prettytable pandas pillow scikit-learn scipy timm tensorboard
```

If using the local Python installed at:

```text
C:\Users\KIIT\AppData\Local\Programs\Python\Python313\python.exe
```

run:

```powershell
& "C:\Users\KIIT\AppData\Local\Programs\Python\Python313\python.exe" -m pip install pytorch-lightning pytorch-metric-learning faiss-cpu prettytable pandas pillow scikit-learn scipy timm tensorboard
```

## Quick Sanity Checks

From the repository folder:

```powershell
cd "E:\MixVPR\MixVPR++"
```

Compile-check the modified files:

```powershell
& "C:\Users\KIIT\AppData\Local\Programs\Python\Python313\python.exe" -m py_compile main.py models\aggregators\mixvpr.py models\helper.py dataloaders\CVUSADataset.py dataloaders\CVUSADataModule.py train_cvusa.py
```

Check the MixVPR++ aggregator forward pass:

```powershell
& "C:\Users\KIIT\AppData\Local\Programs\Python\Python313\python.exe" -c "import torch; from models.aggregators.mixvpr import MixVPRPlusPlus; m=MixVPRPlusPlus(in_channels=8,in_h=4,in_w=4,out_channels=16,mix_depth=1,out_rows=2,region_scales=(1,2)); y=m(torch.randn(2,8,4,4)); print(tuple(y.shape), y.norm(dim=1).mean().item())"
```

Expected output shape:

```text
(2, 32)
```

## Local CPU Pilot Training

80/20 split:

```powershell
cd "E:\MixVPR\MixVPR++"
.\run_CVUSA_80_20_mixvprpp_cpu_fast.ps1
```

70/30 split:

```powershell
cd "E:\MixVPR\MixVPR++"
.\run_CVUSA_70_30_mixvprpp_cpu_fast.ps1
```

These scripts use reduced descriptor settings:

```text
out_channels = 256
out_rows = 2
epochs = 1
batch_size = 4
```

They are for smoke testing only, not final reporting.

## Full CVUSA Training Command

Example 80/20 MixVPR++ training command:

```powershell
& "C:\Users\KIIT\AppData\Local\Programs\Python\Python313\python.exe" train_cvusa.py `
  --aggregator "MixVPR++" `
  --root "E:\CVUSA Dataset\Dataset_CVUSA" `
  --train_csv "E:\Transgeo VPR\TransGeo2022\data\custom_splits\train_80.csv" `
  --test_csv "E:\Transgeo VPR\TransGeo2022\data\custom_splits\test_20.csv" `
  --save_dir ".\LOGS\cvusa_80_20_mixvprpp" `
  --batch_size 32 `
  --workers 2 `
  --epochs 20 `
  --lr 0.03 `
  --out_channels 1024 `
  --out_rows 4
```

## Colab Training

A Colab-oriented script is provided:

```bash
bash run_CVUSA_80_20_mixvprpp_colab.sh
```

It assumes the following paths inside Colab:

```text
/content/Dataset_CVUSA
/content/TransGeo2022/data/custom_splits/train_80.csv
/content/TransGeo2022/data/custom_splits/test_20.csv
```

Edit the script if your uploaded/extracted dataset paths are different.

## Evaluation Metric

Validation uses reference-first/query-second descriptor extraction:

```text
satellite images = references
ground/streetview images = queries
```

Recall is computed using FAISS L2 nearest-neighbor search. For CVUSA, each query has one positive reference with the same CSV row index.

Logged metrics include:

```text
cvusa/R1
cvusa/R5
cvusa/R10
```

## Comparison Target

The TransGeo baseline previously evaluated on the current CVUSA-style dataset gave:

| Model | Recall@1 | Recall@5 | Recall@10 | Top 1% |
|---|---:|---:|---:|---:|
| TransGeo pretrained CVUSA | 99.19 | 99.91 | 99.95 | 99.95 |

After MixVPR++ training, compare:

| Model | Split | Recall@1 | Recall@5 | Recall@10 |
|---|---|---:|---:|---:|
| MixVPR++ | 80/20 | TBD | TBD | TBD |
| MixVPR++ | 70/30 | TBD | TBD | TBD |
| TransGeo | same dataset | 99.19 | 99.91 | 99.95 |

## GitHub Upload Notes

Do not upload datasets, checkpoints, or logs to GitHub.

Recommended ignored paths:

```text
LOGS/
datasets/
*.ckpt
*.pth
*.pth.tar
*.pt
__pycache__/
```

## Citation

Original MixVPR paper:

```bibtex
@inproceedings{ali2023mixvpr,
  title={{MixVPR}: Feature Mixing for Visual Place Recognition},
  author={Ali-bey, Amar and Chaib-draa, Brahim and Giguere, Philippe},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  pages={2998--3007},
  year={2023}
}
```

If you use this repository, also cite the MixVPR++ paper that motivated the modifications.
