import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from dataloaders.CVUSADataModule import CVUSADataModule
from main import VPRModel


def parse_region_scales(value):
    return tuple(int(item.strip()) for item in value.split(',') if item.strip())


def parse_args():
    parser = argparse.ArgumentParser(description='Train MixVPR or MixVPR++ on CVUSA-style splits.')
    parser.add_argument('--root', default=r'E:\CVUSA Dataset\Dataset_CVUSA')
    parser.add_argument('--train_csv', default=r'E:\Transgeo VPR\TransGeo2022\data\custom_splits\train_80.csv')
    parser.add_argument('--test_csv', default=r'E:\Transgeo VPR\TransGeo2022\data\custom_splits\test_20.csv')
    parser.add_argument('--save_dir', default='./LOGS/cvusa_mixvprpp')
    parser.add_argument('--aggregator', default='MixVPR++', choices=['MixVPR', 'MixVPR++'])
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--image_size', type=int, default=320)
    parser.add_argument('--out_channels', type=int, default=512)
    parser.add_argument('--out_rows', type=int, default=4)
    parser.add_argument('--mix_depth', type=int, default=4)
    parser.add_argument('--mlp_ratio', type=float, default=1.0)
    parser.add_argument('--region_scales', default='1,2,4')
    parser.add_argument('--gabor_kernel_size', type=int, default=7)
    parser.add_argument('--gabor_orientations', type=int, default=4)
    parser.add_argument('--fast_dev_run', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(seed=190223, workers=True)

    datamodule = CVUSADataModule(
        root=args.root,
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        num_workers=args.workers,
        show_data_stats=True,
    )

    agg_config = {
        'in_channels': 1024,
        'in_h': args.image_size // 16,
        'in_w': args.image_size // 16,
        'out_channels': args.out_channels,
        'mix_depth': args.mix_depth,
        'mlp_ratio': args.mlp_ratio,
        'out_rows': args.out_rows,
    }
    if args.aggregator == 'MixVPR++':
        agg_config.update({
            'region_scales': parse_region_scales(args.region_scales),
            'gabor_kernel_size': args.gabor_kernel_size,
            'gabor_orientations': args.gabor_orientations,
        })

    model = VPRModel(
        backbone_arch=args.backbone,
        pretrained=args.pretrained,
        layers_to_freeze=2,
        layers_to_crop=[4],
        agg_arch=args.aggregator,
        agg_config=agg_config,
        lr=args.lr,
        optimizer='sgd',
        weight_decay=args.weight_decay,
        momentum=0.9,
        warmpup_steps=100,
        milestones=[2, 4, 8, 12],
        lr_mult=0.3,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner',
        miner_margin=0.1,
        faiss_gpu=False,
    )

    checkpoint_cb = ModelCheckpoint(
        monitor='cvusa/R1',
        filename=f'{args.backbone}_{args.aggregator}_cvusa_' + 'epoch({epoch:02d})_R1[{cvusa/R1:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max',
    )

    trainer = pl.Trainer(
        accelerator='cpu',
        devices=1,
        default_root_dir=str(Path(args.save_dir)),
        precision=32,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=0,
        log_every_n_steps=10,
        fast_dev_run=args.fast_dev_run,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == '__main__':
    main()

