import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloaders.CVUSADataset import CVUSAPairedDataset, CVUSAValDataset, DEFAULT_ROOT
from dataloaders.GSVCitiesDataloader import IMAGENET_MEAN_STD


class CVUSADataModule(pl.LightningDataModule):
    def __init__(
        self,
        root=DEFAULT_ROOT,
        train_csv=r'E:\Transgeo VPR\TransGeo2022\data\custom_splits\train_80.csv',
        test_csv=r'E:\Transgeo VPR\TransGeo2022\data\custom_splits\test_20.csv',
        batch_size=16,
        image_size=(320, 320),
        num_workers=0,
        mean_std=IMAGENET_MEAN_STD,
        show_data_stats=True,
    ):
        super().__init__()
        self.root = root
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.show_data_stats = show_data_stats
        self.val_set_names = ['cvusa']
        self.save_hyperparameters()

        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=2, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': False,
            'shuffle': True,
        }

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': max(self.num_workers // 2, 0),
            'drop_last': False,
            'pin_memory': False,
            'shuffle': False,
        }

    def setup(self, stage=None):
        if stage in (None, 'fit'):
            self.train_dataset = CVUSAPairedDataset(
                csv_path=self.train_csv,
                root=self.root,
                transform=self.train_transform,
            )
            self.val_datasets = [
                CVUSAValDataset(
                    csv_path=self.test_csv,
                    root=self.root,
                    transform=self.valid_transform,
                )
            ]
            if self.show_data_stats:
                self.print_stats()

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        return [
            DataLoader(dataset=val_dataset, **self.valid_loader_config)
            for val_dataset in self.val_datasets
        ]

    def print_stats(self):
        print()
        print('CVUSA Training Dataset')
        print(f'  root: {self.root}')
        print(f'  train csv: {self.train_csv}')
        print(f'  test csv: {self.test_csv}')
        print(f'  # train pairs: {len(self.train_dataset)}')
        print(f'  # test pairs: {self.val_datasets[0].num_references}')
        print(f'  batch size: {self.batch_size}')
        print(f'  image size: {self.image_size}')
