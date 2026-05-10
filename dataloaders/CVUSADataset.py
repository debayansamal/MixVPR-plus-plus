from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


DEFAULT_ROOT = Path(r'E:\CVUSA Dataset\Dataset_CVUSA')


class CVUSAPathResolver:
    def __init__(self, root=DEFAULT_ROOT):
        self.root = Path(root)

    def resolve(self, csv_path):
        path = Path(str(csv_path).replace('\\', '/'))
        direct_path = self.root / path
        if direct_path.exists():
            return direct_path

        folder = path.parts[0] if len(path.parts) > 1 else ''
        stem = path.stem
        suffix = path.suffix
        numeric_stem = stem.replace('input', '') if stem.startswith('input') else stem

        candidates = [
            self.root / folder / f'{numeric_stem}{suffix}',
            self.root / folder / f'{numeric_stem}.jpg',
            self.root / folder / f'{numeric_stem}.png',
            self.root / folder / f'input{numeric_stem}{suffix}',
            self.root / folder / f'input{numeric_stem}.jpg',
            self.root / folder / f'input{numeric_stem}.png',
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise FileNotFoundError(f'Could not resolve CVUSA path "{csv_path}" under "{self.root}"')


def read_cvusa_csv(csv_path, max_pairs=None):
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 2:
        raise ValueError(f'CVUSA csv must contain at least satellite and ground columns: {csv_path}')
    df = df.iloc[:, :2].copy()
    df.columns = ['satellite', 'ground']
    if max_pairs is not None:
        df = df.head(max_pairs).copy()
    return df


class CVUSAPairedDataset(Dataset):
    def __init__(self, csv_path, root=DEFAULT_ROOT, transform=None, max_pairs=None):
        self.csv_path = Path(csv_path)
        self.root = Path(root)
        self.transform = transform
        self.dataframe = read_cvusa_csv(self.csv_path, max_pairs=max_pairs)
        self.resolver = CVUSAPathResolver(self.root)
        self.total_nb_images = len(self.dataframe) * 2

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        sat = self._load_image(row['satellite'])
        ground = self._load_image(row['ground'])
        images = torch.stack([sat, ground])
        labels = torch.tensor(index).repeat(2)
        return images, labels

    def _load_image(self, csv_path):
        image = Image.open(self.resolver.resolve(csv_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image


class CVUSAValDataset(Dataset):
    def __init__(self, csv_path, root=DEFAULT_ROOT, transform=None, max_pairs=None):
        self.csv_path = Path(csv_path)
        self.root = Path(root)
        self.transform = transform
        self.dataframe = read_cvusa_csv(self.csv_path, max_pairs=max_pairs)
        self.resolver = CVUSAPathResolver(self.root)
        self.num_references = len(self.dataframe)
        self.pIdx = [[idx] for idx in range(self.num_references)]

        self.images = list(self.dataframe['satellite']) + list(self.dataframe['ground'])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = Image.open(self.resolver.resolve(self.images[index])).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, index


