import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms

import numpy as np
import pandas as pd
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import os.path
from os import path

from utils import read_df

CWD           = Path.cwd()
print(CWD)
DATA_PATH     = Path(CWD/'data'/'qol')
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

IMAGE_SIZE = 128
BATCH_SIZE = 64

def match_file2image(imdir, metafile):
    ims = set(int(p.stem) for p in imdir.glob('*.png') if not p.stem.endswith('mask'))
    metainfo = read_df(metafile)
    missing = [row['cluster'] for (_,row) in metainfo.iterrows()
                              if row['cluster'] not in ims]

    metainfo[~metainfo.cluster.isin(missing)].to_csv(metafile, index=False)


class SentinelDataset(Dataset):
    def __init__(self, metric, imdir, metafile, train=True, transform=None, filter=None):
        super().__init__()
        self.imdir = imdir
        self.metafile = metafile
        self.train = train
        self.transform = transform
        self.filter = filter

        self.cols = [metric]

        metainfo = read_df(metafile)
        # TODO: Only consider RURAL AREAS
        # metainfo = metainfo[metainfo.uor == 'R']
        metainfo = metainfo[['cluster'] + self.cols]

        # if self.filter:
        #     metainfo = metainfo[metainfo[target] == filter]

        self.targets = dict()
        for col in self.cols:
            # print(col)
            unique_classes = sorted(metainfo[col].unique())
            weights = {k:1/v for k,v in metainfo[col].value_counts().to_dict().items()}

            self.targets[col] = {
                'classes': unique_classes,
                'o2i': {o:i for i,o in enumerate(unique_classes)},
                'i2p': {i:weights[o] for i,o in enumerate(unique_classes)}
            }

        self.data = self.split(metainfo, 0.1, 42)

    def split(self, data, test_size, random_state):
        train, valid = train_test_split(data, test_size=test_size, random_state=random_state)
        train.to_csv('data/qol/train.csv', index=False)
        valid.to_csv('data/qol/valid.csv', index=False)
        return train if self.train else valid

    def get_y(self, idx, col):
        row = self.data.iloc[idx]

        return self.targets[col]['o2i'][row[col]]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        fake = 0
        labels = dict()
        for col in self.cols:
            labels[col] = self.targets[col]['o2i'][row[col]]
        value_row = int(row['cluster'])   
        if path.exists(self.imdir/f"{value_row}.png"):
            im = Image.open(self.imdir/f"{value_row}.png").convert('RGB')
        elif path.exists(self.imdir/f"{value_row-1}.png"):
            im = Image.open(self.imdir/f"{value_row-1}.png").convert('RGB')
        elif path.exists(self.imdir/f"{value_row+1}.png"):
            im = Image.open(self.imdir/f"{value_row+1}.png").convert('RGB')
        elif path.exists(self.imdir/f"{value_row+2}.png"):
            im = Image.open(self.imdir/f"{value_row+2}.png").convert('RGB')
        else:
            im = Image.open(self.imdir/"10001.png").convert('RGB')
            fake = fake+1

        # cluster, label = row[self.imid], self.o2i[row[self.target]]
        # im = Image.open(self.imdir/f"{row['cluster']}.png").convert('RGB')
        # im1 = Image.open(self.imdir/f"{row['cluster']}.png").convert('RGB')
        # im2 = Image.open(self.imdir/f"{row['cluster']}-mask.png").convert('RGB')
        # print(fake)
        return self.transform(im), labels

class DataBunch:
    def __init__(self, metric,img_sz, bs):
        transform = {
            'train': transforms.Compose([
                transforms.Resize(img_sz + 32),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop(img_sz),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ]),
            'valid': transforms.Compose([
                transforms.Resize(img_sz + 32),
                transforms.CenterCrop(img_sz),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
        }

        self.train_ds = SentinelDataset(metric,DATA_PATH/'sentinel', DATA_PATH/'sentinel.csv', True,  transform['train'], None)
        self.valid_ds = SentinelDataset(metric,DATA_PATH/'sentinel', DATA_PATH/'sentinel.csv', False, transform['valid'], None)

        o2i     = self.train_ds.targets[metric]['o2i']
        weights = self.train_ds.targets[metric]['i2p']
        sampler_weights = torch.tensor([weights[self.train_ds.get_y(i,metric)] for i in range(len(self.train_ds))])
        sampler = WeightedRandomSampler(sampler_weights.type(torch.DoubleTensor), len(sampler_weights))

        self.train_dl = DataLoader(self.train_ds, batch_size=bs, sampler=sampler)
        self.valid_dl = DataLoader(self.valid_ds, batch_size=bs*2, shuffle=False)
        self.targets = self.train_ds.targets

def get_data(metric, img_sz=IMAGE_SIZE, bs=BATCH_SIZE):
    # print(metric)
    return DataBunch(metric,img_sz, bs)

# if __name__ == '__main__':
#     # match_file2image(DATA_PATH/'sentinel', DATA_PATH/'sentinel.csv')
#     db = get_data()
