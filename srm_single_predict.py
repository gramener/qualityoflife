import torch
from torchvision import transforms

import pdb
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path

from srm_single_model import get_model
from srm_single_dataset import get_data

device = torch.device('cuda')
metrics = ['$', 'H20', 'TOI', '^', 'COK', 'DRT', 'POP', 'LS', 'AG']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224
CP = 'single_d121_ur_phase3_bal_4_final.pt'


def load_model():
    wrapper, db = get_model(), get_data()
    model = wrapper.model
    mapper = db.targets
    model.load_state_dict(torch.load(CP))
    model.eval()

    return model, mapper


def load_image(file):
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE + 32),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    img = Image.open(file).convert('RGB')
    img_tfm = transform(img)
    return img_tfm.unsqueeze_(0)


def main():
    model, mapper = load_model()
    valid = pd.read_csv('data/qol/valid.csv')
    results = []

    for _,row in valid.iterrows():
        try:
            img_tfm = load_image(f'data/qol/sentinel_1/{row.cluster}.png')
        except:
            continue    
        out = model(img_tfm)
        r = {'cluster': row.cluster}
        # print(out[0].detach().numpy())
        for i,k in enumerate(mapper):
            _,score = out[i].topk(2)
            score.squeeze_()

            r[f'{k}_0'] = mapper[k]['classes'][score[0]]
            r[f'{k}_1'] = mapper[k]['classes'][score[1]]
            # print((k, row[k], mapper[k]['classes'][score[0]], mapper[k]['classes'][score[1]]))

        results.append(r)

    results = pd.DataFrame(results)
    results.to_csv('data/qol/results.csv', index=False)


if __name__ == "__main__":
    main()
