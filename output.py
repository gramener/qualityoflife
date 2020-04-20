import torch
from torchvision import transforms

import pdb
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from os import listdir

from srm_single_model import get_model
from srm_single_dataset import get_data

from visualize import split, merge

device = torch.device('cuda')
# metrics = ['AGR']
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

IMG_DIR = Path('./data/qol/sentinel')
CSV_DIR = Path('./test_images')
metrics_list = ['agriculture_land_bin','pop_density','employment','literacy_rate']

COLOR = {
    'GREEN': (52,168,82),
    'LIGHT_GREEN': (12,234,8),
    'YELLOW': (236,251,7),
    'ORANGE': (251,188,7),
    'RED': (234,67,53)
}

w2s = {
    'a:poorest': COLOR['RED'] + (128,),
    'b:poorer': COLOR['ORANGE'] + (128,),
    'c:middle': COLOR['YELLOW'] + (128,),
    'd:richer': COLOR['LIGHT_GREEN'] + (128,),
    'e:richest': COLOR['GREEN'] + (128,)
}

p2s = {
    'high': COLOR['RED'] + (128,),
    'mid': COLOR['YELLOW'] + (128,),
    'low': COLOR['LIGHT_GREEN'] + (128,)
}

d2s = {
    'high': COLOR['RED'] + (128,),
    'mid': COLOR['YELLOW'] + (128,),
    'low': COLOR['GREEN'] + (128,)
}

t2s = {
    'flush to septic tank': COLOR['GREEN'] + (128,),
    'flush to pit latrine': COLOR['GREEN'] + (128,),
    'flush to piped sewer system': COLOR['GREEN'] + (128,),
    'flush to somewhere else': COLOR['GREEN'] + (128,),
    "flush, don't know where": COLOR['GREEN'] + (128,),
    "flush": COLOR['GREEN'] + (128,),
    'dry toilet': COLOR['YELLOW'] + (128,),
    'composting toilet': COLOR['YELLOW'] + (128,),
    "other": COLOR['YELLOW'] + (128,),
    "don't know where": COLOR['YELLOW'] + (128,),
    'pit latrine with slab': COLOR['ORANGE'] + (128,),
    'pit latrine without slab/open pit': COLOR['ORANGE'] + (128,),
    'ventilated improved pit latrine (vip)': COLOR['ORANGE'] + (128,),
    "no facility/bush/field": COLOR['RED'] + (128,),
}

def load_model(metric):
    CP = 'single_d121_ur_phase3_bal_4_'+f'{metric}.pt'
    wrapper, db = get_model(metric), get_data(metric)
    model = wrapper.model
    mapper = db.targets
    model.load_state_dict(torch.load(CP))
    model.eval()

    return model, mapper


transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE + 32),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

def process_one(img, model, mapper):
    img_tfm = transform(img); img_tfm.unsqueeze_(0)
    out = model(img_tfm)

    t = dict()
    for i,k in enumerate(mapper):
        index = out[i].argmax()
        t[k] = mapper[k]['classes'][index]
    return t

def single_predict(filename, ntiles, metric, mapper, model, wrapper, **kwargs):
    img = Image.open(filename)
    d = split(img, ntiles)
    for k,v in d.items():
        v['score'] = process_one(v['img'],model,wrapper)
        # print(v)

    new_img = merge(d, ntiles, metric, mapper)
    # new_img.save('./test_images/bangalore-64km_toilet_type.png')

    return (img, new_img)

def main():
    for metric in metrics_list:
        dir_img = listdir(IMG_DIR)
        readCsv = pd.read_csv(CSV_DIR/"sentinel_main.csv")
        model, mapper = load_model(metric)
        for img in dir_img:
            i = int(img.split('.')[0])
            # print(i)
            place_df = readCsv.loc[readCsv['cluster'] == i]['region']
            plDF_index = place_df.index[0]
            place = place_df[plDF_index]
            print(place)
            filename = IMG_DIR/img
            # print(filename)
            # _, wealth_img = single_predict(filename, 4, 'wealth', w2s)
            # _, toilet_type_img = single_predict(filename, 4, 'toilet_type', t2s)
            _, pop_density_img = single_predict(filename, 4, metric, p2s, model, mapper)
            # _, drought_img = single_predict(filename, 4, 'drought', d2s)
        
            # wealth_img.save('./test_images/Mumbai_1-wealth.png')
            # toilet_type_img.save('./test_images/Mumbai_1-toilet_type.png')
            pop_density_img.save('./test_images/'+f'{metric}/'+f'{i}_{place}-{metric}.png')
            # drought_img.save('./test_images/Mumbai_1-drought.png')


if __name__ == "__main__":
    main()
