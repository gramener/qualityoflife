#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 16:47:12 2020

@author: gramener
"""

import numpy as np
import os
from os import listdir
from PIL import Image
import pandas as pd

from visualize import split, merge, merge_final

csv_dir = '/media/gramener/HDD/For GPU Lappy/US_QOL_County/data/qol/sentinel.csv'
Img_dir = '/media/gramener/HDD/For GPU Lappy/US_QOL_County/results_single'
test_img = '/media/gramener/HDD/For GPU Lappy/US_QOL_County/data/qol/sentinel'
list_images = listdir(Img_dir)

sentinel_csv = pd.read_csv(csv_dir)

COLOR = {
    'GREEN': (52,168,82),
    'LIGHT_GREEN': (12,234,8),
    'YELLOW': (236,251,7),
    'ORANGE': (251,188,7),
    'RED': (234,67,53)
}

p2s = {
    'high': COLOR['RED'] + (128,),
    'mid': COLOR['YELLOW'] + (128,),
    'low': COLOR['LIGHT_GREEN'] + (128,)
}

create_dict_rural = {}
create_dict_rural['agriculture_land_bin'] = 0.4
create_dict_rural['literacy_rate'] = 0.2
create_dict_rural['employment'] = 0.2
create_dict_rural['pop_density'] = 0.2

create_dict_urban =  {}
create_dict_urban['agriculture_land_bin'] = 0.1
create_dict_urban['literacy_rate'] = 0.3
create_dict_urban['employment'] = 0.3
create_dict_urban['pop_density'] = 0.3
 

print(list_images)
# weights = [1,1,1,1,-1,1]
labels_1 = ["low","mid","high"]
for county_fold in list_images:
    img_fold_dir = Img_dir+'/'+county_fold
    imgs_dir = listdir(img_fold_dir)
    scores_all = []
    score = 0
    i=0
    for img in imgs_dir:
        attribute = (img.split("-")[1].split(".")[0])
        county_code = int(img.split("-")[0].split("_")[0])
        ranking = list(sentinel_csv.loc[sentinel_csv["cluster"] == county_code]["ranking_bin"])[0]
        print(ranking, county_fold)
        if(ranking == "Rural"):
            weights = create_dict_rural[attribute]
        else:
            weights = create_dict_urban[attribute]
        score_list=[]
        # img_orig = Image.open(Img_dir+'/'+county_fold+'.png').convert('RGB')
        image_county = Image.open(img_fold_dir+'/'+img).convert('RGB')
        # diff_img = img_orig-image_county
        # Image.fromarray(diff_img).show()
        d = split(image_county,4)
        for k,v in d.items(): 
            im_c = v['img'].load()
            pixel_v = im_c[64,64]
            if(pixel_v[0] >= 100):
                if(pixel_v[1] >= 100):
                    score = 2
                else:
                    score = 3
            elif(pixel_v[1] >= 100):
                    score = 1
            score_list.append(score)
        # print(attribute,county_fold)    
        # print(score_list)
        score_list = [x*weights for x in score_list]
        i=i+1
        scores_all.append(score_list)
    total_sum = np.sum(scores_all,axis=0)
    print(total_sum)
    total_bins = list(pd.cut(total_sum,[0.9,1.6,2.3,3.1],labels=labels_1))
    orig_img = Image.open(test_img+'/'+str(county_code)+'.png')
    split_d = split(orig_img,4)
    j=0
    for k,v in split_d.items():
        v['score'] = total_bins[j]
        j=j+1
    new_img = merge_final(split_d, 4, p2s)
    new_img.save('./results_single/'+f'{county_fold}/'+f'{county_fold}-qol.png')
        
        