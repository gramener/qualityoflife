#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 13:02:12 2020

@author: gramener
"""
import numpy as np
import os
from os import listdir
from PIL import Image

Img_dir = '/media/gramener/HDD/For GPU Lappy/US_QOL_County/results_single'
test_img = '/media/gramener/HDD/For GPU Lappy/US_QOL_County/test_images/'

test_folder_list = listdir(test_img)
single_list = listdir(Img_dir)
# cluster_single_list = [x.split("_")[0] for x in single_list]

metrics_list = ['agriculture_land_bin','pop_density','employment','literacy_rate']

for folder in test_folder_list:
    if folder in metrics_list:
        img_file = listdir(test_img+'/'+folder)
        for img in img_file:
            image_att = Image.open(test_img+'/'+folder+'/'+img)
            directory = img.split("-")[0]
            cluster = directory.split("_")[0]
            # print(img)
            for folder_single in single_list:
                # print(folder_single.split("_")[0])
                
                if cluster == folder_single.split("_")[0]:
                    image_att.save(Img_dir+'/'+folder_single+'/'+img)
            # path = os.path.join(Img_dir,directory)
            # os.mkdir(path)            
    