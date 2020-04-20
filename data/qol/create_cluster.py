#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 21:08:49 2020

@author: gramener
"""

import pandas as pd
input_csv = pd.read_csv("uscities.csv")

list_of_clusters = []
for i in range (len(input_csv["city"])):
    list_of_clusters.append(i)
input_csv.insert(2,"cluster",list_of_clusters)
input_csv.to_csv("us_Cities.csv")
