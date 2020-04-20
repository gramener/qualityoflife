#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 18:02:44 2020

@author: gramener
"""


import pandas as pd




sentinel_csv = pd.read_csv("sentinel.csv")

county_urban_rural = pd.read_excel("NCHSURCodes2013.xlsx")

# metro_col = county_urban_rural["2013 code"]
county_code = list(county_urban_rural["FIPS code"])

county_sentinel = list(sentinel_csv["cluster"])
i=0
l=0
ranking_bin=[]
# for county in county_sentinel:
#     list_county.append(county)

for county in county_sentinel:
    if county in county_code:
        i=i+1
        ranking = int(county_urban_rural.loc[county_urban_rural["FIPS code"] == county]["2013 code"])
        if(ranking == 6):
            # print(county)
            ranking_bin.append("Rural")
            l= l+1
        else:
            ranking_bin.append("Urban")
sentinel_csv["ranking_bin"] = ranking_bin
sentinel_csv.to_csv("sentinel.csv")            
print(i,l)