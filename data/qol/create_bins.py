# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 13:35:34 2020

@author: nandakishore.m
"""
import pandas as pd
import numpy as np

def create_bin(g_vec, bins_1):
    # if(bins == 3):
    #     labels = ["low","mid","high"]
    # elif(bins == 5):
    labels_1 = ["low","mid","high"]
    # g_vec_bins= pd.cut(np.array(g_vec),[-1,5000,20000,4000000],labels=labels_1)  #Employment
    # g_vec_bins= pd.cut(np.array(g_vec),[-1,1000,5000,3000000],labels=labels_1)   #Population
    g_vec_bins= pd.cut(np.array(g_vec),[-1,310,900,90000],labels=labels_1)       #Death
    # g_vec_bins= pd.cut(np.array(g_vec),[-1,15000,100000,800000],labels=labels_1)         #Agriculture
    print(g_vec_bins)
    return g_vec_bins

def write_to_csv(csv,input_cols,output_cols,bins):
    for i in range(len(input_cols)):
        input_vec = csv[input_cols[i]]
        input_vec.fillna( 0, inplace=True)
        output_vec_bin = create_bin(input_vec, bins[i])
        output_bin=list(output_vec_bin)
        csv[output_cols[i]] = output_bin
    return csv

def main():  
    input_cols = ["No of deaths"]
    output_cols = ["death_toll"]
    bins = [5]
    orig_csv = pd.read_csv('sentinel.csv')
    csv_bin = write_to_csv(orig_csv,input_cols,output_cols,bins)
    csv_bin.to_csv("sentinel.csv")
    
    
if __name__ == "__main__":
    orig_csv = pd.read_csv('sentinel_main.csv')
    input_cols = ["No of deaths"]
    mnk = (orig_csv[input_cols]).sort_values(by = ["No of deaths"])
    main()