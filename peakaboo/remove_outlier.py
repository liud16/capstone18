#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 14:12:45 2018

@author: demiliu
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression


def id_outliers_replacewith_interp(x_array, data, m, win_len):
    reshape_x_array = []
    reshape_data = []
    quotient_array = np.empty(len(data))
    remainder_array = np.empty(len(data))
    quotient_array[0] = 0
    remainder_array[0] = 0  
    #print divmod(len(data), win_len)   
    quotient_max = divmod(len(data), win_len)[0]
    print (quotient_max)
    #quotient_array_new = []
    data_idx = np.arange(0, len(data), 1)
    for i in range(1, len(data_idx)):
        
        quotient = divmod(data_idx[i], win_len)[0]
        quotient_array[i] = quotient
        remainder = divmod(data_idx[i], win_len)[1]
        remainder_array[i] = remainder
        
        if quotient != quotient_array[i-1]:
            newslice = data[i - win_len: i]
            newslice_x = x_array[i - win_len: i]
            #print newslice
            reshape_data.append(newslice)
            reshape_x_array.append(newslice_x)
    
        else:
            pass
    quotient_max_idx = np.where(quotient_array == quotient_max)
    #print quotient_max_idx
    reshape_data.append(data[quotient_max_idx[0]])
    reshape_x_array.append(x_array[quotient_max_idx[0]])
    #print reshape_data
    reshape_data_shape = np.shape(reshape_data)[0]
    #print reshape_data_shape
    def id_outliers_and_delete(d,x, m):
        d_mean = np.mean(d)  
        d_stdev = np.std(d)
        new_d = np.empty_like(d)    
        
        for i in range(len(d)):
            d_pt = d[i]
          
            if abs(d_pt - d_mean) > m * d_stdev and x[i] != x_array[0] and x[i] != x_array[len(x_array) - 1]:
                new_d[i] = 1
            else:
                new_d[i] = 0
    
        outlier_idx = np.nonzero(new_d)[0]
        d_delete = np.delete(d, outlier_idx)
        x_delete = np.delete(x, outlier_idx)
        
        #print data2[outlier_idx]
        return x_delete, d_delete
    
    new_x_array = []
    new_data = []
    for i in range(reshape_data_shape):
        new_data.append(id_outliers_and_delete(reshape_data[i],reshape_x_array[i], 1)[1])#(id_outliers_replacewith_mean(reshape_data[i], m))
        new_x_array.append(id_outliers_and_delete(reshape_data[i],reshape_x_array[i],1)[0])
    new_data_flat = np.concatenate(new_data[:-1]).ravel().tolist()#.flatten()
    new_x_array_flat = np.concatenate(new_x_array[:-1]).ravel().tolist()#.flatten()
    new_data_final = np.concatenate((new_data_flat, new_data[reshape_data_shape - 1]))
    new_x_array_final = np.concatenate((new_x_array_flat, new_x_array[reshape_data_shape - 1]))
    
    new_data_final_interp = np.interp(x_array, new_x_array_final, new_data_final)    
    
    return new_data_final_interp


def isotonic(x, y):
    ir = IsotonicRegression()
    y_ = ir.fit_transform(x, y)
    
    return y_