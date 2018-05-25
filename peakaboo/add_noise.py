#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 06:43:25 2018

@author: demiliu
"""
import numpy as np

#add noise to one array
def add_noise(nm_array, y_array, noise_coefficient):
    # Add noise
    np.random.seed(1800)
    y_noise = noise_coefficient * np.random.normal(size=nm_array.size)
    y_proc = y_array + y_noise
    
    return y_proc

#add noise a matrix
def noise_matrix(nm_array,data_matrix,noise_coefficient):
    num_array = np.shape(data_matrix)[1]
    noise_matx = np.empty_like(data_matrix)

    for i in range(num_array):
        data_array = data_matrix[:, i]
        noise_array = add_noise(nm_array, data_array, noise_coefficient)
        noise_matx[:, i] = noise_array
    
    return noise_matx

