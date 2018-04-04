#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 19:53:57 2018

@author: demiliu
"""
import numpy as np
import matplotlib.pyplot as plt


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def loadtxt(filename):
    file = np.loadtxt(filename, delimiter='\t')
    nm = file[1:, 0]
    time = file[0, 1:]
    data = file[1:, 1:]
    
    return nm, time, data


def plot_decay(nm, time, select_nms, data):
    """plot the decay at chosen wavelengths,
    absolute intensity and normalized intensity"""
    
    select_nms_idx = [find_nearest(nm, x) for x in select_nms]
    num_nms = len(select_nms_idx)
    data_decay = np.empty((len(time), num_nms))
    data_decay_norm = np.empty_like(data_decay)
    
    plt.figure()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity (a.u.)')
    for i in range(num_nms):
        data_decay_i = data[select_nms_idx[i], :]
        data_decay[:, i] = data_decay_i
        plt.plot(time, data_decay_i, label = select_nms[i])
        
    plt.legend()

    plt.figure()
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Norm. intensity (a.u.)')    
    for i in range(num_nms):
        data_decay_i = data[select_nms_idx[i], :]
        data_decay_i_max = np.max(data_decay_i)
        data_decay_i_norm = data_decay_i / data_decay_i_max
        data_decay_norm[:, i] = data_decay_i_norm
                
        plt.plot(time, data_decay_i_norm, label = select_nms[i])
        
    plt.legend()
    
    return data_decay, data_decay_norm

#load data
nm, time, data = loadtxt('threegaussian_spectralshift_widthchange_posonly.txt')

#select time points to plot
nms = [950, 1000, 1500]

#plot 1d time-slice 
timeslice_matx = plot_decay(nm, time, nms, data)
   
        
    