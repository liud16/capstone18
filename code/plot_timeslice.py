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

def plot_timeslice(nm, time, select_times, data):
    select_times_idx = [find_nearest(time, x) for x in select_times]
    num_times = len(select_times_idx)
    data_timeslice = np.empty((len(nm), num_times))
    
    plt.figure()
    for i in range(num_times):
        data_timeslice_i = data[:, select_times_idx[i]]
        data_timeslice[:, i] = data_timeslice_i
        plt.plot(nm, data_timeslice_i)
    
    return data_timeslice

#load data
nm, time, data = loadtxt('onegaussian_spectralshift_widthchange.txt')

#select time points to plot
times = [1, 200]

#plot 1d time-slice 
timeslice_matx = plot_timeslice(nm, time, times, data)

    
    
    
        
    