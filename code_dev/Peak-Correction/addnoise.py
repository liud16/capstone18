#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 09:21:22 2018

@author: demiliu
"""

import peakutils
import numpy as np
from peakutils.plot import plot as pplot
import matplotlib.pyplot as plt
from pyearth import Earth

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


def loaddata(data_filename):
    """load matrix data"""
    data = np.genfromtxt(data_filename + '.txt', delimiter='\t')
    data_nm = data[1:,0]    #wavelength in nm
    data_time = data[0,1:]
    data_z = data[1:, 1:]

    return data_nm, data_time, data_z


def loaddata_timeslice(data_filename):
    """load array data"""
    data = np.genfromtxt(data_filename + '.txt', delimiter='\t')
    data_nm = data[:,0]    #wavelength in nm
    data_z_array = data[:, 1]

    return data_nm, data_z_array

def add_noise(nm_array, y_array, noise_coefficient):
    # Add noise
    np.random.seed(1800)
    y_noise = noise_coefficient * np.random.normal(size=nm_array.size)
    y_proc = y_array + y_noise
    
    return y_proc

def add_noise_matx(nm_array,data_matrix,noise_coefficient):
    num_array = np.shape(data_matrix)[1]
    noise_matx = np.empty_like(data_matrix)

    for i in range(num_array):
        data_array = data_matrix[:, i]
        noise_array = add_noise(nm_array, data_array, noise_coefficient)
        noise_matx[:, i] = noise_array
    
    return noise_matx

array_filename = 'twogaussian_array'
datanm, dataz = loaddata_timeslice(array_filename)

noisyz = add_noise(datanm, dataz, 0.05)
plt.figure()
plt.plot(datanm, noisyz, label = 'adding noise')
plt.plot(datanm, dataz, label = 'original data')
plt.legend()

out1 = np.empty((len(datanm), 2))
out1[:, 0] = datanm
out1[:, 1] = noisyz
np.savetxt(array_filename + '_0.05noise.txt', out1, fmt='%.2e', delimiter='\t')

matx_filename = '20180418_twogaussian_spectralshfit'
datanm, datatime, datazmatx = loaddata(matx_filename)

noisyzmatx = add_noise_matx(datanm, datazmatx, 0.05)


out2 = np.empty((len(datanm)+1, len(datatime)+1))
out2[1:, 0] = datanm
out2[0, 1:] = datatime
out2[1:, 1:] = noisyzmatx

np.savetxt(matx_filename + '_0.05noise.txt', out2, fmt='%.2e', delimiter='\t')