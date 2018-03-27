# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:33:37 2017

@author: lyundemi
"""

import numpy as np
from scipy.optimize import differential_evolution, fmin_tnc
import itertools
import multiprocessing as mp
import random
import matplotlib.pyplot as plt


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

#plot 2D contour plot
def twodcontourplot(tadata_nm, tadata_timedelay, tadata_z_corr):
    nm, timedelay = np.linspace(tadata_nm.min(), tadata_nm.max(), 100), np.linspace(tadata_timedelay.min(), tadata_timedelay.max(), 100)    
    timedelayi, nmi = np.meshgrid(tadata_timedelay, tadata_nm)

    z_min = np.amin(np.amin(tadata_z_corr, axis = 1))
    z_max = np.amax(np.amax(tadata_z_corr, axis = 1))

    return [nmi, timedelayi, z_min, z_max]


#one gaussian
def gaussian(nm, a, x0, sigma):
    gaussian_array = a * np.exp(- ((nm - x0) ** 2.0) / (2 * (sigma ** 2.0))) 
    
    return gaussian_array


#monoexponential
def monoexp(t, tau):
    exp_array = np.exp(- (1.0/tau) * t)
    
    return exp_array
    

def data_matrix(nm_array, time_coeff_array, spectrum):
    data_matrix = np.empty((np.shape(nm_array)[0], np.shape(time_coeff_array)[0]))
    for i in range(np.shape(time_coeff_array)[0]):
        data_matrix[:, i] = time_coeff_array[i] * spectrum
    
    return data_matrix


def spectral_shift(start_nm, end_nm, time):
    step = float((end_nm - start_nm)) / (len(time))
    
    x0 = np.arange(start_nm, end_nm, step)

    return x0


def gaussian_shift(nm, a, x0_shiftarray, sigma):
    gaussian_matrix = np.empty((len(nm), len(x0_shiftarray)))
    for i in range(len(x0_shiftarray)):
        gaussian_matrix[:, i] = a * np.exp(- ((nm - x0_shiftarray[i]) ** 2.0) / (2 * (sigma ** 2.0)))
        
    return gaussian_matrix

def data_matrix_spectralshift(nm_array, time_coeff_array, spectrum_matrix):
    data_matrix = np.empty((np.shape(nm_array)[0], np.shape(time_coeff_array)[0]))
    for i in range(np.shape(time_coeff_array)[0]):
        data_matrix[:, i] = time_coeff_array[i] * spectrum_matrix[:, i]
    
    return data_matrix


    
time = np.arange(0, 5000, 1)
nm = np.arange(850, 1600, 1)

a1 = 1
x0_1 = 950
sigma_1 = 30
tau1 = 10


a2 = 0.3
x0_2 = 1300
sigma_2 = 100
tau2 = 5000

species_1 = gaussian(nm, a1, x0_1, sigma_1)
time_coeff_1 = monoexp(time, tau1)
data_matrix_1 = data_matrix(nm, time_coeff_1, species_1)

species_2 = gaussian(nm, a2, x0_2, sigma_2)
time_coeff_2 = monoexp(time, tau2)
data_matrix_2 = data_matrix(nm, time_coeff_2, species_2)

data_matrix = data_matrix_1 + data_matrix_2


x0_1_shift = spectral_shift(1200, 1300, time) 
species_1_matrix = gaussian_shift(nm, a1, x0_1_shift, sigma_1)
data_1_matrix_shift = data_matrix_spectralshift(nm, time_coeff_1, species_1_matrix)

plt.figure()
plt.plot(time, data_1_matrix_shift[find_nearest(nm, 960), :])

#data_matrix = data_1_matrix_shift + data_matrix_2


data_matrix_norm = np.empty_like(data_matrix)
for i in range(np.shape(data_matrix)[1]):
    data_matrix_norm[:, i] = data_matrix[:, i] / np.max(data_matrix[:, i])

data_matrix_norm = data_matrix
"""make 2d contour plot, use def two contourplot"""
plt.figure()
#plt.xlim(450,800)
plt.title('Two gaussians with spectral relaxation', fontsize = 16, fontweight = 'bold')
#plt.ylim(0,50)
plt.xlabel('Wavelength (nm)', fontsize = 16, fontweight = 'bold')
plt.ylabel('Time delay (ps)', fontsize = 16, fontweight = 'bold')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
nmi_9, timedelayi_9, z_min_9, z_max_9 = twodcontourplot(nm, time, data_matrix_norm)
plt.pcolormesh(nmi_9, timedelayi_9, data_matrix_norm, cmap = 'PiYG', vmin=z_min_9, vmax=z_max_9)
plt.colorbar()
plt.tight_layout(pad=0.25, h_pad=None, w_pad=None, rect=None)
