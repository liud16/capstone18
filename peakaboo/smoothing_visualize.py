#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 10:27:03 2018

@author: demiliu
"""
import data_smoothing
from data_smoothing import earth_smooth_matrix
import numpy as np
import matplotli.pyplot as plt

def twodcontourplot(tadata_nm, tadata_timedelay, tadata_z_corr):
    """
    make contour plot
    
    Args:
        tadata_nm: wavelength array
        tadata_timedelay: time delay array
        tadata_z_corr: matrix of z values
        
    """
    nm, timedelay = np.linspace(tadata_nm.min(), tadata_nm.max(), 100), np.linspace(tadata_timedelay.min(), tadata_timedelay.max(), 100)    
    timedelayi, nmi = np.meshgrid(tadata_timedelay, tadata_nm)

    #find the maximum and minimum
    #these are used for color bar
    z_min = np.amin(np.amin(tadata_z_corr, axis = 1))
    z_max = np.amax(np.amax(tadata_z_corr, axis = 1))

    
    return [nmi, timedelayi, z_min, z_max]

def smoothing(nm, time, z):
    #smoothing data
    z_smooth = earth_smooth_matrix(nm, z)
    
    #contour plot of original data before smoothing
    original_contour = twodcontourplot(nm, time, z)
    nm_contour, time_contour, min_contour, max_contour = original_contour[0], original_contour[1], original_contour[2], original_contour[3]

    plt.figure()
    plt.title('Two gaussians with added noise', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Wavelength (nm)', fontsize = 16, fontweight = 'bold')
    plt.ylabel('Time delay (ps)', fontsize = 16, fontweight = 'bold')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.pcolormesh(nm_contour, time_contour, z, cmap = 'PiYG', vmin=min_contour, vmax=max_contour)
    plt.colorbar()
    plt.tight_layout(pad=0.25, h_pad=None, w_pad=None, rect=None)
    plt.show()
    
    smooth_contour = twodcontourplot(nm, time, z_smooth)    
    nm_contour, time_contour, min_contour, max_contour = smooth_contour[0], smooth_contour[1], smooth_contour[2], smooth_contour[3]
    
    plt.figure()
    plt.title('Two gaussians with added noise', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Wavelength (nm)', fontsize = 16, fontweight = 'bold')
    plt.ylabel('Time delay (ps)', fontsize = 16, fontweight = 'bold')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    
    plt.pcolormesh(nm_contour, time_contour, z_smooth, cmap = 'PiYG', vmin=min_contour, vmax=max_contour)
    plt.colorbar()
    plt.tight_layout(pad=0.25, h_pad=None, w_pad=None, rect=None)
    plt.show()

    return z_smooth