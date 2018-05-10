#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 09:36:40 2018

@author: demiliu
"""
import peakutils
import numpy as np
from peakutils.plot import plot as pplot
import matplotlib.pyplot as plt
from pyearth import Earth


def loaddata(data_filename):
    """load matrix data"""
    data = np.genfromtxt(data_filename + '.txt', delimiter='\t')
    data_nm = data[1:,0]    #wavelength in nm
    data_time = data[0,1:]
    data_z = data[1:, 1:]

    return data_nm, data_time, data_z


def Earth_Smoothing(nm_array, y_array):        
    """
    ============================================
     Plotting derivatives of simple sine function
    ============================================

     A simple example plotting a fit of the sine function
    and the derivatives computed by Earth.
    
    Notes
    -----   
    generates a denoise curve from the TA data
    Parameters
    ----------
        nm_array: wavelength array
        timedelay: time delay array
        noise_coefficient: the noise coefficients that user want to generate
    Returns
    -------
        a smoothing curve from the original noise curve   
    """

    
   # Fit an Earth model
    model = Earth(smooth=True)
    model.fit(nm_array, y_array)

   # Print the model
    #print(model.trace())
    #print(model.summary())

   # Get the predicted values and derivatives
    y_hat = model.predict(nm_array)
    
    return  y_hat


def smooth_matrix(nm_array,noise_matrix):
    num_array = np.shape(noise_matrix)[1]
    smooth_matx = np.empty_like(noise_matrix)
    for i in range(num_array):
        noise_array = noise_matrix[:, i]
        smooth_array = Earth_Smoothing(nm_array, noise_array)
        smooth_matx[:, i] = smooth_array
    
    return smooth_matx


matx_filename = '20180418_twogaussian_spectralshfit_0.05noise'
datanm, datatime, dataz_matx = loaddata(matx_filename)
earthz_matx = smooth_matrix(datanm, dataz_matx)

out2 = np.empty((len(datanm)+1, len(datatime)+1))
out2[1:, 0] = datanm
out2[0, 1:] = datatime
out2[1:, 1:] = earthz_matx

np.savetxt(matx_filename + '_earthsmooth.txt', out2, fmt='%.2e', delimiter='\t')