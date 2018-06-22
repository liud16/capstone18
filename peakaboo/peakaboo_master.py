# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import data_smoothing
from peak_finding_userinter import findpeaks
import smoothing_visualize
import peak_classify
import feature_visualizer
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """main function to run functions based on user interaction"""
    
    print ('--- Peak Detection in Transient Absorption Spectra ---')
    print ('Hello! Welcome to Peakaboo!')
    
    #asks the file-type
    print ('Please enter filetype (.csv or .txt)')
    filetype = input('Filetype: ')

    print ('Please enter filename (without extension)')
    filename = input('Filename: ')
    
    print ('Please choose cut-on wavelength (only number, in nm)')
    cuton_nm = input('Cut-on wavelength: ')

    print ('Please choose cut-off wavelength (only number, in nm)')
    cutoff_nm = input('Cut-off wavelength: ')

    print ('Please enter true time zero (only number, in ps)')
    timezero = input('Time zero at: ')
    
    if filetype == '.txt':
        nm, time, z = data_smoothing.load_data(filename+'.txt', cuton_nm, cutoff_nm, timezero)

    elif filetype == '.csv':
        nm, time, z = data_smoothing.load_data_csv(filename+'.csv', cuton_nm, cutoff_nm, timezero)
    
    #smooth data and visualize
    z_smooth = smoothing_visualize.smoothing(nm, time, z)

    print ('Next step is to find peaks in each time-slice.')
    idx, height, fwhm = findpeaks(nm, time, z_smooth)
    
    
    
    
