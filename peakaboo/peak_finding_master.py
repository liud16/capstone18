#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 08:59:44 2018

@author: demiliu
"""

from peak_character import peak_matrix

def findpeaks(nm, time, z):
    
    peaks_ok = 'N'
    while peaks_ok == 'N':
        
        print ('Please enter the parameters for finding peaks')
        default_parameter = input('Default parameters? Y/N ')
        
        assert default_parameter == str('Y') or ('N'), ('Response to "default parameters?" can only be Y or N.')
        
        if default_parameter == 'Y':
            threshold = 0
            min_dist = 0
            idx, height, fwhm = peak_matrix(nm, z, threshold, min_dist)
            print (idx)
            print (height)
            print (fwhm)
            peaks_ok = input('Are you satisfied with peak-finding? Y/N ')            
        
        elif default_parameter == 'N':
            threshold = int(input('Threshold (0 to 100): '))
            min_dist = int(input('Minimum distance between peaks (integer): '))
            idx, height, fwhm = peak_matrix(nm, z, threshold, min_dist)
            print (idx)
            print (height)
            print (fwhm)
            peaks_ok = input('Are you satisfied with peak-finding? Y/N ')            
            
        else:
            print ('Please enter Y or N only for "Default parameters?".')
            peaks_ok = 'N'
        
        
        return idx, height, fwhm
        
        