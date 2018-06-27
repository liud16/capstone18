#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 10:00:11 2018

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

matx_filename = '20180418_twogaussian_spectralshfit_0.05noise_earthsmooth'
datanm, datatime, earthz_matx = loaddata(matx_filename)


datamatx_filename = '20180418_twogaussian_spectralshfit'
datanm, datatime, data_matx = loaddata(datamatx_filename)


"""Copy-pasted from peakutils source code: Peak detection algorithms."""
from scipy import optimize
from scipy.integrate import simps

eps = np.finfo(float).eps

def indexes(y, thres, min_dist):
    """Peak detection routine.

    Finds the numeric index of the peaks in *y* by taking its first order difference. By using
    *thres* and *min_dist* parameters, it is possible to reduce the number of
    detected peaks. *y* must be signed.

    Parameters
    ----------
    y : ndarray (signed)
        1D amplitude data to search for peaks.
    thres : float between [0., 1.]
        Normalized threshold. Only the peaks with amplitude higher than the
        threshold will be detected.
    min_dist : int
        Minimum distance between each detected peak. The peak with the highest
        amplitude is preferred to satisfy this constraint.

    Returns
    -------
    ndarray
        Array containing the numeric indexes of the peaks that were detected
    """
    if isinstance(y, np.ndarray) and np.issubdtype(y.dtype, np.unsignedinteger):
        raise ValueError("y must be signed")

    thres = thres * (np.max(y) - np.min(y)) + np.min(y)
    min_dist = int(min_dist)

    # compute first order difference
    dy = np.diff(y)

    # propagate left and right values successively to fill all plateau pixels (0-value)
    zeros,=np.where(dy == 0)
    
    # check if the singal is totally flat
    if len(zeros) == len(y) - 1:
        return np.array([])
    
    while len(zeros):
        # add pixels 2 by 2 to propagate left and right value onto the zero-value pixel
        zerosr = np.hstack([dy[1:], 0.])
        zerosl = np.hstack([0., dy[:-1]])

        # replace 0 with right value if non zero
        dy[zeros]=zerosr[zeros]
        zeros,=np.where(dy == 0)

        # replace 0 with left value if non zero
        dy[zeros]=zerosl[zeros]
        zeros,=np.where(dy == 0)

    # find the peaks by using the first order difference
    peaks = np.where((np.hstack([dy, 0.]) < 0.)
                     & (np.hstack([0., dy]) > 0.)
                     & (y > thres))[0]

    # handle multiple peaks, respecting the minimum distance
    if peaks.size > 1 and min_dist > 1:
        highest = peaks[np.argsort(y[peaks])][::-1]
        rem = np.ones(y.size, dtype=bool)
        rem[peaks] = False

        for peak in highest:
            if not rem[peak]:
                sl = slice(max(0, peak - min_dist), peak + min_dist + 1)
                rem[sl] = True
                rem[peak] = False

        peaks = np.arange(y.size)[~rem]

    return peaks


def findpeak(data_z_array, threshold, min_dist):
    """find peaks and return indices of the peaks"""    
    peak_indices = indexes(data_z_array, thres=threshold, min_dist=min_dist)
    
    return peak_indices


def peakchar(data_nm, data_z_array, peak_index):
    """find the peak width, and intensity"""
    num_peaks = len(peak_index)
    
    #array of peak height
    height = [data_z_array[idx] for idx in peak_index]
    
    #array of peak width
    half_height = [ht / 2 for ht in height]

    fwhm_idx_1 = np.empty_like(half_height)
    fwhm_idx_2 = np.empty_like(fwhm_idx_1)
    fwhm_nm_1 = np.empty_like(fwhm_idx_1)
    fwhm_nm_2 = np.empty_like(fwhm_idx_1)
    
    for i in range(num_peaks):
        #find the index and nmof the left side of the fwhm
        if i == 0:
            fwhm_idx_1[i] = find_nearest(data_z_array[0:peak_index[i]], half_height[i])
        else:
            fwhm_idx_1[i] = find_nearest(data_z_array[peak_index[i-1]:peak_index[i]], half_height[i]) + peak_index[i-1]

        fwhm_nm_1[i] = data_nm[int(fwhm_idx_1[i])]
        
        #find the index and nm of the right side of the fwhm   
        fwhm_idx_2[i] = find_nearest(data_z_array[peak_index[i]:], half_height[i]) + peak_index[i]

        fwhm_nm_2[i] = data_nm[int(fwhm_idx_2[i])]
    
    #find fwhm
    fwhm = fwhm_nm_2 - fwhm_nm_1

    return height, fwhm



def peak_matrix(nm_array,data_matrix,num_timeslice, threshold, mindist):
    """find peaks in a data matrix"""
    peak_idx_matx = np.zeros((num_timeslice,2))
    peak_height_matx = np.empty_like(peak_idx_matx)
    peak_fwhm_matx = np.empty_like(peak_height_matx)
    
    for i in range(num_timeslice):
        data_timeslice = data_matrix[:, i]
        
        peak_idx = findpeak(data_timeslice, threshold, mindist)
        
        
        peak_idx_matx[i, :] = peak_idx
        
        peak_height, peak_fwhm = peakchar(nm_array, data_timeslice, peak_idx)
        peak_height_matx[i, :], peak_fwhm_matx[i, :] = peak_height, peak_fwhm 
    
   
    return peak_idx_matx, peak_height_matx, peak_fwhm_matx



def peak_matrix_mod(nm_array,data_matrix,num_timeslice, threshold, mindist):
    """find peaks in a data matrix"""
    
    #find peaks in the first timeslice to get an idea
    #of how many peaks there maybe
    #this number is used to construct a matrix for
    #all peaks
    data_timeslice = data_matrix[:, 0]
    peak_idx = findpeak(data_timeslice, threshold, mindist)
    #entire data may have 2 more species    
    num_peaks_pos = len(peak_idx) + 2
    
    #find peak in each timeslice
    peak_idx_matx = np.zeros((num_timeslice,num_peaks_pos))
    peak_height_matx = np.empty_like(peak_idx_matx)
    peak_fwhm_matx = np.empty_like(peak_height_matx)
    
    for i in range(num_timeslice):
        data_timeslice = data_matrix[:, i]
        peak_idx = findpeak(data_timeslice, threshold, mindist)
        peak_idx_matx[i, :] = peak_idx
        
        peak_height, peak_fwhm = peakchar(nm_array, data_timeslice, peak_idx)
        peak_height_matx[i, :], peak_fwhm_matx[i, :] = peak_height, peak_fwhm 
    
    return peak_idx_matx, peak_height_matx, peak_fwhm_matx


num_timeslice = np.shape(data_matx)[1]
peak_idx_matx, peak_height_matx, peak_fwhm_matx = peak_matrix(datanm,data_matx,num_timeslice, 0.0, 300)

peak1_idx = peak_idx_matx[:, 0]
peak1_idx_gradcorr= np.copy(peak1_idx)
peak1_idx_gradient = np.gradient(peak1_idx)

num_wrongpeaks = 1
while num_wrongpeaks != 0:
    fix_peak_idx = []
    for i, gradient in enumerate(peak1_idx_gradient):
        if np.abs(gradient) >= 10: #threshold for difference in peak idx
            fix_peak_idx += [i]
        else:
            fix_peak_idx = fix_peak_idx

    num_wrongpeaks = len(fix_peak_idx)
    peak1_idx_gradcorr = np.copy(peak1_idx)   
    for i in range(num_wrongpeaks):
        if fix_peak_idx[i] == np.shape(peak1_idx)[0]-1:
            peak1_idx[fix_peak_idx[i]] = peak1_idx[fix_peak_idx[i]]
        else:
            peak1_idx[fix_peak_idx[i]+1] = peak1_idx[fix_peak_idx[i]]
    peak1_idx_gradient = np.gradient(peak1_idx)

plt.figure()
plt.plot(datatime, peak1_idx, 'o')

peakpos = [datanm[int(idx)] for idx in peak1_idx]
plt.figure()
plt.plot(datatime, peakpos, 'o')


""""""
def peakposition_corr(peak_idx_matx):
    
    peak1_idx = peak_idx_matx[:, 0]
    peak1_idx_gradient = np.gradient(peak1_idx)
    
    num_wrongpeaks = 1
    while num_wrongpeaks != 0:
        fix_peak_idx = []
        for i, gradient in enumerate(peak1_idx_gradient):
            if np.abs(gradient) >= 10: #threshold for difference in peak idx
                fix_peak_idx += [i]
            else:
                fix_peak_idx = fix_peak_idx
    
        num_wrongpeaks = len(fix_peak_idx)
        for i in range(num_wrongpeaks):
            if fix_peak_idx[i] == np.shape(peak1_idx)[0]-1:
                peak1_idx[fix_peak_idx[i]] = peak1_idx[fix_peak_idx[i]]
            else:
                peak1_idx[fix_peak_idx[i]+1] = peak1_idx[fix_peak_idx[i]]
        peak1_idx_gradient = np.gradient(peak1_idx)
    
#    plt.figure()
#    plt.plot(datatime, peak1_idx, 'o')
    
    peakpos = [datanm[int(idx)] for idx in peak1_idx]

    return peakpos    




peak1_height = peak_height_matx[:, 1]
#peak1_idx_ediff = np.ediff1d(peak1_idx)
peak1_height_gradient = np.ediff1d(peak1_height)


"""#height doesn't work yet
mean_height = 0.1
peak1_height_gradcorr = np.copy(peak1_height)
num_wrongheights = 1
while num_wrongheights != 0:
    fix_peak_height = []
    for i, gradient in enumerate(peak1_height_gradient):
        if np.abs(gradient) >= mean_height: #threshold for difference in peak idx
            fix_peak_height += [i]
            
        else:
            fix_peak_height = fix_peak_height
    
    num_wrongheights = len(fix_peak_height)
    peak1_height_gradcorr = np.copy(peak1_height)   
    for i in range(num_wrongpeaks):
        if fix_peak_height[i] == np.shape(peak1_height)[0]-1:
            peak1_height[fix_peak_height[i]] = peak1_idx[fix_peak_height[i]]
        else:
            peak1_height[fix_peak_height[i]+1] = peak1_idx[fix_peak_height[i]]
    peak1_height_gradient = np.gradient(peak1_height)
    num_wrongheights = 0

plt.figure()
plt.plot(datatime, peak1_height, 'o')
plt.plot(datatime, peak1_height_gradcorr, 'o', label = 'original')
plt.legend()
"""

peak_height1 = peak_height_matx[:, 0]
peak_height2 = peak_height_matx[:, 1]
sameheight = np.where(peak_height1 == peak_height2)[0]

mean1 = np.mean(peak_height1)
mean2 = np.mean(peak_height2)

if mean1 < mean2:
    peak_disappear = peak_height1
else:
    peak_disappear = peak_height2
    
len_repeat = len(sameheight)
#print (len_repeat)
for i in range(len_repeat):
    heightidx = sameheight[i]

#    print (heightidx)
    peak_height2[heightidx] = peak_height2[sameheight[0]-1]
#    print (peak_height2[heightidx])

plt.figure()
plt.plot(datatime, peak_height2)