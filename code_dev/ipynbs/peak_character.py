import numpy as np
import pandas as pd
import find_peaks as findpeak

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

 # get peak height and fwhm info
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


def peak_matrix(nm_array,data_matrix, threshold, mindist):
    """find peaks in a data matrix
    and calculate the height and width of the peaks"""
    
    peak_idx_matx = []
    peak_height_matx = []
    peak_fwhm_matx = []
    
    num_timeslice = np.shape(data_matrix)[1]
    
    for i in range(num_timeslice):
        data_timeslice = data_matrix.values[:, i]
        
        peak_idx = findpeak.indexes(data_timeslice, threshold, mindist).tolist()
        peak_idx_matx.append(peak_idx)
        
        
        peak_height, peak_fwhm = peakchar(nm_array, data_timeslice, peak_idx)
        
        peak_height_matx.append(peak_height)
        peak_fwhm_matx.append(peak_fwhm)
        
        # transfer to dataframe
        peak_idx_df=pd.DataFrame(peak_idx_matx)
        peak_height_df=pd.DataFrame(peak_height_matx)
        peak_fwhm_df=pd.DataFrame(peak_fwhm_matx)
        
    return peak_idx_df, peak_height_df, peak_fwhm_df
