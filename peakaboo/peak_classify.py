import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def data_grouping(index_df, height_df, fwhm_df):
    peak_list = []
    
    for i in range(index_df.shape[0]):
        for j in range(index_df.shape[1]):
            peak_list.append(
            [index_df.loc[i,j], height_df.loc[i,j], fwhm_df.loc[i,j], i])
        
    all_points = pd.DataFrame(peak_list, 
    columns=['Position', 'Height', 'Width', 'Time'])
    corrected_output = all_points.fillna(value=0)
    
    return corrected_output

def cluster_classifier(index_df, corrected_output):
    found_peak = index_df.shape[1]
    cluster = KMeans(n_clusters=found_peak).fit(corrected_output.iloc[:,:-1])
    peak_dict = {}

    for i in range(index_df.shape[0]):
        peak = cluster.predict([corrected_output.iloc[i,:-1]])
        signal = corrected_output.iloc[i][1]
        for j in range(found_peak):
            peak_dict['peak_%s' % j] = []
            
            if ( peak == j and (signal >= 0.001 or signal <= -0.001)):
            peak_dict['peak_%s' % j].append(corrected_output.iloc[i])

            else:
                pass
    return peak_dict
