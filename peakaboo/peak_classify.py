import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def data_grouping(index_df, height_df, fwhm_df):
    for i in range(index_df.shape[0]):
        for j in range(index_df.shape[1]):
            all_points = np.array(
            [peak_idx_df.loc[i][j], peak_height_df.loc[i][j],
            peak_fwhm_df.loc[i][j], i])

    all_points_df = pd.DataFrame(all_points,
    columns=['Position', 'Height', 'Width', 'Time'])

    corrected_output = all_points_df.fillna(value=0)

    return corrected_output

def cluster_classifier(index_df, corrected_output):
    found_peak = index_df.shape[1]
    cluster = KMeans(n_clusters=found_peak).fit(corrected_output.iloc[:,:-1])

    peak1_list = []
    peak2_list = []
    peak3_list = []

    for i in range(1500):
        peak = cluster.predict([corrected_output.iloc[i,:-1]])
        signal = corrected_output.iloc[i][1]
        if ( peak == 0 and (signal >= 0.001 or signal <= -0.001)):
            peak1_list.append(corrected_output.iloc[i])
        elif ( peak == 1 and (signal >= 0.001 or signal <= -0.001)):
            peak2_list.append(corrected_output.iloc[i])
        elif ( peak == 2 and (signal >= 0.001 or signal <= -0.001)):
            peak3_list.append(corrected_output.iloc[i])
        else:
            pass
