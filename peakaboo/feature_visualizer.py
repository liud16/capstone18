import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

def visualize(peak_dict, data_nm):
        nm = pd.DataFrame(data_nm)
    for i in range(len(peak_dict)):
        nm_list = []
        df = pd.DataFrame(peak_dict['peak_%s' % i], 
        columns=['Position', 'Height', 'Width', 'Time'])
        df = df.drop_duplicates(subset= 'Time')
        df = df.reset_index(drop=True)
        for j in df['Position']:
            nm_list.append(nm.loc[j].values)    
        df['Wavelength'] = nm_list
        height_norm = np.linalg.norm(df['Height'], keepdims=True)
        fit_params, exp_fit, data, time = fit_single_exp_diffev(df['Time'], df['Height'])
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 8))
        
        ax1.plot(df['Time'], df['Height'], '.')
        #ax1.plot(exp fit)
        #ax1.axis(between 0 and 1?)
        ax1.set_title('Peak %s Dynamics' % (i+1), fontsize=18, fontweight='bold')
        ax1.set_ylabel('Intensity', fontsize=18, fontweight='bold')
        ax1.grid()
        
        ax2.plot(df['Time'], df['Position'], '.')
        ax2.plot(np.unique(df['Time']), np.poly1d(np.polyfit(df['Time'], df['Position'], 1))
        (np.unique(df['Time'])))
        ax2.set_ylabel('Position', fontsize=18, fontweight='bold')
        ax2.set_ylim((0, data_nm.shape[0]))
        ax2.grid()
        
        ax3.plot(df['Time'], df['Width'], '.')
        ax3.plot(np.unique(df['Time']), np.poly1d(np.polyfit(df['Time'], df['Width'], 1))
        (np.unique(df['Time'])))
        ax3.set_ylabel('Width', fontsize=18, fontweight='bold')
        ax3.set_xlabel('Time', fontsize=18, fontweight='bold')
        ax3.grid()
        
    return
