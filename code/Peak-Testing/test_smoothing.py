import smoothing
import numpy as np
from scipy.optimize import curve_fit
from scipy import interpolate
from astropy.modeling import models, fitting
import pandas as pd

def test_output():
    matx_filename = '20180418_twogaussian_spectralshfit.txt'
    datanm, datatime, dataz_matx = smoothing.loaddata(matx_filename)
    noisez_matx, smooth_matx = smoothing.earth_smooth_matrix(datanm,dataz_matx,0.1)
    if isinstance(noisez_matx, pd.DataFrame):
        pass
    else:
        raise Exception('Bad type', 'Not a dataframe')
    if isinstance(smooth_matx, pd.DataFrame):
        pass
    else:
        raise Exception('Bad type', 'Not a dataframe')

    return

def test_values():
    matx_filename = '20180418_twogaussian_spectralshfit.txt'
    datanm, datatime, dataz_matx = smoothing.loaddata(matx_filename)
    noisez_matx, smooth_matx = smoothing.earth_smooth_matrix(datanm,dataz_matx,0.1)
    i=0
    assert noisez_matx.iloc[i][i] == smoothing.add_noise(datanm, dataz_matx[:,i], 0.1)[i],'cannot add right noisy value'
    #check if we can get right smooth value from noisy dataset
    i=0
    assert smooth_matx.iloc[i][i]  == smoothing.earth_Smoothing(datanm,dataz_matx[:,i],0.1)[i],'cannot get right smooth value'
    
    return
