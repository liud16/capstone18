import numpy as np
from matplotlib import pyplot as plt
from pyearth import Earth
import pandas as pd


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


# load .txt data
def load_data(data_filename):
    """load matrix data"""
    data = np.genfromtxt(data_filename, delimiter='\t')
    data_nm = data[1:,0]    #wavelength in nm
    data_time = data[0,1:]
    data_z = data[1:, 1:]

    return data_nm, data_time, data_z

# load .csv data
def load_data_csv(data_filename):
    """load matrix data"""
    data = np.genfromtxt(data_filename, delimiter=',', skip_footer = 20)
    data_nm = np.nan_to_num(data[1:,0])    #wavelength in nm
    data_time = np.nan_to_num(data[0,1:])
    data_z = np.nan_to_num(data[1:, 1:])

    data_nm_use = data_nm[find_nearest(data_nm, 900):find_nearest(data_nm, 1400)]
    data_z_use = data_z[find_nearest(data_nm, 900):find_nearest(data_nm, 1400), :]
    

    return data_nm_use, data_time, data_z_use


def earth_smoothing(nm_array, y_array):
    """
    =============================================
    Smoothen noisy data using py-earth,
    based on multivariate adaptive regression spline
    =============================================


    Notes
    -----
    generates a de-noised curve from the TA data
    
    Parameters
    ----------
        nm_array: wavelength array
        y-array: intensity array
    Returns
    -------
        a smoothing curve from the original noise curve
    """
   # Fit an Earth model
    model = Earth(smooth=True)
    np.random.seed(42)
    model.fit(nm_array, y_array)

   # Print the model
    #print(model.trace())
    #print(model.summary())

   # Get the predicted values and derivatives
    y_hat = model.predict(nm_array)

    return  y_hat


def earth_smooth_matrix(nm_array, data_matrix):
    num_array = np.shape(data_matrix)[1]
    smooth_matx = pd.DataFrame(np.empty((len(nm_array),1)), columns = ['a'])

    #print (num_array)
    for i in range(num_array):
        data_array = data_matrix[:, i]
        smooth_array = earth_smoothing(nm_array, data_array).tolist()

        # get smooth dataframe
        df = pd.DataFrame(smooth_array, columns = [i])
        smooth_matx = smooth_matx.join(df)

    # drop the first columns
    smooth_matx = smooth_matx.drop(columns='a')

    return smooth_matx