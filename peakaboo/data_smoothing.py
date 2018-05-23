import numpy as np
from matplotlib import pyplot as plt
from pyearth import Earth

# load some data
def load_data(data_filename):
    """load matrix data"""
    data = np.genfromtxt(data_filename, delimiter='\t')
    data_nm = data[1:,0]    #wavelength in nm
    data_time = data[0,1:]
    data_z = data[1:, 1:]

    return data_nm, data_time, data_z

def earth_smoothing(nm_array, y_array):
    """
    =============================================
     Plotting derivatives of simple sine function
    =============================================

     A simple example plotting a fit of the sine function
    and the derivatives computed by Earth.

    Notes
    -----
    generates a denoise curve from the TA data
    Parameters
    ----------
        nm_array: wavelength array
        timedelay: time delay array
    Returns
    -------
        a smoothing curve from the original noise curve
    """
    from pyearth import Earth
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
    num_array = np.shape(data_matrix)[0]
    smooth_matx = pd.DataFrame(np.empty((num_array,1)), columns = ['a'])

    for i in range(500):
        data_array = data_matrix[:, i]
        smooth_array = Earth_Smoothing(nm_array, data_array).tolist()

        # get smooth dataframe
        df = pd.DataFrame(smooth_array, columns = [i])
        smooth_matx = smooth_matx.join(df)

    # drop the first columns
    smooth_matx = smooth_matx.drop(columns='a')

    return smooth_matx
