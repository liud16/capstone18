# importing the libraries
import numpy as np
import peakutils
import syntheticdata
import threegaussians
import lorentzian
from peakutils.plot import plot as pplot
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy import interpolate

# using savitzky_golay function to smoothen the noise data
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def savitzky_golay_smoothing(nm_array, timedelay, noise_coefficient):
    import numpy as np
    import scipy.fftpack
    from matplotlib import pyplot as plt
    
    
    """
    Notes
    -----   
    generates a denoise curve from the TA data
    Parameters
    ----------
        nm_array: wavelength array
        timedelay: intensity array
        noise_coefficient: the noise coefficients that user want to generate
    Returns
    -------
        a smoothing curve from the original noise curve   
    """
 
   # generate some noisy data from syntheticdata:
    np.random.seed(1729)
    y_noise = noise_coefficient * np.random.normal(size=nm_array.size)
    ydata = timedelay + y_noise
    y_hat = savitzky_golay(ydata, 51, 3) # window size 51, polynomial order 3
    
   # plot the noise data curve and denoise curve
    plt.figure(figsize=(15,8))
    plt.title('Peak Smoothening', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Wavelength (nm)', fontsize = 16, fontweight = 'bold')
    plt.ylabel('Intensity', fontsize = 16, fontweight = 'bold')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.plot(nm_array,ydata)
    plt.plot(nm_array,y_hat, color='red')
    
    plt.show()
    return y_hat


def Earth_Smoothing(nm_array, timedelay, noise_coefficient):
    import numpy
    import matplotlib.pyplot as plt
    from pyearth import Earth
    
    """
     ============================================
     Plotting derivatives of simple sine function
    ============================================

     A simple example plotting a fit of the sine function
    and the derivatives computed by Earth.
    
    Notes
    -----   
    generates a denoise curve from the TA data
    Parameters
    ----------
        nm_array: wavelength array
        timedelay: intensity array
        noise_coefficient: the noise coefficients that user want to generate
    Returns
    -------
        a smoothing curve from the original noise curve   
    """
    # Create some fake data
    # generate some noisy data from syntheticdata:
    np.random.seed(1729)
    y_noise = noise_coefficient * np.random.normal(size=nm_array.size)
    ydata = timedelay + y_noise
    
   # Fit an Earth model
    model = Earth(max_degree=2, minspan_alpha=.5, smooth=True)
    model.fit(nm_array, ydata)

   # Print the model
    print(model.trace())
    print(model.summary())

   # Get the predicted values and derivatives
    y_hat = model.predict(nm_array)

    # Plot true and predicted function values 
    plt.figure(figsize=(15,8))
    plt.title('Peak Smoothening', fontsize = 16, fontweight = 'bold')
    plt.xlabel('Wavelength (nm)', fontsize = 16, fontweight = 'bold')
    plt.ylabel('Intensity', fontsize = 16, fontweight = 'bold')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.plot(nm_array, ydata, 'r.')
    plt.plot(nm_array, y_hat, 'b.')
    plt.ylabel('function')
    
    return

    
