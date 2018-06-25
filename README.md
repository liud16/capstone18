This is an overview of all the ready-to-use algorithms we've found to perform peak detection in smooth/noisy dataset.
peak smoothing --"pyearth"/"astropy", perform peak detection --"peakutils", peak correction --"LinearRegression"/
"Isotonic regression", peak classifying--"K-means" used in Python.

## Overview

| Algorithm | Integration | Filters | Comments /(`restriction`)|
|-----------| ----------- | ------- | -----------------------  |
| [pyearth.peak.smoothing](#pyearthpeaksmoothing) | py-earth package<br>Splines algorithm | Multivariate Adaptive Regression<br>Splines algorithm | less trivial<br>direct than other algorithms|
| [astropy.peak.smoothing](#astropypeaksmoothing) | astropy.modeling Package | parameter constraints |requires initial<br> noisy data guess |
| [peakutils.peak.indexes](#peakutilspeakindexes) | PyPI package PeakUtils<br> Depends on Scipy | Amplitude threshold<br>Minimum distance |limited peak return infos<br>indexes only |


## How to make your choice?

When you're selecting an algorithm, you might consider:

* **The function interface.** You may want the function to work on smoothing the noise TA data or may search and detect the peak values of dataset during different timeslices? or you want to correct peak values? and seek for relationship between peaks?
* **The accuracy.** Does user want extra accuarcy?
* **The input info support**. Does the algorithm require user to define initial guess ? Which ones do you need?

--------------------------------


## pyearth.peak.smoothing
This function searches for peaks based on convoluted value compared to neighboring points and returns those peaks whose properties match optionally specified conditions (minimum and/or maximum) for their height, width, indices, threshold and distance to each other.

<img width="498" alt="image" src="https://user-images.githubusercontent.com/35111515/41867976-827f06e0-7869-11e8-9735-a158b403ab98.png">

<img width="483" alt="image" src="https://user-images.githubusercontent.com/35111515/41868666-a254bd50-786b-11e8-99ba-99eec47f9556.png">



```python
def earth_Smoothing(nm_array, y_array,noise_coefficient):        
    """
    Smoothen noisy data using py-earth,
    based on multivariate adaptive regression spline
    
    Args:
        nm_array: wavelength array
        y-array: intensity array
        
    Returns:
        a smoothing curve from the original noise curve
    
    """
    from pyearth import Earth
   # Fit an Earth model
    model = Earth(smooth=True)
    np.random.seed(42)
    ydata = y_array + noise_coefficient*np.random.normal(size=nm_array.size)
    model.fit(nm_array, ydata)
   # Print the model
    print(model.trace())
    print(model.summary())
   # Get the predicted values and derivatives
    y_hat = model.predict(nm_array)

    return  y_hat
```

[Documentation](https://contrib.scikit-learn.org/py-earth/content.html).
[Sample code](http://localhost:8888/edit/getbest/py.docs/py-earth.py).



## astropy.peak.smoothing
This function smoothens the original noisy data while not losing the information. Compared to other smoothening algorithms, *astropy* can best preserve the shape of the curve and effectively reduces the noise. However, the algorithm requires a rough initial guess of the peak info of the data.

<img width="476" alt="image" src="https://user-images.githubusercontent.com/35111515/41868275-7584ffe8-786a-11e8-97dc-4e615a9e0493.png">


```python
def astropy_smoothing(nm_array, timedelay, noise_coefficient,gg_init):
    # Generate fake data
    np.random.seed(42)
    ydata = timedelay + noise_coefficient*np.random.normal(size=nm_array.size)
    # Now to fit the data create a new superposition with initial
    # guesses for the parameters:
    fitter = fitting.SLSQPLSQFitter()
    gg_fit = fitter(gg_init, nm_array, ydata)
    # Plot the data with the best-fit model
    plt.figure(figsize=(50,40))
    plt.subplot(311)
    plt.title('Smoothening data', fontsize = 30, fontweight = 'bold')
    plt.xlabel('Wavelength (nm)', fontsize = 30, fontweight = 'bold')
    plt.ylabel('Time delay (ps)', fontsize = 30, fontweight = 'bold')
    plt.plot(nm_array, ydata, 'ko')
    plt.plot(nm_array, gg_fit(nm_array), color='blue')
    plt.subplot(312)
    plt.title('True data', fontsize = 30, fontweight = 'bold')
    plt.xlabel('Wavelength (nm)', fontsize = 30, fontweight = 'bold')
    plt.ylabel('Time delay (ps)', fontsize = 30, fontweight = 'bold')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.plot(nm_array,timedelay,color='red')
    plt.plot(nm_array,gg_fit(nm_array), color='blue')
    plt.subplot(313)
    plt.title('data differ',fontsize = 30, fontweight = 'bold')
    plt.ylabel('Time delay (ps)', fontsize = 15, fontweight = 'bold')
    plt.xlabel('Wavelength (nm)', fontsize = 30, fontweight = 'bold')
    plt.ylabel('Time delay (ps)', fontsize = 30, fontweight = 'bold')
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 14)
    plt.axhline(y=0, color='red', linestyle='-')
    plt.plot(nm_array,timedelay-gg_fit(nm_array), color='blue')

    return gg_fit(nm_array)
```

[Documentation](http://docs.astropy.org/en/stable/modeling/).
[Sample code](http://localhost:8888/edit/getbest/py.docs/astropy.py).



## peakutils.peak.indexes
This algorithm can be used as an equivalent of the MatLab `findpeaks` and will give easily give consistent results if you only need minimal distance and height filtering.

<img width="457" alt="image" src="https://user-images.githubusercontent.com/35111515/41868068-d4f4b262-7869-11e8-8a6a-6564e94f95e0.png">

```python
filename = '20180418_twogaussian_spectralshfit.txt'
nm, time, z = loaddata(filename)
num_timeslice = 3
def peak_matrix(nm_array,data_matrix,num_timeslice, threshold, mindist):
    peak_idx_matx = np.zeros((num_timeslice,2))
    peak_height_matx = np.empty_like(peak_idx_matx)
    peak_fwhm_matx = np.empty_like(peak_height_matx)
    for i in range(num_timeslice):
        data_timeslice = data_matrix[:, i]
        peak_idx = findpeak(data_timeslice, threshold, mindist)
        peak_idx_matx[i, :] = peak_idx
        peak_height, peak_fwhm = peakchar(nm, data_timeslice, peak_idx)
        peak_height_matx[i, :], peak_fwhm_matx[i, :] = peak_height, peak_fwhm
    return peak_idx_matx, peak_height_matx, peak_fwhm_matx

peak_idx_matx, peak_height_matx, peak_fwhm_matx = peak_matrix(nm,z,num_timeslice, 0, 0)


```

[Documentation](http://peakutils.readthedocs.io/en/latest/).
[Package](https://bitbucket.org/lucashnegri/peakutils).
[Sample code](http://localhost:8888/edit/peakaboo/code/Peak-Smoothing/peakutils_2.py).



## remove outliers in peak position
Applying this function to the output (peak indices) from peakutils.peak.indexes removes outliers and sudden fluctuation in peak position

<img width="493" alt="image" src="https://user-images.githubusercontent.com/35111515/41867681-a5525970-7868-11e8-9402-4bbaaa02a0ab.png">

<img width="476" alt="image" src="https://user-images.githubusercontent.com/35111515/41868197-340e5ce4-786a-11e8-9e12-4b9d96b16305.png">

<img width="448" alt="image" src="https://user-images.githubusercontent.com/35111515/41868404-def1b480-786a-11e8-92f1-c0789c9b948d.png">



```python
def id_outliers_replacewith_interp(x_array, data, m, win_len):
    """
    identify outliers in an array of peak positions
    
    Args:
        x_array: time array, numpy array
        data: 1D wavelength or index array, numpy array
        m: scale of standard deviation used to identify outlier,
            float between 0. and 1.
        win_len: window size to search for outlier
    
    Returns:
        new_data_final_interp: a new data array that's absent of outliers
        
    """
    reshape_x_array = []
    reshape_data = []
    quotient_array = np.empty(len(data))
    remainder_array = np.empty(len(data))
    quotient_array[0] = 0
    remainder_array[0] = 0  
    #print divmod(len(data), win_len)   
    quotient_max = divmod(len(data), win_len)[0]
    print (quotient_max)
    #quotient_array_new = []
    data_idx = np.arange(0, len(data), 1)
    for i in range(1, len(data_idx)):
        
        quotient = divmod(data_idx[i], win_len)[0]
        quotient_array[i] = quotient
        remainder = divmod(data_idx[i], win_len)[1]
        remainder_array[i] = remainder
        
        if quotient != quotient_array[i-1]:
            newslice = data[i - win_len: i]
            newslice_x = x_array[i - win_len: i]
            #print newslice
            reshape_data.append(newslice)
            reshape_x_array.append(newslice_x)
    
        else:
            pass
    quotient_max_idx = np.where(quotient_array == quotient_max)
    #print quotient_max_idx
    reshape_data.append(data[quotient_max_idx[0]])
    reshape_x_array.append(x_array[quotient_max_idx[0]])
    #print reshape_data
    reshape_data_shape = np.shape(reshape_data)[0]
    #print reshape_data_shape
    def id_outliers_and_delete(d,x, m):
        d_mean = np.mean(d)  
        d_stdev = np.std(d)
        new_d = np.empty_like(d)    
        
        for i in range(len(d)):
            d_pt = d[i]
          
            if abs(d_pt - d_mean) > m * d_stdev and x[i] != x_array[0] and x[i] != x_array[len(x_array) - 1]:
                new_d[i] = 1
            else:
                new_d[i] = 0
    
        outlier_idx = np.nonzero(new_d)[0]
        d_delete = np.delete(d, outlier_idx)
        x_delete = np.delete(x, outlier_idx)
        
        #print data2[outlier_idx]
        return x_delete, d_delete
    
    new_x_array = []
    new_data = []
    for i in range(reshape_data_shape):
        new_data.append(id_outliers_and_delete(reshape_data[i],reshape_x_array[i], 1)[1])#(id_outliers_replacewith_mean(reshape_data[i], m))
        new_x_array.append(id_outliers_and_delete(reshape_data[i],reshape_x_array[i],1)[0])
    new_data_flat = np.concatenate(new_data[:-1]).ravel().tolist()#.flatten()
    new_x_array_flat = np.concatenate(new_x_array[:-1]).ravel().tolist()#.flatten()
    new_data_final = np.concatenate((new_data_flat, new_data[reshape_data_shape - 1]))
    new_x_array_final = np.concatenate((new_x_array_flat, new_x_array[reshape_data_shape - 1]))
    
    new_data_final_interp = np.interp(x_array, new_x_array_final, new_data_final)    
    
    return new_data_final_interp


def isotonic(x, y):
    """
    predict y-array based on isotonic regression model
    
    Args:
        x: time array, numpy array
        y: peak position or nm, numpy array
    
    Returns:
        y_: the isotonic transform based on the given time array and peak array
    
    """
    ir = IsotonicRegression()
    y_ = ir.fit_transform(x, y)
    
    return y_