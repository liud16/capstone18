import numpy as np
import matplotlib.pyplot as plt


def find_nearest(array,value):
    """
    find the nearest value to a given value
    Returns:
        the index of the nearest value in the array
    """
    idx = (np.abs(array-value)).argmin()
    return idx


def twodcontourplot(tadata_nm, tadata_timedelay, tadata_z_corr):
    """
    make contour plot
    
    Args:
        tadata_nm: wavelength array
        tadata_timedelay: time delay array
        tadata_z_corr: matrix of z values
        
    """
    nm, timedelay = np.linspace(tadata_nm.min(), tadata_nm.max(), 100), np.linspace(tadata_timedelay.min(), tadata_timedelay.max(), 100)    
    timedelayi, nmi = np.meshgrid(tadata_timedelay, tadata_nm)

    #find the maximum and minimum
    #these are used for color bar
    z_min = np.amin(np.amin(tadata_z_corr, axis = 1))
    z_max = np.amax(np.amax(tadata_z_corr, axis = 1))

    return [nmi, timedelayi, z_min, z_max]


#one gaussian
def gaussian(nm, a, x0, sigma):
    """
    gaussian function
    """
    gaussian_array = a * np.exp(- ((nm - x0) ** 2.0) / (2 * (sigma ** 2.0))) 
    
    return gaussian_array


#monoexponential
def monoexp(t, tau):
    """
    mono-exponential function
    
    Args:
        t: time array
        tau: life-time
    
    """
    exp_array = np.exp(- (1.0/tau) * t)
    
    return exp_array
 
#monoexponential
def monoexp_grow(t, tau):
    """
    mono-exponential function
    
    Args:
        t: time array
        tau: life-time
    
    """
    exp_array = 1- np.exp(- (1.0/tau) * t)
    
    return exp_array


def data_matrix(nm_array, time_coeff_array, spectrum):
    """
    generates a two-way data matrix based on a known 
    spectrum and the spectrum's decay
    
    Args:
        nm_array: wavelength array
        time_coeff_array: an array that describes the decay
        spectrum: an array of the spectrum
    
    Returns:
        data_matrix: a matrix that contains the spectrum at each time
    """
    
    data_matrix = np.empty((np.shape(nm_array)[0], np.shape(time_coeff_array)[0]))
    for i in range(np.shape(time_coeff_array)[0]):
        data_matrix[:, i] = time_coeff_array[i] * spectrum
    
    return data_matrix



def spectral_shift(start_nm, end_nm, time):
    """
    generates a linear spectral shift
    
    Args:
        start_nm: the starting peak position
        end_nm: the ending peak position
        time: an array of time
    
    Returns:
        an array of peak position within the given time
    
    """
    
    #calculate the step of peak shift at each time interval 
    #the peak shift is linear
    step = float((end_nm - start_nm)) / (len(time))
    
    x0 = np.arange(start_nm, end_nm, step)

    return x0


def gaussian_shift(nm, a, x0_shiftarray, sigma):
    """
    generates a matrix that contains a gaussian model that spectrally shifts
    
    Args:
        nm: wavelength array
        a: intensity of the gaussian
        x0_shiftarray: an array of peak positions
        sigma: gaussian FWHM
    
    Returns:
        a matrix that contains gaussian function that contains spectral shift
    
    """
    
    gaussian_matrix = np.empty((len(nm), len(x0_shiftarray)))
    for i in range(len(x0_shiftarray)):
        gaussian_matrix[:, i] = a * np.exp(- ((nm - x0_shiftarray[i]) ** 2.0) / (2 * (sigma ** 2.0)))
        
    return gaussian_matrix



def data_matrix_spectralshift(nm, time_coeff_array, spectrum_matrix):
    """
    generates a matrix that contains a gaussian model with a known decay
    
    Args:
        nm_array: wavelength array
        time_coeff_array: array of time coefficients that 
            describes the kinetics
        spectrum_matrix: a matrix that contains gaussian function at each time
    
    Returns:
        a matrix that contains gaussian function that evolves in time
        
    """
    data_matrix = np.empty((np.shape(nm)[0], np.shape(time_coeff_array)[0]))
    for i in range(np.shape(time_coeff_array)[0]):
        data_matrix[:, i] = time_coeff_array[i] * spectrum_matrix[:, i]
    
    return data_matrix


"""time and wavelength arrays"""
#create time array  
time = np.arange(0, 5000, 1)

#create wavelength array
nm = np.arange(850, 1600, 1)


"""define gaussian parameters"""
#intensity of the gaussian, 
#this is arbitrary but when there're 2 and more gaussians
#in the model, the intensity of each gaussian describes 
#its weight
a1 = 1

#center and FWHM of the gaussian 
x0_1 = 1000
sigma_1 = 300

#life-time of the gaussian
tau1 = 30


#create a second gaussian
a2 = 1
x0_2 = 1300
sigma_2 = 150
tau2 = 2000

#create a third gaussian
a3 = 2
x0_3 = 1400
sigma_3 = 60
tau3 = 2000

#generate a gaussian model
species_1 = gaussian(nm, a1, x0_1, sigma_1)

#generate an array of time-coefficients 
#describing a mono-exponential decay with a given lifetime 
time_coeff_1 = monoexp(time, tau1)

#generate a data matrix that contains a gaussian model at each
#time and decays mono-exponentially
data_matrix_1 = data_matrix(nm, time_coeff_1, species_1)


#generate a second data matrix that contains a gaussian model
#at each time and decays mono-exponentially
species_2 = gaussian(nm, a2, x0_2, sigma_2)
time_coeff_2 = monoexp(time, tau2)
data_matrix_2 = data_matrix(nm, time_coeff_2, species_2)

#generate a third data matrix that contains a gaussian model
#at each time and decays mono-exponentially
species_3 = gaussian(nm, a3, x0_3, sigma_3)
time_coeff_3 = monoexp_grow(time, tau3)
data_matrix_3 = data_matrix(nm, time_coeff_3, species_3)

#generate three gaussian mixture model by adding
#the two gaussians above
data_matrix_0 = data_matrix_1 + data_matrix_2 + data_matrix_3


#generate an array of peak positions that shifts from 1200 to 1300
#within a time array
x0_1_shift = spectral_shift(1200, 1000, time) 

#generates a matrix that contains a gaussian at each time with a shift
#in peak position
species_1_matrix = gaussian_shift(nm, a1, x0_1_shift, sigma_1)

#generates a matrix that contains a gaussian at each time with a peak
#shift and monoexponential decay
data_1_matrix_shift = data_matrix_spectralshift(nm, time_coeff_1, species_1_matrix)

#generate an array of peak positions that shifts from 1200 to 1300
#within a time array
x0_2_shift = spectral_shift(1100, 900, time) 

x0_3_shift = np.repeat(x0_3, len(time)) #no shift actually

#generates a matrix that contains a gaussian at each time with a shift
#in peak position
species_2_matrix = gaussian_shift(nm, a2, x0_1_shift, sigma_2)

#generates a matrix that contains a gaussian at each time with a peak
#shift and monoexponential decay
data_2_matrix_shift = data_matrix_spectralshift(nm, time_coeff_2, species_2_matrix)


#generate three gaussian mixture with spectral evolution
#by adding one gaussian that contains spectral shift and two without
#spetral shift
data_matrix = data_matrix_1 + data_2_matrix_shift + data_matrix_3


peakcenter_2 = [find_nearest(nm, center) for center in x0_2_shift]
peakcenter_1 = [find_nearest(nm, center) for center in x0_1_shift]
peakcenter_3 = [find_nearest(nm, center) for center in x0_3_shift]

peakwidth_1 = np.repeat(sigma_1, len(time))
peakwidth_2 = np.repeat(sigma_2, len(time))
peakwidth_3 = np.repeat(sigma_3, len(time))

peakheight_1 = np.max(data_1_matrix_shift, axis=0)
peakheight_2 = np.max(data_2_matrix_shift, axis=0)
peakheight_3 = np.max(data_matrix_3, axis=0)

#plt.figure()
#plt.plot(time, peakheight_3)

def output_peakinfo(peakcenter, peakheight, peakwidth):
    outpeak = np.empty((len(time), 3))
    outpeak[:, 0] = peakcenter
    outpeak[:, 1] = peakheight
    outpeak[:, 2] = peakwidth
    
    return outpeak

peak1info = output_peakinfo(peakcenter_1, peakheight_1, peakwidth_1)
peak2info = output_peakinfo(peakcenter_2, peakheight_2, peakwidth_2)
peak3info = output_peakinfo(peakcenter_3, peakheight_3, peakwidth_3)

out1 = np.savetxt('Three_gaussians_oneblue_onered_peak1.txt', peak1info, fmt='%.3e', delimiter='\t')
out2 = np.savetxt('Three_gaussians_oneblue_onered_peak2.txt', peak2info, fmt='%.3e', delimiter='\t')
out3 = np.savetxt('Three_gaussians_oneblue_onered_peak3.txt', peak3info, fmt='%.3e', delimiter='\t')

"""make 2d contour plot"""
#
plt.figure()
#plt.xlim(450,800)
plt.title('Three gaussians with spectral relaxation', fontsize = 16, fontweight = 'bold')
#plt.ylim(0,50)
plt.xlabel('Wavelength (nm)', fontsize = 16, fontweight = 'bold')
plt.ylabel('Time delay (ps)', fontsize = 16, fontweight = 'bold')
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
nmi_9, timedelayi_9, z_min_9, z_max_9 = twodcontourplot(nm, time, data_matrix)
plt.pcolormesh(nmi_9, timedelayi_9, data_matrix, cmap = 'PiYG', vmin=z_min_9, vmax=z_max_9)
plt.colorbar()
plt.tight_layout(pad=0.25, h_pad=None, w_pad=None, rect=None)


"""output data"""
#1st columns: wavelength
#1st rows: time
output = np.empty((np.shape(data_matrix)[0]+1, np.shape(data_matrix)[1]+1))
output[1:, 1:] = data_matrix
output[0, 1:] = time
output[1:, 0] = nm
outfile = np.savetxt('Three_gaussians_oneblue_onered.txt', output, fmt='%.3e', delimiter='\t')