# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 09:08:14 2017

@author: lyundemi
"""

import numpy as np
from scipy.optimize import differential_evolution, fmin_tnc
import itertools
import multiprocessing as mp
import random
import matplotlib.pyplot as plt
from math import e


def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx



def index_expo(lim):
    array_expo = [int(random.expovariate(0.01)) for i in range(lim)]    
    array_expo_cut = array_expo[::2]
    return array_expo_cut
    
    

def loaddata(tadata_filename):
    tadata = np.genfromtxt(tadata_filename, delimiter=',', skip_footer = 19)
    tadata_z = tadata[1:,1:]
    tadata_timedelay = tadata[0,1:] #timedelay in ps
    tadata_nm = tadata[1:,0]    #wavelength in nm
    tadata_z_corr = np.empty((len(tadata_nm), len(tadata_timedelay))) #
    tadata_z_corr = np.nan_to_num(tadata_z)
    
    return [tadata_timedelay, tadata_nm, -tadata_z_corr]

    
#calculate sigma    
def sigma(tadata_file1, tadata_file2, intsec):
    
    tadata_time1, tadata_nm1, tadata_z1 = loaddata(tadata_file1)
    tadata_time2, tadata_nm2, tadata_z2 = loaddata(tadata_file2)


    N = intsec / 0.002
    
    mean = np.mean(np.array([tadata_z1, tadata_z2]), axis=0)
    sigma = np.std(np.array([tadata_z1, tadata_z2]), axis=0) / np.sqrt(N)
    
    return [mean, sigma, tadata_time1, tadata_nm1, N]
    

def sigma_2mat(tadata_file, intsec):
    
    tadata_time_avg, tadata_nm_avg, tadata_z_avg = loaddata(tadata_file)


    N = intsec / 0.002
    
    return [tadata_z_avg, tadata_time_avg, tadata_nm_avg, N]


def time_list_idx(time, randomtimes):
    times_idx = np.empty(len(randomtimes))
    for i in range(len(randomtimes)):
        times_idx[i] = find_nearest(time, randomtimes[i])
    
    return times_idx


def fourgaussbounds(x1_bounds, sig1_bounds, x2_bounds, sig2_bounds, x3_bounds, sig3_bounds, x4_bounds, sig4_bounds, a1_bounds, a2_bounds, a3_bounds, a4_bounds, tpnum):
    a1_bounds_tot = [a1_bounds] * tpnum
    a2_bounds_tot = [a2_bounds] * tpnum
    a3_bounds_tot = [a3_bounds] * tpnum
    a4_bounds_tot = [a4_bounds] * tpnum

    
    gaussbounds_tot = [x1_bounds, sig1_bounds, x2_bounds, sig2_bounds, x3_bounds, sig3_bounds, x4_bounds, sig4_bounds] 
    for i in range(len(a1_bounds_tot)):
        gaussbounds_tot += tuple([a1_bounds_tot[i]])
    
    for i in range(len(a1_bounds_tot)):
        gaussbounds_tot += tuple([a2_bounds_tot[i]])

    for i in range(len(a1_bounds_tot)):
        gaussbounds_tot += tuple([a3_bounds_tot[i]])

    for i in range(len(a1_bounds_tot)):
        gaussbounds_tot += tuple([a4_bounds_tot[i]])

    return gaussbounds_tot


def threegaussbounds(x1_bounds, sig1_bounds, x2_bounds, sig2_bounds, x3_bounds, sig3_bounds, a1_bounds, a2_bounds, a3_bounds, tpnum):
    a1_bounds_tot = [a1_bounds] * tpnum
    a2_bounds_tot = [a2_bounds] * tpnum
    a3_bounds_tot = [a3_bounds] * tpnum


    
    gaussbounds_tot = [x1_bounds, sig1_bounds, x2_bounds, sig2_bounds, x3_bounds, sig3_bounds] 
    for i in range(len(a1_bounds_tot)):
        gaussbounds_tot += tuple([a1_bounds_tot[i]])
    
    for i in range(len(a1_bounds_tot)):
        gaussbounds_tot += tuple([a2_bounds_tot[i]])

    for i in range(len(a1_bounds_tot)):
        gaussbounds_tot += tuple([a3_bounds_tot[i]])



    return gaussbounds_tot


def twogaussbounds(a1_bounds, a2_bounds, tpnum):
    a1_bounds_tot = [a1_bounds] * tpnum
    a2_bounds_tot = [a2_bounds] * tpnum

    
    gaussbounds_tot = [] 
    for i in range(len(a1_bounds_tot)):
        gaussbounds_tot += tuple([a1_bounds_tot[i]])
    
    for i in range(len(a1_bounds_tot)):
        gaussbounds_tot += tuple([a2_bounds_tot[i]])

    return gaussbounds_tot



#one gaussian
def onegauss(x, abcparams):

    gauss_array_sing = np.empty(len(x))
    gauss_array_sing = abcparams[0] * np.exp(- ((x - abcparams[1]) ** 2.0) / (2 * (abcparams[2] ** 2.0))) 
    
        
    return gauss_array_sing


#multi gaussian
def multi_gauss(x, params, num_gauss):
    gauss_array = np.empty((len(x), num_gauss))

    i = 0
    while (i<len(params)):
        remain = i % 3
        if remain == 0:
            
            gauss = params[i] * np.exp(- ((x - params[i + 1]) ** 2.0) / (2 * (params[i + 2] ** 2.0))) 
            gauss_idx = i / 3
            gauss_array[:,gauss_idx] = gauss
        else:
            pass
        i = i + 1
    
    gauss_result = np.sum(gauss_array, axis = 1)
    
    return gauss_result


#bounds for gaussian parameters
a1 = (0, 5e-3)
x1 = 966
sig1 =  37
a2 = (0e-3, 4e-3)
x2 = 1080
sig2 =  67
a3 = (0, 8e-3)
x3 = (400, 1300)
sig3 = (0,800)
#a4 = (0, 8e-3)
#x4 = (400, 1300)
#sig4 = (0,800)


#four gaussian at multiple timepoints
def four_gauss_mult_tpt(x, params, num_tpt):

    a1_repeats = params[8: (8 + num_tpt)]
    a2_repeats = params[(8+num_tpt) : (8 + 2 * num_tpt)]
    a3_repeats = params[(8 + 2 * num_tpt) :(8 + 3 * num_tpt)]
    a4_repeats = params[(8 + 3 * num_tpt) :(8 + 4 * num_tpt)]
    
    gauss_mult_tpt = np.empty((len(x), num_tpt))
    
    for i in range(len(a1_repeats)):
        gauss_single_tpt_1 = a1_repeats[i] * np.exp(- ((x-params[0]) ** 2.0) / (2 * (params[1] ** 2.0)))
        gauss_single_tpt_2 = a2_repeats[i] * np.exp(- ((x-params[2]) ** 2.0) / (2 * (params[3] ** 2.0)))        
        gauss_single_tpt_3 = a3_repeats[i] * np.exp(- ((x-params[4]) ** 2.0) / (2 * (params[5] ** 2.0)))    
        gauss_single_tpt_4 = a4_repeats[i] * np.exp(- ((x-params[6]) ** 2.0) / (2 * (params[7] ** 2.0)))

        
        gauss_mult_tpt[:, i] = gauss_single_tpt_1 + gauss_single_tpt_2 + gauss_single_tpt_3 + gauss_single_tpt_4
    
    return gauss_mult_tpt

#three gaussian at multiple timepoints
def three_gauss_mult_tpt(x, params, num_tpt):

    a1_repeats = params[6: (6 + num_tpt)]
    a2_repeats = params[(6+num_tpt) : (6 + 2 * num_tpt)]
    a3_repeats = params[(6 + 2 * num_tpt) :(6 + 3 * num_tpt)]

    
    gauss_mult_tpt = np.empty((len(x), num_tpt))
    
    for i in range(len(a1_repeats)):
        gauss_single_tpt_1 = a1_repeats[i] * np.exp(- ((x-params[0]) ** 2.0) / (2 * (params[1] ** 2.0)))
        gauss_single_tpt_2 = a2_repeats[i] * np.exp(- ((x-params[2]) ** 2.0) / (2 * (params[3] ** 2.0)))        
        gauss_single_tpt_3 = a3_repeats[i] * np.exp(- ((x-params[4]) ** 2.0) / (2 * (params[5] ** 2.0)))    


        
        gauss_mult_tpt[:, i] = gauss_single_tpt_1 + gauss_single_tpt_2 + gauss_single_tpt_3
    
    return gauss_mult_tpt


#two gaussian at multiple timepoints
def two_gauss_mult_tpt(x, params, num_tpt):

    a1_repeats = params[0: num_tpt]
    a2_repeats = params[num_tpt : 2 * num_tpt]


    
    gauss_mult_tpt = np.empty((len(x), num_tpt))
    
    for i in range(len(a1_repeats)):
        gauss_single_tpt_1 = a1_repeats[i] * np.exp(- ((x-x1) ** 2.0) / (2 * (sig1 ** 2.0)))
        gauss_single_tpt_2 = a2_repeats[i] * np.exp(- ((x-x2) ** 2.0) / (2 * (sig2 ** 2.0)))        

        gauss_mult_tpt[:, i] = gauss_single_tpt_1 + gauss_single_tpt_2
    
    return gauss_mult_tpt


#load data, define wavelength range
experimentname = 'exp03_20180103 -t0 -chirp'
csv_filename = experimentname + '.csv'
dt_t_matx, time, nm, N = sigma_2mat(csv_filename, 3.5)
nm_start_idx = find_nearest(nm, 880)
nm_stop_idx = find_nearest(nm, 1400)

nm_use = np.copy(nm[nm_start_idx:nm_stop_idx])
dt_t_matx_use = np.copy(dt_t_matx[nm_start_idx:nm_stop_idx, :])
#standev_use = np.copy(standev[nm_start_idx:nm_stop_idx, :])

#bounds for gaussian parameters
a1 = (0, 5e-3)
x1 = 966
sig1 =  37
a2 = (0, 4e-3)
x2 = 1080
sig2 =  67
a3 = (0, 8e-3)
x3 = (400, 1300)
sig3 = (0,800)
#a4 = (0, 8e-3)
#x4 = (400, 1300)
#sig4 = (0,800)


#choose time points, and number of gaussians

time_start = find_nearest(time, 19)
time_end = find_nearest(time, 51)
num_time_tot = len(time)
time_idx_list = np.arange(time_start, time_end)
num_times = len(time_idx_list)
random_times = time[time_start:time_end]



#random_times = []
#random_times_log = np.logspace(0, 3.5, num=25, base = e)
#random_times = random_times + random_times_log.tolist()
#num_times = len(random_times)
#time_idx_list = time_list_idx(time, random_times)
#

num_gauss = 2



#cost function for fitting four gaussians at various time points
def min_four_gauss_global(params):
    model = four_gauss_mult_tpt(nm_use, params, num_times)

    cost = np.empty(len(random_times))
    
    for i in range(len(cost)):
        data_idx = int(time_idx_list[i])
        
        cost[i] = np.sum((model[:, i] - dt_t_matx_use[:, data_idx]) ** 2.0)

    cost_result = np.sum([i ** 2 for i in cost])
    
    return cost_result


#cost function for fitting three gaussians at various time points
def min_three_gauss_global(params):
    model = three_gauss_mult_tpt(nm_use, params, num_times)

    cost = np.empty(len(random_times))
    
    for i in range(len(cost)):
        data_idx = int(time_idx_list[i])
        
        cost[i] = np.sum((model[:, i] - dt_t_matx_use[:, data_idx]) ** 2.0)

    cost_result = np.sum([i ** 2 for i in cost])
    
    return cost_result


#cost function for fitting two gaussians at various time points
def min_two_gauss_global(params):
    model = two_gauss_mult_tpt(nm_use, params, num_times)

    cost = np.empty(len(random_times))
    
    for i in range(len(cost)):
        data_idx = int(time_idx_list[i])
        
        cost[i] = np.sum((model[:, i] - dt_t_matx_use[:, data_idx]) ** 2.0)

    cost_result = np.sum([i ** 2 for i in cost])
    
    return cost_result


#generate a list of gaussian parameters
#model_bounds = threegaussbounds(x1, sig1, x2, sig2, x3, sig3, a1, a2, a3, num_times)
model_bounds = twogaussbounds(a1, a2, num_times)

#FIT DATA, use differential evolution built-in function
modelresult = differential_evolution(min_two_gauss_global, bounds = model_bounds)
model = modelresult.x


#define parameters for each gaussian at each time point
#x1_m, sig1_m, x2_m, sig2_m, x3_m, sig3_m = [model[0], model[1], model[2], model[3], model[4], model[5]]
#a1_m_list = model[6:(6 + num_times)]
#a2_m_list = model[(6 + num_times) : (6 + 2 * num_times)]
#a3_m_list = model[(6 + 2 * num_times) :(6 + 3 * num_times)]

x1_m, sig1_m, x2_m, sig2_m = [x1, sig1, x2, sig2]
a1_m_list = model[0:num_times]
a2_m_list = model[num_times : 2 * num_times]


"""plotting"""
#plot multi-gaussian model and data at each timepoint
model_matx = np.empty((len(nm_use), num_times))
residual_matx = np.empty_like(model_matx)
gauss1_matx = np.empty_like(residual_matx)
gauss2_matx = np.empty_like(residual_matx)

for i in range(num_times):
    model_params = [a1_m_list[i], x1_m, sig1_m, a2_m_list[i], x2_m, sig2_m]
    
    model_timeslice = multi_gauss(nm_use, model_params, num_gauss)
    data_timeslice = dt_t_matx_use[:, int(time_idx_list[i])]
#    plt.figure()
#    plt.plot(nm_use, data_timeslice, label = 'data, ' + str(random_times[i]) + ' ps')
#    plt.plot(nm_use, model_timeslice, label = 'model, ' + str(random_times[i]) + ' ps')
#    plt.legend()

    gauss1_params = [a1_m_list[i], x1_m, sig1_m]
    gauss2_params = [a2_m_list[i], x2_m, sig2_m]

    gauss1_timeslice = onegauss(nm_use, gauss1_params)
    gauss2_timeslice = onegauss(nm_use, gauss2_params)
    
#    plt.figure()
#    plt.title(str(random_times[i]) + ' ps')
#    plt.plot(nm_use, gauss1_timeslice, label = 'gaussian 1')
#    plt.plot(nm_use, gauss2_timeslice, label = 'gaussian 2')
#    plt.legend()

    gauss1_matx[:, i] = gauss1_timeslice
    gauss2_matx[:, i] = gauss2_timeslice
    model_matx[:, i] = model_timeslice
    residual_matx[:, i] = data_timeslice - model_timeslice

    
#plot one gaussian component in at different time points    
#plt.figure()    
#for i in range(num_times):
#    gauss1_params = [a1_m_list[i], x1_m, sig1_m]
#    plt.title('gaussian 1')
#    plt.plot(nm_use, onegauss(nm_use, gauss1_params), label = str(random_times[i]) + ' ps')
#plt.legend()
#
#plt.figure()    
#for i in range(num_times):    
#    gauss2_params = [a2_m_list[i], x2_m, sig2_m]
#    plt.title('gaussian 2')
#    plt.plot(nm_use, onegauss(nm_use, gauss2_params), label = str(random_times[i]) + ' ps')
#plt.legend()


    
out_1 = np.empty((np.shape(residual_matx)[0]+1, np.shape(residual_matx)[1]+1))
out_1[1:,0] = nm_use
out_1[0,1:] = random_times
out_1[1:,1:] = residual_matx
np.savetxt(experimentname + '_' + str(num_gauss) + '20to50_gaussian_residuals.csv', out_1, delimiter = ',', fmt = '%.6e')

out_2 = np.empty((np.shape(residual_matx)[0]+1, np.shape(residual_matx)[1]+1))
out_2[1:,0] = nm_use
out_2[0,1:] = random_times
out_2[1:,1:] = model_matx
np.savetxt(experimentname + '_' + str(num_gauss) + '20to50_gaussian_model.csv', out_2, delimiter = ',', fmt = '%.6e')

out_3 = np.empty((np.shape(residual_matx)[0]+1, np.shape(residual_matx)[1]+1))
out_3[1:,0] = nm_use
out_3[0,1:] = random_times
out_3[1:,1:] = gauss1_matx
np.savetxt(experimentname + '_' + str(num_gauss) + '20to50_gaussian_model_component1.csv', out_3, delimiter = ',', fmt = '%.6e')

out_4 = np.empty((np.shape(residual_matx)[0]+1, np.shape(residual_matx)[1]+1))
out_4[1:,0] = nm_use
out_4[0,1:] = random_times
out_4[1:,1:] = gauss2_matx
np.savetxt(experimentname + '_' + str(num_gauss) + '20to50_gaussian_model_component2.csv', out_4, delimiter = ',', fmt = '%.6e')
    
