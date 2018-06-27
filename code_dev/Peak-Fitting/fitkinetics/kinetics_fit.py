#-*- coding: utf-8 -*-
"""
Created on Fri Aug 25 10:10:12 2017

@author: lyundemi
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.special import gamma




def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx


#load data
def loaddata(decay_filename):
    decay_data = np.genfromtxt(decay_filename, delimiter='\t')
    time = decay_data[:, 0]
    time_use = time[:]
    decay_use = decay_data[:, 1]

    return time_use, decay_use



#obtain an array of z value at a particular time
#time_est_ps is an array of timepoints of interest
def timeslice(time_est_ps, time, nm, tadata_z_corr):
    tadata_z_timeslice_matx = np.empty((len(nm), len(time_est_ps)))
    for i in range(len(time_est_ps)):
        time_idx = find_nearest(time, time_est_ps[i])
        tadata_z_timeslice = tadata_z_corr[:, time_idx]
        tadata_z_timeslice_matx[:,i] = tadata_z_timeslice
       
    return tadata_z_timeslice_matx



def singleexpfunc(t, params):
    exp_array = params[0] *np.exp((-1.0/params[1]) * t)

    return exp_array



def exp_stretch(t, params):
    exp_stretch_array = params[0] * np.exp(-((1.0 / params[1]) * t) ** params[2])

    return exp_stretch_array



def fit_single_exp_diffev(t, data, bounds):
    time_array = t
    data_array = data
    def fit(params):
        decaymodel = singleexpfunc(time_array, params[:])
        cost = np.sum(((data_array - decaymodel) ** 2.0))
        return cost
    bestfit = differential_evolution(fit, bounds = bounds, polish = True)
    bestfit_params = bestfit.x
    def bestfit_decay(params):
        decaymodel = singleexpfunc(time_array, params[:])
        return decaymodel    
    bestfit_model = bestfit_decay(bestfit_params)   
    
    ss_res = np.sum((data_array - bestfit_model) ** 2.0)
    ss_tot = np.sum((data_array - np.mean(data_array)) ** 2.0)
    rsquare = 1 - (ss_res / ss_tot)
    #print '--Single exponential best fit parameters--'
    print ('a = %.5f  \ntau = %.5f ps  \nR-square = %.5f' %(bestfit_params[0], bestfit_params[1], rsquare))
    plt.figure()
    plt.ylabel('-$\Delta$T/T')   
    plt.xlabel('Time (ps)')

    plt.plot(time_array, data_array, 'o', color = 'b', label = 'Data')
    plt.plot(time_array, bestfit_model, color = 'r', label = 'Monoexponential')
#    plt.text(10, 0.002, 'tau = 3ps', fontsize = 14)

    plt.legend(loc = 'best')

    plt.figure()
    #plt.xlim(0, 200)
    plt.ylabel('-$\Delta$T/T')   
    plt.xlabel('Time (ps)')
    plt.xscale('log')    
    plt.plot(time_array, data_array, 'o', color = 'b', label = 'Data')
    plt.plot(time_array, bestfit_model, color = 'r', label = 'single exp fit')
    plt.legend(loc = 'best')

    return bestfit_params, bestfit_model, data_array, time_array


def fit_exp_stretch_diffev(t, data, bounds):
    time_array = t
    data_array = data
    def fit(params):
        decaymodel = exp_stretch(time_array, params[:])
        cost = np.sum(((data_array - decaymodel) ** 2.0))
        return cost
    bestfit = differential_evolution(fit, bounds = bounds, polish = True)
    bestfit_params = bestfit.x
    def bestfit_decay(params):
        decaymodel = exp_stretch(time_array, params[:])
        return decaymodel    
    bestfit_model = bestfit_decay(bestfit_params)   
    
    ss_res = np.sum((data_array - bestfit_model) ** 2.0)
    ss_tot = np.sum((data_array - np.mean(data_array)) ** 2.0)
    rsquare = 1 - (ss_res / ss_tot)
    
    print ('-Single exponential stretch Best Fit Parameters--')
    avg_tau = avg_tau_from_exp_stretch(bestfit_params[1], bestfit_params[2])
    print ('a = %.9f \ntau = %.5e us \nbeta = %.5e  \naverage tau = %.5e us' %(bestfit_params[0], bestfit_params[1], bestfit_params[2], avg_tau))
            
    print ('R-squared = %.5f'%(rsquare))
   
    plt.figure()
    #plt.xlim(0, 200)
    plt.ylabel('-$\Delta$T/T')   
    plt.xlabel('Time (ps)')
    plt.plot(time_array, data_array, 'o', color = 'b', label = 'Data')
    plt.plot(time_array, bestfit_model, color = 'r', label = 'Single exponential stretch fit')
    plt.legend(loc = 'best')

    plt.figure()
    #plt.xlim(0, 200)
    plt.ylabel('-$\Delta$T/T')   
    plt.xlabel('Time (ps)')
    plt.xscale('log')    
    plt.plot(time_array, data_array, 'o', color = 'b', label = 'Data')
    plt.plot(time_array, bestfit_model, color = 'r', label = 'single exponential stretch fit')
    plt.legend(loc = 'best')
    plt.tight_layout(pad=0.25, h_pad=None, w_pad=None, rect=None)

    return bestfit_params, bestfit_model, data_array, time_array


    
def avg_tau_from_exp_stretch(tc, beta):
    return (tc / beta) * gamma(1.0 / beta)



"""load TA data"""
#experiment name
experiment = ''

times, decaytrace = loaddata(experiment)



"""exponential decay parameters"""
a1_bounds = (0, 1)
tau1_bounds = (0, 100)
beta1_bounds = (0,1)

sing_expdec_bounds = [a1_bounds, tau1_bounds]
exp_stret_bounds = [a1_bounds, tau1_bounds, beta1_bounds]



"""fit data"""
#fit_data_sing_expdec = fit_single_exp_diffev(times, decaytrace, sing_expdec_bounds)

#fit_data_exp_stretch = fit_exp_stretch_diffev(times, decaytrace, exp_stret_bounds)
