#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 10:36:43 2018

@author: demiliu
"""


import numpy as np
import sklearn
from sklearn.mixture import GaussianMixture as Gaussian
import matplotlib.pyplot as plt
import pomegranate

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def loaddata(filename):
    data = np.genfromtxt(filename, delimiter = '\t')
    
    return data

data = loaddata('syntheticdata_one_timeslice.txt')
time = data[:, 0]
data = data[find_nearest(time, 900):, :]
time = data[:, 0]
signal_array= -data[:, 1]
signal_num = np.shape(signal_array)[0]
signal = np.reshape(signal_array, (-1, 1))


#Center the x data in zero and normalize
mid_idx = int(len(time) / 2)
time_mid = time - time[mid_idx]

#normalize y
signal_norm = signal_array / np.sum(signal_array)

signal = np.reshape(signal_norm, (-1, 1))

gmm = Gaussian(2, max_iter = 1000)
model = gmm.fit(signal, y=None)
print model.means_

X = gmm.predict(signal)
X_prob = gmm.predict_proba(signal)
params = gmm.get_params(deep=True)
score = gmm.score(signal)
score_samples = gmm.score_samples(signal)

plt.figure()
plt.title('data corrected')
plt.plot(time_mid, signal_norm)

plt.figure()
plt.title('data')
plt.plot(time, signal)

plt.figure()
plt.title('predict')
plt.plot(time, X, 'o')


plt.figure()
plt.title('predict_proba')
plt.plot(time, X_prob[:, 0], 'o')

#plt.figure()
#plt.title('score_samples')
#plt.plot(time, score_samples)


#print gmm.aic(signal)

#sample = gmm.sample(n_samples = 1000)
#plt.figure()
#plt.hist(sample[0])
#
#plt.figure()
#plt.hist(signal)

component_1 = signal_array * X_prob
plt.figure()
plt.plot(time, component_1)