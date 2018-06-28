#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 21:49:38 2018

@author: demiliu
"""

from peakaboo.data_smoothing import load_data_csv, load_data

def test_load_data_csv():
    file_1 = 1
    try:
        load_data_csv(file_1, 900, 1400, 1)
    except AssertionError:
        pass
    else:
        'TypeError not handled'
    
    file_2 = 'test.csv'
    try:
        load_data_csv(file_2, '1', 1400, 1)
    except TypeError:
        pass
    else:
        'TypeError not handled'
        
    try:
        load_data_csv(file_2, 700, '5', 1)
    except AssertionError:
        pass
    else:
        'TypeError not handled'
        
    try:
        load_data_csv(file_2, 700, 1400, '1')
    except TypeError:
        pass
    else:
        'TypeError not handled'
        
        
    try:
        load_data_csv(file_2, 700, 1400, 1)
    except AssertionError:
        pass
    else:
        'ValueError not handled'
    
    try:
        load_data_csv(file_2, 900, 1800, 1)
    except AssertionError:
        pass
    else:
        'ValueError not handled'

    try:
        load_data_csv(file_2, 900, 1400, -1)
    except AssertionError:
        pass
    else:
        'ValueError not handled'
    
    return


def test_load_data():
    file_1 = 1
    try:
        load_data(file_1, 900, 1400, 1)
    except AssertionError:
        pass
    else:
        'TypeError not handled'
    
    file_2 = 'test.txt'
    try:
        load_data(file_2, '1', 1400, 1)
    except TypeError:
        pass
    else:
        'TypeError not handled'
        
    try:
        load_data(file_2, 700, '5', 1)
    except AssertionError:
        pass
    else:
        'TypeError not handled'
        
    try:
        load_data(file_2, 700, 1400, '1')
    except TypeError:
        pass
    else:
        'TypeError not handled'
        
        
    try:
        load_data(file_2, 700, 1400, 1)
    except AssertionError:
        pass
    else:
        'ValueError not handled'
    
    try:
        load_data(file_2, 900, 1800, 1)
    except AssertionError:
        pass
    else:
        'ValueError not handled'

    try:
        load_data(file_2, 900, 1400, -1)
    except AssertionError:
        pass
    else:
        'ValueError not handled'
    
    return