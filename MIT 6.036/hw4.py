# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:45:53 2022

@author: phiriv
"""

import numpy as np

#utilities
def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))

def f1(x):
    return float((2 * x + 3)**2)

def df1(x):
    return 2 * 2 * (2 * x + 3)

def f2(v):
    x = float(v[0]); y = float(v[1])
    return (x - 2.) * (x - 3.) * (x + 3.) * (x + 1.) + (x + y -1)**2

def df2(v):
    x = float(v[0]); y = float(v[1])
    return cv([(-3. + x) * (-2. + x) * (1. + x) + \
               (-3. + x) * (-2. + x) * (3. + x) + \
               (-3. + x) * (1. + x) * (3. + x) + \
               (-2. + x) * (1. + x) * (3. + x) + \
               2 * (-1. + x + y),
               2 * (-1. + x + y)])
        
#da moony
#simple gradient descent
def gd(f, df, x0, step_size_fn, max_iter):
    x=x0
    fs=np.zeros(max_iter)
    xs=np.zeros(max_iter)
    
    for t in range(max_iter):
        x=x.copy()-(step_size_fn(t)*df(x)*f(x))
        fs.append(f(x))
        xs.append(x)
    
    return (x,fs,xs)

