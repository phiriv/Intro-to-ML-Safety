# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:30:46 2022

@author: phiri
"""

import numpy as np

#implementation of soft(arg)max
#z is a kx1 col. vec.
def softmax(z):
    return (np.exp(z)/np.sum(np.exp(z)))

a=np.array([-1,0,1]).T
b=softmax(a)

w = np.array([[1, -1, -2], [-1, 2, 1]])
x = np.array([[1], [1]])
y = np.array([[0, 1, 0]]).T

gradWL=np.dot(x.T,(np.subtract(w,y.T)))#wrong dims!

z = np.dot(w.T, x)
a = softmax(z)
g = np.dot(x, (a - y).T)

w2 = w - 0.5 * g
print(w2)

z = np.dot(w2.T, x)
a = softmax(z)
g = np.dot(x, (a - y).T)