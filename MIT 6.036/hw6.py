# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:30:46 2022

@author: phiri
"""

import numpy as np

#implementation of soft(arg)max for P1
#z is a kx1 col. vec.
def softmax(z):
    return (np.exp(z)/np.sum(np.exp(z)))

a=np.array([-1,0,1]).T
b=softmax(a)

#Problem 2

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


#Problem 3

# layer 1 weights
w_1 = np.array([[1, 0, -1, 0], [0, 1, 0, -1]])
w_1_bias = np.array([[-1, -1, -1, -1]]).T
# layer 2 weights
w_2 = np.array([[1, -1], [1, -1], [1, -1], [1, -1]])
w_2_bias = np.array([[0, 2]]).T

z = np.dot(w_1.T, w_1_bias)
a = softmax(z)
g = np.dot(x, (a - y).T)


T  = np.matrix([[0.0 , 0.1 , 0.9 , 0.0],
[0.9 , 0.1 , 0.0 , 0.0],
[0.0 , 0.0 , 0.1 , 0.9],
[0.9 , 0.0 , 0.0 , 0.1]])
g = 0.9
r = np.matrix([0, 1., 0., 2.]).reshape(4, 1)

print(np.linalg.solve(np.eye(4) - g * T, r))