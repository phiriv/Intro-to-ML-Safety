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
    xp=np.shape(x)
    fs=np.zeros(max_iter)
    xs=np.zeros((xp[0],max_iter)) #col vecs of any size
    
    for t in range(max_iter):
        x=x.copy()-(step_size_fn(t)*df(x)*f(x))
        fs[t]=(f(x))
        xs[0,t]=x[0]
    
    return (x,fs,xs)

#alternative soln. (neater)
def gd2(f, df, x0, step_size_fn, max_iter):
    x=x0
    fs=[]
    xs=[]
    
    for t in range(max_iter):
        f2, df2 = f(x), df(x)
        fs.append(f2)
        xs.append(x)
        step=step_size_fn(t)
        x=x-step*df2
        return x, fs, xs

# TEST
def package_ans(gd_vals):
    x, fs, xs = gd_vals
    return [x.tolist(), [fs[0], fs[-1]], [xs[0].tolist(), xs[-1].tolist()]]

# Test case 1
ans=package_ans(gd2(f1, df1, cv([0.]), lambda i: 0.1, 1000))

# Test case 2
ans=package_ans(gd2(f2, df2, cv([0., 0.]), lambda i: 0.01, 1000))

#numerical gradient via finite differences
def num_grad(f, delt=0.001):
    
    def df(x):
        gv=np.zeros(np.shape(x))
        for i in range(np.size(x)):
            #compact but hard to read!
            gv[i,0]=((f(cv(x[i,0]+delt))-f(cv(x[i,0]-delt))/(2*delt))) 
            return gv.T
    
    return df

#alternate soln.
def num_grad2(f, delt=0.001):
    def df(x):
        g=np.zeros(x.shape)
        #more vars but much more legible
        for i in range(x.shape[0]):
            xi=x[i,0]
            x[i,0]=xi+delt
            fxp=f(x)
            x[i,0]=xi-delt
            fxm=f(x)
            xi=x[i,0]
            g[i,0]=(fxp-fxm)/(2*delt)
        return g
    return df

#TEST
x = cv([0.])
ans=(num_grad2(f1)(x).tolist(), x.tolist())

x = cv([0.1])
ans=(num_grad2(f1)(x).tolist(), x.tolist())

x = cv([0., 0.])
ans=(num_grad2(f2)(x).tolist(), x.tolist()) 
#IndexError?

x = cv([0.1, -0.1])
ans=(num_grad2(f2)(x).tolist(), x.tolist())

#use num_grad to find local minima
def minimize(f, x0, step_size_fn, max_iter):
    df=num_grad2(f)
    return gd(f, df, x0, step_size_fn, max_iter) #ICH LIEBE REKUR
    


ans = package_ans(minimize(f1, cv([0.]), lambda i: 0.1, 1000))

ans = package_ans(minimize(f2, cv([0., 0.]), lambda i: 0.01, 1000))

def hinge(v):
    #return ("HINGE OF HISTORY IS JAHMMED")
    #if (v<1):
     #   return (1-v)
    #else:
        #return 0
    return max(0,1-v)

# x is dxn, y is 1xn, th is dx1, th0 is 1x1
def hinge_loss(x, y, th, th0):
    return hinge(np.dot(y,(np.dot(x.T,th)+th0)))

# x is dxn, y is 1xn, th is dx1, th0 is 1x1, lam is a scalar
def svm_obj(x, y, th, th0, lam):
    J=0
    
    sub=0
    for i in range(np.size(x)):
        sub+=hinge_loss(x,y,th,th0)
    
    return J+lam*np.dot(th.T,th)

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

sep_e_separator = np.array([[-0.40338351], [1.1849563]]), np.array([[-2.26910091]])

# Test case 1
x_1, y_1 = super_simple_separable()
th1, th1_0 = sep_e_separator
ans = svm_obj(x_1, y_1, th1, th1_0, .1)

# Test case 2
ans = svm_obj(x_1, y_1, th1, th1_0, 0.0)
