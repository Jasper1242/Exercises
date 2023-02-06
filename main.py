# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:35:25 2023

@author: Kasper
"""
#Q8
import solvers
import math
from scipy import optimize
import scipy

x=-1

def f(x):
    return math.cos(x)-x

def fdash(x):
    return -math.sin(x)-1

for i in range(10):
    
    x = solvers.iteration(x,f,fdash)
    print(x)

#%%
#c
def fun(x):
    return math.cos(x)-x

root =  scipy.optimize.root(fun,1)
print(root)