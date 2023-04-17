# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:28:09 2023

@author: Kasper
"""
import numpy as np
import matplotlib.pyplot as plt 

def main(method):
    """
    :f: Function defining an ODE or system of ODEs
    :N: The lowest order of h to calculate the error of (if N = 5, h = 10^-5)
    :x0: Starting x value(s)
    :t0: Starting time value
    """
    
    N = 5
    x0 = 1
    t0 = 0
  
    xError, deltaTlist = [] , []
    for deltaTmax in np.logspace(-N, -1, N):
        
        x1 = solveTo(method,deltaTmax, x0, t0)
        
        xErr.append(x1)
        deltaTlist.append(deltaTmax)
        
        
    plottingError(xError, deltaTlist, method)
    
    
    

def f(x,t):
    
    return x


    
def eulerStep(t0,x0,h):
    print("Euler")
    m = f(x0,t0)
    tN = t0 + h
    xN  = x0 + h * m
    return tN, xN

def RK4Step(t0,x0,h):
    print("RK")
    k1 = h * (f(x0, t0))
    k2 = h * (f((x0+h/2), (t0+(k1/2))))
    k3 = h * (f((x0+h/2), (t0+(k2/2))))
    k4 = h * (f((x0+h), (t0+k3)))
    k = (k1+2*k2+2*k3+k4)/6    
        
    tN = t0 + h
    xN = x0 + k
    
    return tN, xN

    
def solveTo(method,deltaTmax,x0,t0):
    
    h = deltaTmax
    x, t = t0, x0
    
    if method == "euler":
        x, t = eulerStep(t, x, h)
    elif method == "RK4":
        x,t = RK4Step(t ,x, h)
    else:
        print("Please input valid method!")
    
    return x
    
    
def plottingError(xError, deltaTlist,method):
    
    """
    This function creates a log-log graph of the error for the method when solving the function dx/dt = x
  
    """
    
  
        
    plt.loglog(deltaTlist, xError, label=('{} Method').format(method))
    plt.ylabel('|$x_{n}- x(t_{n})$|')
    plt.xlabel('h')
    plt.title('Order of error for RK4')
    plt.legend()
    plt.show()
    
main("RK4")
main("euler")