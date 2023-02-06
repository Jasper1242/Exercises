# -*- coding: utf-8 -*-
"""
Created on Mon Feb  6 16:28:09 2023

@author: Kasper
"""
import numpy as np
import matplotlib.pyplot as plt 

def main():
    plottingError(5, 1, 0)
    

def f(x,t):
    return x
    
def eulerStep(t0,x0,h):

    m = f(x0,t0)
    tNew  = t0 + h
    xNew  = x0 + h* m
    return tNew, xNew
    
    
def solveTo(deltaTmax,x0,t0):
    
    h = deltaTmax
    x, t = t0, x0
    x, t = eulerStep(t, x, h)
    
    return x
    
    
def plottingError(N,x0,t0):
    
    """
    This function creates a log-log graph of the error for the euler method when solving the function dx/dt = x
    :f: Function defining an ODE or system of ODEs
    :N: The lowest order of h to calculate the error of (if N = 5, h = 10^-5)
    :x0: Starting x value(s)
    :t0: Starting time value
    :t1: Final time value
    """
    
    
  
    xError, deltaTlist = [] , []
    for deltaTmax in np.logspace(-N, -1, 2*N):
        x1 = solveTo(deltaTmax, x0, t0)
        xError.append(abs(np.exp(1) - x1))
        deltaTlist.append(deltaTmax)
        
        

    plt.loglog(deltaTlist, xError, label='Euler Method')
    plt.ylabel('|$x_{n}- x(t_{n})$|')
    plt.xlabel('h')
    plt.title('Order of error for Euler')
    plt.legend()
    plt.show()
    
main()