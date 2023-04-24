# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:48:54 2023

@author: jaspi
"""
import numpy as np 


def f(x,t):
    # Function for ODE eqiuation dx/dt=x
    
    """
    Simulates the ODE x' = x
    
    Parameters
    ----------
    t  : float
         Time
    x  : array
         x-value at time t
    Returns
    -------
    x : ODE value
    """
        
    return x

def fAnalytical(t):
    x = np.exp(t)
    return x


def g(V,t):
    """ 
    Function for 2nd order ODE x'' = -x,
    reduced to a system of 1st order equations equivalent to,
    dx/dt = y , dy/dt = -x
    
    Parameters
    ----------
    V  : array
         Contains x and y value at given t
    t  : Float
         Time
         
    Returns
    -------
    V : ODE values for dy/dt, dx/dt
    
    """
    x = V[0]
    y = V[1]
    xDash = y
    yDash = -x
    V = np.array([xDash, yDash])
    
    return V

def gAnalytical(t):
    x = np.sin(t) + np.cos(t)
    y = np.cos(t) - np.sin(t)
    V = np.array([x,y])
    return V


def cubic(x, *args):
    c = args[0]
    return x ** 3 - x + c
