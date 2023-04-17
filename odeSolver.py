# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:19:40 2023

@author: Kasper
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import time

def f(x,t):
    # Function for ODE eqiuation dx/dt=x
    return x

def fNumerical(t):
    x = np.exp(t)
    return x


def g(v,t, **args):
    # Function for 2nd order ODE reduced to system of 1st order
    
    x = v[0]
    y = v[1]
    xDash = y
    yDash = -x
    V = np.array([xDash, yDash])
    
    return V

def GNumerical(t):
    
    x = np.sin(t) + np.cos(t)
    y = np.cos(t) - np.sin(t)
    V = np.array([x,y])
    return V

def eulerStep(x0, t, h, f, **args):
    
    """
    
    Params
    
    'x0':
    
    't':
        
    'h':
    
    'f':
        
    **args:
    """
    
    dxdt = f(x0, t, args)
    xN = x0 + h * dxdt
    tN = t + h
    return xN, tN

def RK4Step(x0, t, h, f, **args):
    """
    Params
    
    'x0':
    
    't':
        
    'h':
    
    'f':
        
    **args:
    """
     
    k1 = np.array( f(x0, t, args) )
    k2 = np.array( f(x0+h*(k1/2), t+(h/2), args) )
    k3 = np.array( f(x0+h*(k2/2), t+(h/2), args ) )
    k4 = np.array( f(x0+h*k3, t+h, args ) )
    
    xN = x0 + (1/6) * h * (k1 + (2*k2) + (2*k3) + k4)
    tN = t + h
    return xN, tN


def solveTo(T, x0, f, method, order, **args):
    
    """
    Params 
   
    'T':
    
    'x0':
    
    'f':
        
    'method':
    
    'order':
        
    '**args':  
    """
    
    h = args(deltaT_max)
    t0 = T[0]
    tend = T[-1]
    
    
    # Create solution array considering 2nd or 1st order
    
  
    
    while t0 + h < tend:
        x0, t0 = method(f, x0, t0, h)
        solArray.append(x0)
    else:
        x0, t0 = method(f, x0, tend -t0, h)
    
    return x0


def main():
    # N = 5
    # t = np.logspace(-N, -1, 2**N)
    
    t = np.linspace(0,10,100)
    for i in range(len(t)-1):
        xi = ()
        
        
        
    
    
    
    