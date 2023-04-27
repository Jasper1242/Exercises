# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:48:54 2023

@author: jaspi
"""
import numpy as np 


def f(t,x):
    """Function for ODE eqiuation dx/dt=x
    """
    return x

def fAnalytical(t):
    x = np.exp(t)
    return x


def g(t,V):
    """ 
    Function for 2nd order ODE x'' = -x,
    reduced to a system of 1st order equations equivalent to,
    dx/dt = y , dy/dt = -x
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


def predatorPrey(t,V,*args):
    args = args[0]
    a,b,d = args[0],args[1],args[2]
    x,y = V[0], V[1]
    xdash = x*(1-x) - (a*x*y)/(d+x)
    ydash = b*y*(1-(y/x))
    V = np.array([xdash,ydash])
    return V


def cubic(x, *args):
    c = args[0]
    return x ** 3 - x + c
