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
    return np.array([xDash, yDash])

def gAnalytical(t):
    x = np.sin(t) + np.cos(t)
    y = np.cos(t) - np.sin(t)
    return np.array([x,y])


def predatorPrey(t,V,*args):
    # print("t",t,"V",V,"args",args)
    # args = args[0]
    # a,b,d = args[0],args[1],args[2]
    a = 1
    b = 0.2
    d = 0.1
    x,y = V[0], V[1]
    xdash = x*(1-x) - (a*x*y)/(d+x)
    ydash = b*y*(1-(y/x))
    return np.array([xdash,ydash])



def hopfBifur(t, V, *args):
    """
    Function to simulate the Hopf bifurcation normal form
    
    Parameters
    ----------
    t  : float
         Time
    V  : array
         x-value at time t
    b  : float
         value of constant 'b'
    Returns
    -------
    Array of du1/dt and du2/dt
    """

    beta, sigma = args[0]
    u1, u2 = V
    du1 = beta * u1 - u2 + sigma * u1 * (u1 ** 2 + u2 ** 2)
    du2 = u1 + beta * u2 + sigma * u2 * (u1 ** 2 + u2 ** 2)
    return np.array([du1, du2])



def hopfExplicit(t, theta, b):
    """
    Function to simulate the Hopf bifurcation explicit solution
    
    Parameters
    ----------
    t      : float
              Time
    theta  : float
              phase
    b      : float
              value of constant 'b'
    Returns
    -------
    Array of u1 and u2
    """
    u1 = np.sqrt(b) * np.cos(t + theta)
    u2 = np.sqrt(b) * np.sin(t + theta)
    return np.array([u1, u2])


def cubic(x, *args):
    c = args[0]
    return x ** 3 - x + c
