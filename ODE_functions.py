# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 12:48:54 2023

@author: jaspi
"""
import numpy as np 


def f(t,x):
    """
    Function for ODE eqiuation dx/dt=x
    
    Parameters    
    ----------
    t (float): Time
    x (float): Current value of the variable x
    
    
    Returns
    -------
    The value of the derivative dx/dt, which is simply x.
    """
    return x

def fAnalytical(t):
    """
    This function calculates the analytical solution of the ordinary 
    differential equation dx/dt = x using an exponential function.
    
    Parameters    
    ----------
    t (float): Time
    
    Returns
    -------
    The calculated value of x at time t, where x = exp(t).
    """
    x = np.exp(t)
    return x


def g(t,V):
    """ 
    Function for 2nd order ODE x'' = -x,
    reduced to a system of 1st order equations equivalent to,
    dx/dt = y , dy/dt = -x
    Parameters    
    ----------
    t (float): Time
    V (array): An array containing the current state of the system, 
    with two elements representing x and y.
    
    Returns
    -------
    An array of derivatives representing the rates of change of x and y over time.
    """
    x = V[0]
    y = V[1]
    xDash = y
    yDash = -x
    return np.array([xDash, yDash])

def gAnalytical(t):
    """
    Analytical solution to the 2nd order ode
    Parameters    
    ----------
    t (float): Time

    Returns
    -------
    Array containing [x, y]
    """
    x = np.sin(t) + np.cos(t)
    y = np.cos(t) - np.sin(t)
    return np.array([x,y])


def predatorPrey(t,V,*args):
    """Function to suimulate predator prey dyncamics
    
    Parameters
    ----------
    t (float): Time
    V (array): An array containing the current state of the system, 
    with two elements representing x and y.
    *args: Additional arguments .
    Returns
    -------
    Array of xdash, ydash
    
    """
    
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

def BVP1_exact(x,a,b,gamma1,gamma2):
    """
    
    This function calculates the exact solution for a boundary 
    value problem with simpler conditions.
    
    Parameters
    ----------
    x (float or array): Independent variable(s)
    a (float): Left boundary value
    b (float): Right boundary value
    gamma1 (float): Value of the solution at x = a
    gamma2 (float): Value of the solution at x = b
    
    Returns
    -------
    Array of the exact solution evaluated at the given x value(s).

    """
    exactSol = ((gamma2-gamma1)/(b-a))*(x-a)+gamma1
    return np.array(exactSol)

def BVP2_exact(x,a,b,gamma1,gamma2, D):
    """
    This function calculates the exact solution for a boundary value problem 
    with more complex conditions, possibly involving a diffusion-like term.
    
    Parameters
    ----------
    x (float or array): Independent variable(s)
    a (float): Left boundary value
    b (float): Right boundary value
    gamma1 (float): Value of the solution at x = a
    gamma2 (float): Value of the solution at x = b
    D (float): Coefficient associated with the diffusion term
    
    
    Returns
    -------
    Array of the exact solution evaluated at the given x value(s).
    """
    exactSol = (-1)/(2*D)*((x-a)*(x-b)) + ((gamma2-gamma1)/(b-a))*(x-a)+gamma1
    return np.array(exactSol)

def cubic(x, *args):
    """
    This function calculates the value of a cubic polynomial 
    with an additional constant term.
    
    Parameters
    ----------
    x (float): The independent variable
    *args (tuple): Additional arguments, expected to contain a single constant c
    
    Returns
    -------
    The value of the cubic polynomial evaluated at the given x value.
    """
    c = args[0]
    return x ** 3 - x + c
