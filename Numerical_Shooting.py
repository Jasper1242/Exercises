# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 15:51:16 2023

@author: jaspi
"""
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from ODE_functions import predatorPrey
from ODE_solver import solveODE

def phaseCondition(f, u0, *args):
    """
    Determines intial phase condition
    Parameters
    ----------
    Returns
    -------
    """
  
    return f(u0,0,*args)[0]



def shootingConds(f, u0, pc, *args):
    """
    Determines intial phase condition
    Parameters
    ----------
    Returns
    -------
    """
    x0 = u0[0]
    t = u0[0]
    tspan = np.linspace(0, t, 100) 
    sol = solveODE(f, u0, tspan, 'rk4', 0.01, True, *args)
    xCond = x0 - sol[:-1]
    tCond = np.asanyarray(pc(f, u0, *args)) 

    G = np.concatenate((xCond, tCond),axis=None)
    return G
    
  
    
    
    
# def orbit()
#     """
    
    
#     """

f = predatorPrey
u0 = [1,1]
params = [1,0.11,0.1]
a = 1
b = 0.11
d = 0.1
pc = phaseCondition

result = shootingConds(f,u0,pc,params)