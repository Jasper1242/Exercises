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
  
    return f(u0[0:2],0,*args)[0]



def shootingConds(f, u0, pc, *args):
    """
    Determines intial phase condition
    Parameters
    ----------
    Returns
    -------
    """
    x0 = u0[0:2]
    t = u0[-1]
    tspan = np.linspace(0, t, 100) 
    sol = solveODE(f, x0, tspan, 'rk4', 0.01, True, *args)#
    # solver = odeint(derive,u0,t,args=(a,b,d))
    # print(sol,"sol")
    xCond = (x0 - sol)[:-1]
    print(len(xCond))
    # print(xCond,"xCond")
    tCond = np.asanyarray(pc(f, u0, *args)) 
    # print(tCond,"tCond")

    G = np.concatenate((xCond, tCond),axis=None)
    print(len(G))
    return G
    
  
def shooting(f, u0, pc, *args):
    """
    Determines intial phase condition
    Parameters
    ----------
    Returns
    -------
    """
    finalSol =  fsolve(lambda u0 : shootingConds(f,u0,pc,*args), u0)
    return finalSol
    
f = predatorPrey
u0 = [1,1,0]
params = [1,0.11,0.1]
a = 1
b = 0.11
d = 0.1
pc = phaseCondition

result = shooting(f,u0,pc,params)