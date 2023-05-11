# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 19:23:28 2023

@author: Kasper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.integrate import solve_ivp
from scipy import optimize
from ODE_solver import solveODE




def rootFinding(f, U, *args):
  
    """
    Parameters
    ----------
    
    Returns
    -------
    
    """
    if callable(f):
        # orbit = fsolve(lambda U, f: numericalShoot(U, f, *args), u0, f)
        orbit =  root(lambda U, f: numericalShoot(U, f, args=args), U, f)
       
    else:
        raise Exception("Input function %f is not a function" % f)
    return orbit


def numericalShoot(U, f, *args):
    """
    Parameters
    ----------

    Returns
    -------
  
    """
    if isinstance(U, np.ndarray):
        u0 = U[:-1]
        T = U[-1]
    else:
        raise Exception("Initial conditions %U must be a list" % U)
        
        
    phaseCondition = np.array([f(0, u0, *args)[0]])

    # sol = solve_ivp(f, (0,T), u0)
    sol = solveODE(f, u0, [0, T], 'rk4', 0.01, True,args=args)
    
    shootCondition = (u0 - sol[-1])
    G = np.append((shootCondition), phaseCondition)
    return G

def plotToCompare(f, U, *args):
    
   
    u0 = U[:-1]
    print(u0,"u0")
    T = U[-1]
    print("T",T)
    evals = np.linspace(0,100,1000)
    integ_solve = solve_ivp(f, (0,100), u0, t_eval=evals, args=args)
    # plt.plot(integ_solve.y[0],integ_solve.y[1])
    
    plt.plot(evals,integ_solve.y[0],label="x")
    plt.plot(evals,integ_solve.y[1],label="y")
    plt.legend()
    plt.show()
    
    
# def chemicalReaction(t, V, *args):
#     k1, k2, k3 = args[0], args[1], args[2]
#     x, y, z = V[0], V[1], V[2]
#     xdash = -k1*x + k2*y*z
#     ydash = k1*x - k2*y*z - k3*y**2
#     zdash = k3*y**2
#     V = np.array([xdash, ydash, zdash])
#     return V

def main():
    """
    x0 = 1.0
    y0 = 0.5
    z0 = 1.5
    k1 = 0.2
    k2 = 0.7
    k3 = 0.1
    """
    
    f = predatorPrey
    u0 = [1,0.5,1.5,10]
    params = [0.2,0.7,0.1]
    plotToCompare(f,u0, params)
    
    # f = predatorPrey
    # u0 = [1,1,10]
    # params = [0.1,0.2,1]

    
    # result = rootFinding(f, u0, params).x # Print the result
    # print("Initial conditions for the limit cycle: x0 = {}, y0 = {}".format(round(result[0],4), round(result[1],4)))
    # print("Period of the limit cycle: T = {}".format(round(result[2],2)))
 

if __name__ == '__main__':
    main()
    
