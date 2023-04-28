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
from ODE_functions import predatorPrey




def rootFinding(f, u0, *args):
  
    """
    Parameters
    ----------
    
    Returns
    -------
    
    """
    if callable(f):
        # orbit = fsolve(lambda U, f: numericalShoot(U, f, *args), u0, f)
        orbit =  root(lambda U, f: numericalShoot(U, f, *args), u0, f)
       
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
    sol = solveODE(f, u0, [0, T], 'rk4', 0.01, True,*args)
    
    shootCondition = (u0 - sol[-1])
    G = np.append((shootCondition), phaseCondition)
    return G

def plotToCompare(f, U, *args):
    
   
    u0 = U[:-1]
    # T = U[-1]
    solve_ivp
    evals = np.linspace(0,100,1000)
    integ_solve = solve_ivp(f, (0,100), u0, method='RK45', t_eval=evals)
    plt.plot(integ_solve.y[0],integ_solve.y[1])
    
    # plt.plot(evals,integ_solve.y[0],label="x")
    # plt.plot(evals,integ_solve.y[1],label="y")
    plt.legend()
    plt.show()
    
    


def main():
    f = predatorPrey
    u0 = [1,1,10]
    params = [0.1,0.2,1]

    # plotToCompare(f,u0, params)
    result = rootFinding(f, u0, params).x # Print the result
    print("Initial conditions for the limit cycle: x0 = {}, y0 = {}".format(round(result[0],4), round(result[1],4)))
    print("Period of the limit cycle: T = {}".format(round(result[2],2)))
 

if __name__ == '__main__':
    main()
    
