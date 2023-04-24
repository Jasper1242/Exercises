# -*- coding: utf-8 -*-
"""
Created on Sun Apr 16 16:19:40 2023

@author: Kasper
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import time
import ODE_functions



def eulerStep(f, x0, t0, h):
    """ 
    Function implements a euler step for (t0 + h)
    
    Parameters
    ----------
    f  : Function
         ODE function that returns value of x
    x0 : Float
         Inital value for x
    t0 : Float
         Inital value for time
    h  : Float
         Step size
    Returns
    -------
    [xN, tN] : Array values for new step h
     
    """
    
    dxdt = f(x0, t0)
    xN = x0 + h * dxdt
    tN = t0 + h
    return xN, tN

def rk4Step(f, x0, t0, h):
    
    """ 
    Performs Runge-Kutta Four step for (t0 + h)
    Parameters
    ----------
     f  : Function
          ODE function that returns value of x
     x0 : Float
          Inital value for x
     t0 : Float
          Inital value for time
     h  : Float
          Step size
    Returns
    -------
    [xN, tN] : Array of values for new step h
     
    """  
    k1 = np.array(f(x0, t0))
    k2 = np.array(f(x0 + h * (k1/2), t0 + (h/2)))
    k3 = np.array(f(x0 + h * (k2/2), t0 + (h/2)))
    k4 = np.array(f(x0 + h * k3, t0 + h))
    
    xN = x0 + ((k1 + 2*k2 + 2*k3 + k4)/6) * h
    tN = t0 + h

    return xN, tN


def solveTo(f, T, x0, h, method):
    
    """
    Solve ode using given method between range of T
    
    Parameters
    ----------
   
    'T' :   Array
            Contains two values t0, the intial value time and
            tend, the final of time
    'x0' :  Array
            Inital x value to solve for
    'f' :   Function 
            ODE to be solved 
    'deltaTmax' : Float
                  Maximum step size
    'method' : String
               Describes which solver is to be used
               
    Returns
    -------
   
    """

    methods = {'euler':eulerStep, 'rk4':rk4Step}
    solver = methods[method]
    
    tN = T[0] 
    tend = T[-1]
   
   
    xN = x0
    solArray = []
    while (tN + h) < tend:
        xN, tN = solver(f, xN, tN, h)
        solArray.append(xN)
    else:
        xN, tN = solver(f, xN, tN, tend - tN)
        solArray.append(xN)

    return xN
    


def solveODE(f, x0, tspan, method, deltaTmax, order):
    """
    Solve ode using given method between range of T
    
    Params 
   
    'f' :   
           
    'x0' :  
           
    'tspan' :  
              
    'method' : 
        
    'deltaTmax' : 
              
    'order' : 

    Returns
    -------

    """


    if order:
       solArray = np.empty(shape=(len(tspan), len(x0)))
    else:
       solArray = np.empty(shape=(len(tspan), 1))
    solArray[0] = x0
 
    for i in range(len(tspan)-1):
        solArray[i+1] = solveTo(f, ([tspan[i],tspan[i+1]]), solArray[i], deltaTmax, method)
    
    return solArray
        
def main():
    """
    Example for solutions to first order ODE x' = x 
    with inital conditions; x(0) = 1 
    solving for t = 0 till t = 1
    """
    f = ODE_functions.f
    fTrue = ODE_functions.fAnalytical
    tspan = np.linspace(0, 1, 100) 
    
    eulerSol = solveODE(f, 1, tspan, 'euler', 0.01, False)
    rk4Sol = solveODE(f, 1, tspan, 'rk4', 0.01, False)
    trueSol = fTrue(tspan)
 

    
    rk4Error = [np.abs(trueSol[i] - rk4Sol[i]) for i in range(len(trueSol))]
    eulerError = [np.abs(trueSol[i] - eulerSol[i]) for i in range(len(trueSol))]
  
    
    
    plt.loglog(tspan, eulerError, label='Euler', marker='x', markersize=10)
    plt.loglog(tspan, rk4Error, label='RK4', marker='o',  markersize=3)
    # plt.loglog(tspan, trueSol, label='True')
    plt.legend()
    plt.xlabel('t')
    plt.ylabel('x')
    
    # # print(eulerSol)
    # plt.plot(tspan, eulerTrue)
    # plt.xlabel('Time')
    # plt.ylabel('Error')
    # plt.title('Error between exact and numerical solutions')
    # plt.show()
        

if __name__ == '__main__':
    main()
    
    
    
    