# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 15:02:56 2023

@author: Kasper
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from ODE_functions import predatorPrey
from ODE_solver import solveODE
from scipy.integrate import solve_ivp
from scipy import optimize



def shooting(u0,f,*args):

    # tspan = np.linspace(0, u0[-1], 100) 
    sol = solve_ivp(f, (0,u0[-1]), y0 = (u0[:-1]), args=args)
    print(sol)
    plt.plot(sol.y[0],sol.y[1])
    plt.show()
    shootingCond = u0[:-1] - sol.y[:,-1]
        
    # shootingCond = u0[:-1] - solveODE(f, u0[:-1], tspan, 'rk4', 0.01, True, *args)[-1]
    phaseCond = f(0,u0[:-1],*args)[0]
   
    return (*shootingCond, phaseCond)



def rootFinding(f,u0,*args):
    
    # u1 = optimize.root(shooting,u0,args=(f,*args))
    u1 = fsolve(lambda u0 : shooting(u0,f,*args), u0)
    return u1



f = predatorPrey
xGuess = 0.1
yGuess = 0.1
T = 2
u0 = [xGuess,yGuess,T]
"params = [a,b,d]"
params = [1,0.2,0.1]
result = rootFinding(f, u0, params)



# u0 = result
# shootingCond = u0[:-1] - solve_ivp(f, (0,u0[-1]), y0 = (u0[:-1]), args=(params,), rtol=0.0001).y[:,-1]
# phaseCond = f(0,u0[:-1],params)[0]
# print(shootingCond)
# print(phaseCond)


# point = (result.x)[:-1]
# plotShooting(u0,f,params)