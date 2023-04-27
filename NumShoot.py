# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
# from ODE_functions import predatorPrey
from ODE_solver import solveODE
from scipy.integrate import solve_ivp




def phaseCondition(f, u0, *args):
    """
    Determines phase condition
    Parameters
    ----------
    Returns
    -------
    """
  
    return np.asarray([f(0,u0,*args)[0]])


def shootingCondition(f, t_span,t_eval, u0, *args):
    """
    Parameters
    ----------
    Returns
    -------
    """

    sol = solveODE(f,u0,t_eval,'rk4',0.01,True,*args)
  
    return np.asarray([sol[-1][0] - u0[0]])
    
def shoot(f, t_span, t_eval, u0, *args):
    """
    Parameters
    ----------
    Returns
    -------
    """
    return np.concatenate((shootingCondition(f, t_span, t_eval, u0, *args), phaseCondition(f, u0, *args)))


def limitCycle(f, t_span, t_eval, u0, *args):
    """
    Parameters
    ----------
    Returns
    -------
    """
    sol = fsolve(lambda u0: shoot(f,t_span,t_eval,u0,*args),u0)
    return sol


# def predatorPrey(t, V, a, b, d):
#     x, y = V
#     xdash = x * (1 - x) - (a * x * y) / (d + x)
#     ydash = b * y * (1 - (y / x))
#     return [xdash, ydash]

def objective(V, a, b, d):
    x, y = V
    t_span = (0, 100)
    t_eval = np.linspace(t_span[0], t_span[1], 10000)
    # sol = solve_ivp(fun=predatorPrey, t_span=t_span, y0=[x, y], t_eval=t_eval, args=(a, b, d))
    # print(np.array([sol.y[0][-1] - sol.y[0][0], sol.y[1][-1] - sol.y[1][0]]))
    # return np.array([sol.y[0][-1] - sol.y[0][0], sol.y[1][-1] - sol.y[1][0]])
    f = predatorPrey
    sol = solveODE(f,u0,t_eval,'rk4',0.01,True,*args)
    print(np.array([sol.y[0][-1] - sol.y[0][0], sol.y[1][-1] - sol.y[1][0]]))
    return np.array([sol.y[0][-1] - sol.y[0][0], sol.y[1][-1] - sol.y[1][0]])
    
   

# Set the parameters
a = 1
b = 0.3
d = 0.1

# Find the root using fsolve
x_guess, y_guess = 0.1, 0.1
x_periodic, y_periodic = fsolve(objective, [x_guess, y_guess], args=(a, b, d))

# Print the result
print("Initial conditions for the periodic orbit: x0 = {}, y0 = {}".format(x_periodic, y_periodic))


def main():
    # Set initial conditions and parameters
    f = predatorPrey
    a = 1
    b = 0.2
    d = 0.1
    params = [a,b,d]
    x0 = 0.1
    y = 0.1
    u0 = [x0,y]
    t_span = (0, 100) # simulation time interval
    t_eval = np.linspace(t_span[0], t_span[1], 10000) # time points to return solution
    # solExact = solve_ivp(fun=predatorPrey, t_span=t_span, y0=[x0,y], t_eval=t_eval, args=([a, b, d],))
    # print(solExact.y[0][-1], solExact.y[1][-1])
    

    sol = limitCycle(f, t_span,t_eval, u0, params)
   
    # plt.plot(t_eval, sol.T[0], label="x")
    # plt.plot(t_eval, sol.T[1], label="y")
    # plt.xlabel("Time")
    # plt.ylabel("Population")
    # plt.legend()
    # plt.show()
    

# if __name__ == '__main__':
#     main()
    
    



