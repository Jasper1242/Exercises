# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:20:28 2023

@author: jaspi
"""

from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt





# ans = exactSol(gamma1, gamma2, a, b, x)

"Problem Params"

#Number of grid points (minus one)
N =20

a = 0
b = 1

alpha = 0.0
beta = 1.0

# create grid
x = np.linspace(a, b, N+1)
dx = (b-a)/N

#interior grid points
x_int = x[1:-1]

"Define python function"

def dirichlet_solver(u, N, dx, alpha, beta):
    F = np.zeros(N-1)
    
    F[0] = (u[1]- 2*u[0] + alpha) / dx**2
    
    for i in range(1, N-2):
        F[i] = (u[i+1] - 2 * u[i] + u[i-1]) / dx**2
        
    F[N-2] = (beta - 2 * u[N-2] + u[N-3]) / dx**2
    
    return F

def exactSol(x, a, b, alpha, beta):
    
    Fexact = np.zeros(N-1)
    
    for i in range(0,N-1):

        Fexact[i] = ((beta-alpha)/(b-a))*(x[i]-a)+alpha
    return Fexact

"Numerical Sol with scipy root"

#inital guess
u_guess = 0.1 * x_int

sol = root(dirichlet_solver, u_guess,  args = (N, dx, alpha, beta))
print(sol.message, sol.x)

#solution at interior points
u_int = sol.x

#exact solution
solExact  = exactSol(x_int, a, b, alpha, beta)


#plot numerical sol
plt.plot(x_int, u_int, 'o', label="Numerical")

#plot exact solution
plt.plot(x_int,solExact, 'k', label ="Exact")

#add labels

plt.xlabel(f'$x$')
plt.ylabel(f'$u(x)$')
plt.legend()
plt.show()




    