# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:20:28 2023

@author: jaspi
"""

from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
from BVP import dirichletBC, finite_grid, constructMatrix

#init params
N=50
a=0
b=1
D=1
gamma1 = 0
gamma2 = 1

# create a finite_dff grid with N+1 points
grid = finite_grid(N, a=0, b=1)

dx = grid[0]
x_int = grid[1]
x = grid[2]

#Create b_DD matrix using a,b boundary conditions
b_DD = dirichletBC(N, a=0, b=1)

#Create A_DD matrix 
a_DD = D * constructMatrix(N, D)

# python function for source term

def q(x):
    return np.ones(np.size(x))


#solve BVP
# def BVP_solver()
u = np.linalg.solve(a_DD, -D * b_DD - (dx**2 * q(x_int)))

def u_exact(x,a,b,alpha,beta, D, integer):
    
    answer = (-integer)/(2*D)*((x-a)*(x-b)) + ((beta-alpha)/(b-a))*(x-a)+alpha
    print(answer)
    return np.array(answer)
u_exact2 = u_exact(x_int, a=0, b=1, alpha=0, beta=0, D=1, integer=1)
# u_exact = -(1/(2*D))*(x-a)*(a-b)+((gamma2-gamma1)/(b-a))*(x-a)+gamma1

#plot to compare

plt.plot(x_int, u, 'o', label='numerical')
plt.plot(x_int, u_exact2, 'k', label ='exact')
plt.xlabel(f'$x$')
plt.ylabel(f'$u(x)$')
plt.legend()
plt.show()


# "Problem Params"

# #Number of grid points (minus one)
# N =20

# a = 0
# b = 1

# alpha = 0.0
# beta = 1.0

# # create grid
# x = np.linspace(a, b, N+1)
# dx = (b-a)/N

# #interior grid points
# x_int = x[1:-1]

# "Define python function"

# def dirichlet_solver(u, N, dx, alpha, beta):
#     F = np.zeros(N-1)
    
#     F[0] = (u[1]- 2*u[0] + alpha) / dx**2
    
#     for i in range(1, N-2):
#         F[i] = (u[i+1] - 2 * u[i] + u[i-1]) / dx**2
        
#     F[N-2] = (beta - 2 * u[N-2] + u[N-3]) / dx**2
    
#     return F

# def exactSol(x, a, b, alpha, beta):
    
#     Fexact = np.zeros(N-1)
    
#     for i in range(0,N-1):

#         Fexact[i] = ((beta-alpha)/(b-a))*(x[i]-a)+alpha
#     return Fexact

# "Numerical Sol with scipy root"

# #inital guess
# u_guess = 0.1 * x_int

# sol = root(dirichlet_solver, u_guess,  args = (N, dx, alpha, beta))
# print(sol.message, sol.x)

# #solution at interior points
# u_int = sol.x

# #exact solution
# solExact  = exactSol(x_int, a, b, alpha, beta)


# #plot numerical sol
# plt.plot(x_int, u_int, 'o', label="Numerical")

# #plot exact solution
# plt.plot(x_int,solExact, 'k', label ="Exact")

# #add labels

# plt.xlabel(f'$x$')
# plt.ylabel(f'$u(x)$')
# plt.legend()
# plt.show()




    