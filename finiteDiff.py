# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 11:20:28 2023

@author: jaspi
"""

from scipy.optimize import root
import numpy as np
import matplotlib.pyplot as plt
from BVP import *
from ODE_functions import BVP2_exact, BVP1_exact



# python function for source term

def sourceTerm(N):
    test = np.ones(np.size(N-1))
    return test



def BVP(N, a, b, gamma1, gamma2, D, source):
    #create finite diff grid with N+1 points
    grid = finite_grid(N, a, b)
    dx = grid[0]
    x_int = grid[1]
    x = grid[2]
    
    #Create A matrix 
    A_DD = construct_A_matrix(N, D)
    
    #Contruct b vector
    b_DD = -construct_b_vector(N, a, b)
    
    #Call solver
    u_sol = solver(N,A_DD,b_DD,dx,x_int,source)
    
    return u_sol, x_int

def solver(N, A_DD, b_DD, dx, x_int, source):
    
    if source:
        b_DD = b_DD - dx**2 * sourceTerm(x_int)

        u = np.linalg.solve(A_DD,b_DD)
    else:
        u = np.linalg.solve(A_DD,b_DD)
    return u
    

def main():
    u_sol, x_int = BVP(N=50, a=0, b=1, gamma1=0, gamma2=1, D=1, source=False)
    u_true = BVP1_exact(x_int, a=0, b=1, gamma1=0, gamma2=1)

    #plot to compare solutions
    plt.plot(x_int, u_sol, 'o', label='numerical')
    plt.plot(x_int, u_true, 'k', label ='exact')
    plt.xlabel(f'$x$')
    plt.ylabel(f'$u(x)$')
    plt.legend()
    plt.show()
    
    #With source term included
    u_sol2, x_int2 = BVP(N=50, a=0, b=1, gamma1=0, gamma2=0, D=1, source=True)
    u_true2 = BVP2_exact(x_int, a=0, b=1, gamma1=0, gamma2=0, D=1)
    
    plt.plot(x_int2, u_sol2, 'o', label='numerical')
    plt.plot(x_int, u_true2, 'k', label ='exact')
    plt.xlabel(f'$x$')
    plt.ylabel(f'$u(x)$')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    main()
 




    