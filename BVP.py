# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 12:37:01 2023

@author: jaspi
"""

import numpy as np 

def finite_grid(N, a, b):
 
    x = np.linspace(a,b,N+1)
    dx = (b-a)/N
    x_int = x[1:-1]
    return dx, x_int, x


def dirichletBC(N, a, b):
    V = np.zeros((N-1),)
    V[0] = a
    V[-1] = b
    return V

def constructMatrix(N,D):

    #Initialise empty matrix
    A_D = np.zeros(((N-1),(N-1)))
    #Fill diagonal with -2
    np.fill_diagonal(A_D, -2)
    #Set off diagonal equal to 1
    for i in range((N-1)):
        for j in range((N-1)):
            if i == j-1 or i==j+1:
                A_D[i][j]= 1
    return A_D*D
