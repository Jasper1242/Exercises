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


def construct_b_vector(N, a, b):
    """Takes boundary conditions a and b and constructs a N-1 lenght vector,
    where a represents the first boundary and b the last
    
    Params
    ------
    'N': Int
    
    'a': Int
    
    'b': Int
    
    Returns
    -------
    'V': np.ndarray
    """
    V = np.zeros((N-1),)
    V[0] = a
    V[-1] = b
    return V

def construct_A_matrix(N,D):

    #Initialise empty matrix
    M = np.zeros(((N-1),(N-1)))
    #Fill diagonal elements with -2
    np.fill_diagonal(M, -2)
    #Set off diagonal elements to 1
    for i in range((N-1)):
        for j in range((N-1)):
            if i == j-1 or i==j+1:
                M[i][j]= 1
    return M

