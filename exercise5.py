# -*- coding: utf-8 -*-
"""
Created on Fri Feb 24 12:15:18 2023

@author: Kasper
"""

from scipy.optimize import fsolve, root
import numpy as np
from numerical_shooting import shooting


def cubic(x, args):
    args[0] = c
    return x ** 3 - x + c


    
def nat_par_continuation(f, u0, paras, paraRange, paraVary, discretisation=shooting, solver=fsolve, phase_cond='none'):

    paraList = np.linspace(paraRange[0], paraRange[1], paraVary)
    
    solList = []
    
    for para in paraList:
        
        paras[paraVary] = para
        
        sol = np.array(solver(discretisation(f), u0, args= paras))
        solList.append(sol)
        
        u0 = sol
        
    return paraList, np.array(sol_list)