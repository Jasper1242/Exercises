# -*- coding: utf-8 -*-
"""
Created on Sat Apr 29 18:54:29 2023

@author: Kasper
"""

from ODE_functions import hopfBifur
from ODE_functions import hopfExplicit
from Numerical_Shooting import plotToCompare
#set variables 

sigma = -1
u1 = 1
u2 = 1
beta = 1
T = 10
f1 = hopfBifur
f2 = hopfExplicit
U = u1,u2,T
args = beta, sigma


plotToCompare(f1, U, args)