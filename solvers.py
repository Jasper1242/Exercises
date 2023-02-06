# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 17:15:36 2023

@author: Kasper
"""

def iteration(x,f,fdash):
    
    xn = x
    xnew  = xn - (f(xn)/fdash(xn))
        
    return xnew

