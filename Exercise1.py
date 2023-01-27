# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 11:15:50 2023

@author: Kasper
"""




def leibniz(n):
    t_sum =0
    for i in range(n):
        term = (8 / ((4*i+1)*(4*i+3)))
        t_sum = t_sum + term
    
    return t_sum



#1a
n= 100
print(leibniz(n), ',n = 100')

n= 1000
print(leibniz(n), ',n = 1000')

n = 10000
print(leibniz(n), ',n = 10000','\n\n')


#1b

import math

#absoulute error
pie = math.pi

n = 100
print('abs error when n = 100 is: ',abs(pie-leibniz(n)))

n = 1000
print('abs error when n = 1000 is: ',abs(pie-leibniz(n)))

n= 10000
print('abs error when n = 10000 is: ',abs(pie-leibniz(n)))



print(10**-7)

n= 400
while pie-leibniz(n) > 10**-3:
    n+= 1
    print(n)
    
print(n)


#%%


