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

#1c


print(10**-7)

n= 400
while pie-leibniz(n) > 10**-3:
    n+= 1
    print(n)
    
print(n)


#%%

#Q2a

import numpy as np

aV = [1,2,3]
bV = [6,5,4]

def dotProduct(a,b):
    product = np.dot(a,b)
    return product

def dotProd(a,b):
    product = sum([a[i][0]*b[i] for i in range(len(b))] )
    return product

print('dot product of a and b : ',dotProduct(aV,bV))


#2b

A = [[1,2],[3,4]]
B = [[5,6],[7,8]]

def matrixProduct(A,B):
    try:
        product = np.dot(A,B)
        return product
    
    except:
        print("Vector or Matrices don't hacve consistant sizes")

print(matrixProduct(A,B))

    