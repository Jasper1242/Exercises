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

C = [[1],[2],[3]]
D = [[1,2],[3,4]]

def matrixProduct(A,B):
    try:
        product = np.dot(A,B)
        return product
    
    except:
        print("Vectors or Matrices don't have consistant sizes")

print(matrixProduct(A,B))
print(matrixProduct(C,D))

#%%
#Q3a

myArray = np.array([5,4,9,2,0,4,7,2])

print(myArray[-1], " - Last entry")

print(myArray[1:6],"- Index 1-6")

print(myArray[:-2], " - up until second last entry")

print(myArray[::2], " - Every second entry" )

myArray[-1] = -9

print(myArray)
    
myArray[0:3] = 1

print(myArray)

#%%
#Q4a
import numpy as np
import random

r = np.random.uniform(1,9,20)
print(r)

idx = r<5
print(idx)

r[idx]=0
print(r)

#%%
#Q5a

A = np.array([[1,0,0,0],[1,-2,1,0],[0,1,-2,1],[0,0,0,1]])
b = np.array([0,1,1,2])
x = np.linalg.solve(A,b)
print(x)

#check solution 

print(np.allclose(np.dot(A,x),b))

#%%
#Q6a
# =============================================================================
# import numpy as np
# 
# 
# t = np.linspace(0, 5, num=500)
# 
# def maths2(t):
#     y = (t**2)*np.exp(-2*t)
#     return y
#  
# y=np.array([])
# 
# for i in t:
#     y = np.append(y, maths2(i))
#     
# =============================================================================
#Faster way
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0, 5, num=500)

def maths2(t):
    return (t**2)*np.exp(-2*t)

y = np.array(list(map(maths2,t)))



plt.plot(t,y)
plt.title('y agaisnt t')
plt.xlabel('Time')
plt.ylabel('y(t)')
plt.show()


indexMax = np.argmax(y)
print(indexMax)

#%%
#Q7a

def pseudoInv(A):
    Astar = np.dot(np.linalg.inv((np.dot(A.transpose(),A))),A.transpose())
    return Astar

#b
A = np.array([[1,2,1,3],[-1,4,1,8]]).transpose()

AstarA = np.dot(pseudoInv(A),A).round()


#c
#Identity multiplied x is x so,

Astar = pseudoInv(A)
b = np.array([2,4,6,8]).transpose()
x = np.linalg.solve(Astar,b)

#%%
#Q8a


    
