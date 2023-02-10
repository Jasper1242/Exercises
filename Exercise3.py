# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:29:20 2023

@author: Kasper
"""
#%%
#q1 using odeint

import random
import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt
import matplotlib.cm as cm


a = 1
b = 0.11
d = 0.1
x0 = 1
y0 = 1
tMax = 100
t0=0
t= np.linspace(0,100,1000)


def derive(XY,t,a,b,d):
    x,y = XY
    xdash = x*(1-x) - (a*x*y)/(d+x)
    ydash = b*y*(1-(y/x))
    return np.array([xdash,ydash])
  

XY = [x0,y0]

solver = odeint(derive,XY,t,args=(a,b,d))

x,y = solver.T

                        

plt.figure()
plt.grid()
plt.title("odeint method")
plt.plot(t, x, 'xb', label = 'Badgers')
plt.plot(t, y, '+r', label = "Crows")
plt.xlabel('Time t, [days]')
plt.ylabel('Population')
plt.legend()

plt.show()

"""
When b > 0.26
Model converges to equlibrium becomes stable 

When b < 0.26
converges to a stable limit cycle


"""

b = np.arange(0.1, 0.49, 0.1)

nums=np.random.random((10,len(b)))
colors = cm.rainbow(np.linspace(0, 1, nums.shape[0]))  # generate the colors for each data set

fig, ax = plt.subplots(2,1)

for b, i in zip(b, range(len(b))):
    solver = odeint(derive, XY, t, args = (a,b,d))
    ax[0].plot(t, solver[:,0], color = colors[i],  linestyle = '-', label = r"$\beta = $" + "{0:.2f}".format(b))
    ax[1].plot(t, solver[:,1], color = colors[i], linestyle = '-', label = r" $\beta = $" + "{0:.2f}".format(b))
    ax[0].legend()
    ax[1].legend()

ax[0].grid()
ax[1].grid()
ax[0].set_xlabel('Time t, [days]')
ax[0].set_ylabel('Crows')
ax[1].set_xlabel('Time t, [days]')
ax[1].set_ylabel('Badgers');

#%%



import numpy as np
from scipy.integrate import odeint 
import matplotlib.pyplot as plt

a = 1
b = 0.2
d = 0.1
x0 = 1
y0 = 1
tMax = 100
t0=0
t= np.linspace(0,100,1000)


def derive(XY,t,a,b,d):
    x,y = XY
    xdash = x*(1-x) - (a*x*y)/(d+x)
    ydash = b*y*(1-(y/x))
    return np.array([xdash,ydash])


plt.figure()
IC = np.linspace(1.0, 6.0, 21) # initial conditions for crow population (prey)
for crow in IC:
    X0 = [crow, 1.0]
    Xs = odeint(derive,XY,t,args=(a,b,d))
    plt.plot(Xs[:,0], Xs[:,1], "-", label = "$x_0 =$"+str(X0[0]))
plt.xlabel("Crows")
plt.ylabel("Badgers")
plt.legend()
plt.title("Badgers Vs Crows");

#%%
