#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 18:34:48 2021

@author: cantin
"""

import numpy as np
import numpy.linalg as npl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#  Discrétisation en espace


xmin = 0.0; xmax = 2; nptx = 61; nx = nptx-2  
hx = (xmax-xmin)/(nptx -1)
xx = np.linspace(xmin,xmax,nptx) 
xx = xx.transpose()
xxint = xx[1:nx+1]
ymin = 0.0; ymax = 1.0; npty = 31; ny = npty-2 
hy = (ymax-ymin)/(npty -1)
yy = np.linspace(ymin,ymax,npty)
yy=yy.transpose() 
yyint = yy[1:ny+1]


# =============================================================================
### Parameters
mu = 0.01 # Diffusion parameter
vx = 1 # Vitesse along x
# =============================================================================

cfl = 0.2  # cfl =mu*dt/hx^2+mu*dt/hy^2 ou v*dt/h
dt = (hx**2)*(hy**2)*cfl/(mu*(hx**2 + hy**2)) # dt = pas de temps
dt = cfl*hx/vx
Tfinal = 0.8   # Temps final souhaitÃ©



###### Matrice de Diffusion Dir/Neumann



#### Matrice de Convection  (Centré)



#### Global matrix : diffusion + convection
A2D = -(K2D + V2Dx) #-mu*Delta + V.grad
#
#
##  Cas explicite
u = np.zeros((nx+2)*(ny+2))
u_ex = np.zeros((nx+2)*(ny+2))
err = np.zeros((nx+2)*(ny+2))
F = np.zeros((nx+2)*(ny+2))
#
#
# =============================================================================
# Time stepping
# =============================================================================
s0 = 0.1
x0 = 0.25
y0=0.5

def Sol_init(x):
    return np.exp( -((x[0]-x0)/s0)**2 -((x[1]-y0)/s0)**2   )



u_init = np.zeros((nx+2)*(ny+2))
for i in range(nptx):
     for j in range(npty):
             coord = np.array([xmin+i*hx,ymin+j*hy])
             u_init[j*(nx+2) + i] = Sol_init(coord)
             
             
uu_init = np.reshape(u_init,(nx+2 ,ny+2),order = 'F');
fig = plt.figure(figsize=(10, 7))
X,Y = np.meshgrid(xx,yy)
ax = plt.axes(projection='3d')
surf = ax.plot_surface(X, Y, uu_init.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
ax.view_init(60, 35)
plt.pause(1.)
             
## Initialize u by the initial data u0
u = u_init.copy()

# Nombre de pas de temps effectues
nt = int(Tfinal/dt)
Tfinal = nt*dt # on corrige le temps final (si Tfinal/dt n'est pas entier)

# Time loop
for n in range(1,nt+1):

  # Schéma explicite en temps


 # Print solution
    if n%5 == 0:
      plt.figure(1)
      plt.clf()
      fig = plt.figure(figsize=(10, 7))
      ax = plt.axes(projection='3d')
      uu = np.reshape(u,(nx+2 ,ny+2),order = 'F');
      surf = ax.plot_surface(X, Y, uu.T, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
      ax.view_init(60, 35)
      plt.title(['Schema explicite avec CFL=%s' %(cfl), '$t=$%s' %(n*dt)])
      plt.pause(0.1)

####################################################################
# comparaison solution exacte avec solution numerique au temps final
j0 = int((npty-1)/2)


plt.figure(2)
plt.clf()
x = np.linspace(xmin,xmax,nptx)
plt.plot(x,uu_init[:,j0],x,uu[:,j0],'k') #,x,uexacte,'or')
plt.legend(['Solution initiale','Schema explicite =%s' %(cfl)]) #,'solution exacte'],loc='best')
plt.show()

