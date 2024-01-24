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


xmin = 0.0; xmax = 1.8; nptx = 16; nx = nptx-2  
hx = (xmax-xmin)/(nptx -1)
xx = np.linspace(xmin,xmax,nptx) 
xx = xx.transpose()
xxint = xx[1:nx+1]
ymin = 0.0; ymax = 1.0; npty = 16; ny = npty-2 
hy = (ymax-ymin)/(npty -1)
yy = np.linspace(ymin,ymax,npty)
yy=yy.transpose() 
yyint = yy[1:ny+1]

#  Matrix system
# On Ox
Kx # Local matrix of size Nx+2 relative to Ox discretization
K2Dx # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2

# On Oy
Ky # Local matrix of size Nx+2 relative to Oy discretization
K2Dy # Global Matrix of (Ny+2)**2 matrices of size (Nx+2)**2
#
#
K2D = K2Dx + K2Dy # Final matrix of Laplacien operator with Dirichlet Boundary conditions
#
##  Solution and source terms
u = np.zeros((nx+2)*(ny+2)) #Numerical solution
u_ex = np.zeros((nx+2)*(ny+2)) #Exact solution
F = np.zeros((nx+2)*(ny+2)) #Source term
#
#
# Source term
def Source_int(x):
    return 2*np.pi**2*(np.sin(np.pi*x[0])*np.sin(np.pi*x[1]))
def Source_bnd(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
def Sol_sin(x):
    return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])
for i in range(nptx):
    for j in range(npty):
        coord = np.array([i*hx,j*hy])
        u_ex[j*(nx+2) + i] = Sol_sin(coord)
    if i==0 or i==nptx-1: # Boundary x=0 ou x=xmax
        for j in range(npty):
            coord = np.array([i*hx,j*hy])
            F[j*(nx+2) + i]=Source_bnd(coord)
    else:
        for j in range(npty):
            coord = np.array([i*hx,j*hy])
            if j==0 or j==npty-1: # Boundary y=0 ou y=ymax
                F[j*(nx+2) + i]=Source_bnd(coord)
            else:
                F[j*(nx+2) + i]=Source_int(coord)




#        
#Post-traintement u_ex+Visualization of the exct solution
uu_ex = np.reshape(u_ex,(nx+2 ,ny+2),order = 'F');
X,Y = np.meshgrid(xx,yy)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,uu_ex.T,rstride = 1, cstride = 1);

#        
u = npl.solve(K2D,F)
uu = np.reshape(u,(nx+2 ,ny+2),order = 'F');
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,uu.T,rstride = 1, cstride = 1);


# Error
print('norm L2 = ',npl.norm(u-u_ex))