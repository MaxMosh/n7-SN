#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 11:32:26 2020

@author: cantin
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as npl


# Definition de l'intervalle ]-R,R[ + mesh size
R = 3.0
N = 59
h = 2*R/(N+1)
X = np.linspace(-R,R,N)

# 1D Finite Difference Matrix
def Rig(N):
    A = np.zeros([N,N])
################
    A[0,0] = 2/h**2
    A[N,N] = 2/h**2
    A[0,1] = -1/h**2
    A[1,0] = -1/h**2
    A[N-1,N] = -1/h**2
    A[N,N-1] = -1/h**2
    for i in range(1,N-1):
        A[i, i-1] = -1/h**2
        A[i-1, i] = -1/h**2
        A[i,i] = 2/h**2
    return A

A = Rig(N)

eigval, eigvec = npl.eig(A)
idx = np.argsort(eigval)
eigval = eigval[idx]
eigvec = eigvec[:,idx]

# First Eigenvector
# Fixer le signe
U1 = eigvec[:,0]
if  U1[1]<0:
    U1 = - U1
    
# Second Eigenvector
# Norlaliser et fixer le signe
U2 = eigvec[:,1]
if  U2[1]<0:
    U2 = - U2 




# Plot Eigenvector and eigenvalues
################


# Potentiel trou
V0 = 10
a=1

################
        
        
    