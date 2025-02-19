#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Affinely parametrized linear BVP:
     - div( lambda(mu) * grad(u) ) + w * grad(u) = f  in domain
                                       u = g  on bdry dirichlet
                         - lambda(mu) nabla(u).n = 0 on bdry Neumann
with w: given velocity field

Single input parameter: mu (the diffusivity coeff.)
    
Goal: Solve this BVP by an offline-online strategy based on a POD.
 
'''

from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import eigh
import time
import random
import numpy.linalg as npl
import scipy
import scipy.linalg   
import math
from mpl_toolkits.mplot3d import axes3d
#

# The PDE parameter: diffusivity lambda(mu)
def Lambda(mu):
#    return  mu + mu0 # affine case
    return np.exp(mu0*(mu+1.)) # non-affine case

# Function to compute the RB dimension (= Nrb)
def energy_number(epsilon_POD,lam):
    # lam: eigenvalues table
    # return the eignvalue number corresponding to energy_ratio
    index_min = 0; s = 0.;s1=np.sum(lam)
    for i in range(len(lam)):
        if s < s1*(1-epsilon_POD):
            s += lam[i]
            index_min = index_min + 1
    return index_min

# Dirichlet boundary conditions
tol_bc = 1.e-10
def u_bdry_0(x, on_boundary):
    return bool(on_boundary and (near(x[0], 0, tol_bc)))
def u_bdry_1(x, on_boundary):
    return bool(on_boundary and (near(x[0], 1, tol_bc)))

###################################################
#    Offline phase
###################################################

# Physical and numerical parameters
# Mesh and function spaces
NP =  50; print('Number of mesh points NP = ', NP)
mesh = UnitSquareMesh(NP,NP)
k = 2 ; print('Order of the Lagrange FE k = ', k)
V = FunctionSpace(mesh, "CG", int(k))
V_vec = VectorFunctionSpace(mesh, "CG", int(k))
NN = V.dim(); print('Resulting number of nodes NN = ', NN)
coordinates = mesh.coordinates()
# Trial and test function
u, v = TrialFunction(V), TestFunction(V)

# Snapshots number
print('How many snapshots do I compute ? ')
M = int(input())

# The parameter range mu
# The input parameter mu_0
mu0 = 0.7
# The input parameter mu
mu_min = 1.0; mu_max = 10. # range of values 
print('Range values for mu: [',mu_min,',',mu_max,']')
mu = np.linspace(mu_min,mu_max,M)

# Plot of the parameter space
Param =  np.zeros(len(mu))
for i in range(len(mu)):
    Param[i] = Lambda(mu[i])
print("Param=",Param)
fig = plt.figure()
ax = fig.gca() 
ax.scatter(mu, Param) 
plt.title("The parameters space")
ax.set_xlabel('The physical parameter mu')
ax.set_ylabel('Lambda(mu)')
plt.legend()
plt.show()

# Velocity field
vel_amp = 1e+2; print('vel_amp =',vel_amp)
vel_exp = Expression(('(1.+abs(cos(2*pi*x[0])))', 'sin(2*pi/0.2*x[0])'), element = V.ufl_element())
#vel_exp = Expression(('0.', '0.'), element = V.ufl_element())
vel = vel_amp * interpolate(vel_exp,V_vec)
#p=plot(vel,title='The velocity field')
#p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

# Peclet number for max value of lambda(mu)
mu_max = Lambda(np.max(mu))
Pe_min = sqrt(dot(vel,vel))/(mu_max)
Pe_func_min = project(Pe_min, V)
Pe_vec_min = Pe_func_min.vector().get_local()
print("min_Pe_min",min(Pe_vec_min))
print("max_Pe_min",max(Pe_vec_min))
p=plot(Pe_func_min.leaf_node(), title="The max Peclet number")
p.set_cmap("rainbow")# ou 'viridis
plt.colorbar(p); plt.show()

# Peclet number for min value of lambda(mu)
mu_min = Lambda(np.min(mu))
Pe_max = sqrt(dot(vel,vel))/(mu_min)
Pe_func_max = project(Pe_max, V)
Pe_vec_max = Pe_func_max.vector().get_local()
print("min_Pe_max",min(Pe_vec_max))
print("max_Pe_max",max(Pe_vec_max))
p=plot(Pe_func_max.leaf_node(), title="The min Peclet number")
p.set_cmap("rainbow")# ou 'viridis
plt.colorbar(p); plt.show()


# Peclet number: mean value
#Pe_mean =  (Pe_min+Pe_max)/2.
#Pe_func_mean = project(Pe_mean, V)
#Pe_vec_mean = Pe_func_mean.vector().get_local()
#print("min_Pe_mean",min(Pe_vec_mean))
#print("max_Pe_mean",max(Pe_vec_mean))
#p=plot(Pe_func_max.leaf_node(), title="The mean Peclet number")
#p.set_cmap("rainbow")# ou 'viridis
#plt.colorbar(p); plt.show()


# RHS of the PDE model
f_exp = Expression('1E+03 * exp( -( abs(x[0]-0.5) + abs(x[1]-0.5) ) / 0.1 )', element = V.ufl_element()) # Gaussian
f = interpolate(f_exp,V)

print('#')
print('# Computation of the M snapshots')
print('#')
Usnap = np.zeros((M,NN)) # Snaphots matrix
uh = np.zeros(NN)
t_0 =  time.time()
for m in range(M):
    print('snapshot #',m,' : mu = ',mu[m])
    diffus = Lambda(mu[m])
    print('snapshot #',m,' : Lambda(mu) = ',diffus)
    # Variational formulation
    F = diffus * dot(grad(v),grad(u)) * dx + v * dot(vel, grad(u)) * dx - f * v * dx
    # Stabilization of the advection term by SUPG 
    r = - diffus * div( grad(u) ) + dot(vel, grad(u)) - f #residual
    vnorm = sqrt( dot(vel, vel) )
    h = MaxCellEdgeLength(mesh)
    delta = h / (2.0*vnorm)
    F += delta * dot(vel, grad(v)) * r * dx
    # Create bilinear and linear forms
    a = lhs(F); L = rhs(F)
    # Dirichlet boundary conditions
    u_diri0_exp = Expression('1.', degree=u.ufl_element().degree())
    bc0 = DirichletBC(V, u_diri0_exp, u_bdry_0)
    # Solve the problem
    u_mu = Function(V)
    solve(a == L, u_mu,bc0)
    # Buid up the snapshots matrices U 
    uh = u_mu.vector().get_local()[:]
    Usnap[m, :] = uh # dim. M x NN

# Plot of the manifold in 3D
# TO BE COMPLETED ...
selected_dims = [1,2,3]
# On plot les 3 premières dimensions
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(Usnap[:,selected_dims[0]], Usnap[:,selected_dims[1]], Usnap[:,selected_dims[2]], c='r', marker='o')
ax.set_xlabel('dim 1')
ax.set_ylabel('dim 2')
ax.set_zlabel('dim 3')
plt.title("The manifold in 3D")
plt.show()

# Transpose the snapshots matrix to be of size NN x M
Usnap = Usnap.transpose()

# Assemble of the rigidity matrix (FE problem in V_h)
f_exp1 = Expression('0.0', element = V.ufl_element())
f1 = interpolate(f_exp1,V)
u1, v1 = TrialFunction(V), TestFunction(V)
F1 =  dot(grad(v1),grad(u1)) * dx +  u1 * v1 * dx + f1 * v1 * dx
a1 = lhs(F1); L1 = rhs(F1)
# Assemble & get the matrix NN x NN
A_ass1, F1 = assemble_system(a1, L1)
# For L2 norm we have:
A_NN1 = np.identity(NN)

####################  POD  method ###################
# Computation of the correlation matrix for L2 norm
# TO BE COMPLETED ...
C = np.dot(Usnap.T,Usnap)
# On affiche la matrice de corrélation C
fig = plt.figure()
ax = fig.gca()
cax = ax.matshow(C)
fig.colorbar(cax)

# Solve the eigenvalue problem C.w = lambda.w
# TO BE COMPLETED ...
eigen_val, eigen_vec = np.linalg.eigh(C)
# Sorting the eigenvalues in decreasing order
idx = eigen_val.argsort()[::-1]
eigen_val = eigen_val[idx]
# DEBUG
print("DEBUG eigen_val=",eigen_val)
# FIN  DEBUG
eigen_vec = eigen_vec[:,idx]

# Computation of the left singular vector from the eigenvectors of C
# TO BE COMPLETED ...
left_sing_vec = Usnap@eigen_vec / np.sqrt(eigen_val)

# Normalization of the eigenvalues 
# TO BE COMPLETED ...
eigen_val_normalized = eigen_val / np.sum(eigen_val)


# Plot of the eigenvalues
decay = np.arange(len(eigen_val))
fig = plt.figure()
ax = fig.gca() 
ax.plot(decay, abs(eigen_val), label='Eigenvalues',color='r') 
plt.title("The decay of the eigen values")
ax.set_xlabel('The eigenvalues index')
ax.set_ylabel('The eigen values')
#plt.xscale("log")
plt.yscale("log")
plt.legend()
# Plot in bar chart
width = 0.5
p =plt.bar(decay,eigen_val, width, color='b');
plt.title('The M eigenvalues');plt.ylabel('Eigenvalues');
plt.show()

# Tolerance epsilon to determine the number of modes Nrb
print('Give a tolerance to compute Nrb')
epsilon_POD = float(input())

# Computation of the number of modes Nrb
Nrb = energy_number(epsilon_POD,eigen_val_normalized)
# DEBUG
print("DEBUG Nrb=",Nrb)
# FIN  DEBUG
print('This corresponds to Nrb = ',Nrb)

# Truncation of the Reduced Basis 
# TO BE COMPLETED ...
t_1 =  time.time()
# The reduced basis matrix
B_rb = left_sing_vec[:,:Nrb]

# The error estimation satisfied by the POD method
# TO BE COMPLETED ...
s_eigen = np.sum(eigen_val)
sum = np.sum(eigen_val[:Nrb])
discarded_eigenvalues = np.sum(eigen_val[Nrb:])
print("sum=",sum)
print("The discarde eigenvalues",discarded_eigenvalues)
print("The error estimation by POD method is",abs(sum- discarded_eigenvalues)/abs(s_eigen))


##################################################
#         Online phase
##################################################
print('#'); print('# Online phase begins... #')
#
## New parameter value mu (must be within the same intervall [mu_min,mu_max])
print('Choose a new value of the parameter mu (within the same intervall [mu_min,mu_max])')
mu = float(input())

# Diffusivity parameter
diffus = Lambda(mu)

print('   You will get the RB solution for mu = ',mu)
print('   This corresponds to lambda(mu) = ',diffus)

# Assemble the rigidity matrix...
# Variational formulation
u, v = TrialFunction(V), TestFunction(V)
F = diffus * dot(grad(v),grad(u)) * dx + v * dot(vel, grad(u)) * dx - f * v * dx
# SUPG stabilisation
r = - diffus * div( grad(u) ) + dot(vel, grad(u)) - f # Residual
vnorm = sqrt( dot(vel, vel) )
h = MaxCellEdgeLength(mesh); delta = h / (2.0*vnorm)
F += delta * dot(vel, grad(v)) * r * dx
# Create bilinear and linear forms
a = lhs(F); L = rhs(F)
u_diri0_exp = Expression('1.', degree=u.ufl_element().degree())
bc0 = DirichletBC(V, u_diri0_exp, u_bdry_0)
# Assemble and get the matrix NN x NN plus the RHS vector NN
A_ass, F = assemble_system(a, L, bc0)
A_NN = A_ass.array()

# Stiffness matrix & RHS of the reduced system
# The reduced stiffness matrix: Brb^T A_NN Brb
# TO BE COMPLETED ...
reduced_stiffness_matrix = B_rb.T@A_NN@B_rb
# The reduced RHS
# TO BE COMPLETED ...
reduced_RHS = B_rb.T@F

# Solve the reduced system
# TO BE COMPLETED ...
Urb = np.linalg.solve(reduced_stiffness_matrix, reduced_RHS)
t_2 =  time.time()
#print('RB solution CPU-time = ',tcpu1 - tcpu2)
print('RB solution CPU-time = ',t_1 - t_2)

#
# Difference ("error") between the HR FE solution and the RB solution
#
# The RB solution in the complete FE basis: Urb = B_rb^T . urb 
# TO BE COMPLETED ...
Urb = np.dot(B_rb, Urb)

# Transform the RB solution to a Fenics object
Urb_V = Function(V)
Urb_V.vector().set_local(Urb)

#
# Computation of the current HR FE solution 
print('Compute the complete FE solution to be compared with uRB !')
# By following the usual way, it would give:
#uh = Function(V)
#solve(a == L, uh,bc0)
#Uh = uh.vector().get_local() # npy vector 
# Compute the HR FE solution by solving the FE system A_NN . Uh = F
# This enables to compare the CPU time
tcpu3 = time.time()
Uh = np.linalg.solve(A_NN,F)
tcpu4 = time.time()
print('FE solution CPU-time = ',tcpu3 - tcpu4)
# The relative diff. vector & its FEniCS object
error_vec = abs(Uh-Urb)/abs(Uh)
error_func = Function(V); error_func.vector().set_local(error_vec)

# Plot of the FE, RB and relative error functions 
# FE solution
uh = Function(V)
uh.vector().set_local(Uh)
p=plot(uh, title="The FE solution")#,mode='color',vmin=1.0, vmax=1.0000040)
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

# RB solution
p=plot(Urb_V, title="The Reduced Basis solution")#,mode='color',vmin=1.0, vmax=1.0000040)
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

# Relative difference solution
p=plot(error_func,title="The relative diff ",mode='color')#,vmin=0.0, vmax=0.024)
p.set_cmap("rainbow"); plt.colorbar(p); plt.show()

# Computation of the relative errors in L2, 2-norm and norm max 
# Relative diff. ("error") in norm max
error_norm_max = np.linalg.norm(Uh-Urb,ord=np.inf)
# Norm max of HR FE solution
norm_max_Uh = np.linalg.norm(Uh,ord=np.inf)
# Relative error in norm max
error_relative_norm_max = (error_norm_max)/(norm_max_Uh)
print('Relative diff. in norm max = ',error_relative_norm_max)

# Relative diff. in norm 2
error_norm2 = np.linalg.norm(Uh-Urb)
# Norm 2 of HR FE solution
norm2_Uh = np.linalg.norm(Uh)
# Relative error in norm 2
error_relative_norm2 = (error_norm2)/(norm2_Uh)
print('Relative diff. in norm 2 = ',error_relative_norm2)

# Relative diff. in L2 norm first method using the energy norm
# Error in L2 norm or in L2 if A_NN = I_NN
error_L2 = np.sqrt(np.dot((Uh-Urb),np.dot(A_NN1,Uh-Urb)))
# H1 norm of HR FE solution
L2_Uh = np.sqrt(np.dot(Uh,np.dot(A_NN1,Uh)))
# Relative error in H1 norm
error_relative_L2_norm = error_L2/L2_Uh
print("Relative diff H1 norm=",error_relative_L2_norm)

# Relative diff. in L2 norm second method (Fenics norm)
# Function to stor the diff between HR FE and RB solutions
diff = Function(V)
# Get the corresponding vectors
diff.vector().set_local(Uh-Urb)
# Error in H1 norm using FEniCS
error_L2_fenics = norm(diff, 'L2', mesh)
# H1 norm of HR FE solution using FEniCS
L2_Uh_fenics = norm(uh, 'L2', mesh)
print('#')
print('#')
print('#')

#
# Print out performances of the method
print('The POD method outputs are: ')
print('#')
print('The CPU time of the offline phase is:',t_1-t_0)
print('#')
print('FE solution CPU-time = ',tcpu4 - tcpu3,'seconds')
print('#')
#print('RB solution CPU-time = ',tcpu2 - tcpu1,'seconds')
print('RB solution CPU-time = ',t_2 - t_1,'seconds')
print('#')
print("Nrb=",Nrb)
print('#')
print('Relative diff. in norm max = ',error_relative_norm_max)
print('#')
print('Relative diff. in norm 2 = ',error_relative_norm2)
print('#')
print("Relative diff L2 norm=",error_relative_L2_norm)
print('#')
print("FEniCS L2 Relative norm is=",error_L2_fenics/L2_Uh_fenics)
print('#')
print("The discarde eigenvalues",discarded_eigenvalues)
print('#')
print("sum=",sum)
print('#')
print("The error estimation by POD method with method1 is",abs(sum- discarded_eigenvalues)/abs(s_eigen))