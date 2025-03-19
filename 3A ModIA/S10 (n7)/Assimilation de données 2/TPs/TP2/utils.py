import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace
import os 
from dolfin import *
#from fenics import *
import pandas as pd #data frame


#
#
#
def create_mesh(param_mesh) : 
  """
  generate a 1D regular mesh & define the BC
  Parameters :
  param_mesh : dict, contains the set of parameters necessary for the mesh construction 
  Returns :
  the mesh object, fonctions that locates the boundary conditions 
  """
  mesh = IntervalMesh(param_mesh['number cells']-1,0,param_mesh['length river'])

  def boundary_up(x, on_boundary):
    return on_boundary and near(x[0],0.0)
  def boundary_down(x, on_boundary):
    return on_boundary and near(x[0],param_mesh['length river'])

  return mesh, boundary_up, boundary_down 


#
# Convert np array 2 FEniCS and reciprocally
#

def array_to_FEfct(V,x) : 
  """
  convert a numpy array to a FEniCS function
  Parameters : 
  V : FEniCS FE space
  x : array to turn into a function 
  Returns : 
  the FEniCS function corresponding to the array 
  """
  xfunc = Function(FunctionSpace(V.mesh(),'CG',1))
  xfunc.vector().vec()[:]=x[::-1]
  res = Function(V)
  res = interpolate(xfunc,V) 
  return res



def FEfct_to_array(mesh,fct):

  """
  convert a FEniCS function to a numpy array
  Parameters : 
  mesh : mesh object, mesh of the geometry 
  fct  : FEniCS function to turn into an array 
  Returns : 
  the array containing the value of the function at the nodes 
  """
  x=mesh.coordinates()[:,0]
  return np.array([fct(pt) for pt in x])

