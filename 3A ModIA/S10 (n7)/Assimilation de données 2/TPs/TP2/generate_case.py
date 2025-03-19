import numpy as np
import matplotlib.pyplot as plt
import sys 	
import os
import glob
import numpy.random as npr
import pandas as pd
import shutil

from dolfin import *
from fenics import *
from class_vda import * 

set_log_level(50) # to minimize automatic FeniCS comments

#
#
#
def generate_case(path):
  """
  generate a river geometry  + define the Dirichlet BC
  (the direct model is here not solved: no solution H is computed)
  output files: "bathy_t.npy", "background.npy", dataframe; all saved in path
  Parameters: 
  path : str, path to the folder that contains the files
  Returns: none
  """
  
  ### Parameters MAY BE CHANGED ###
  L = 100.e+03 # (m)
  npts = 1001 # number of grid points
  # bathymetry
  slope = 1.e-03
  # water depth (this defines the Dirichlet BC at upstream & downstream)
  href = 10.0 #(m)
  n_wave_bathy = 3	
  amp_wave_bathy = href/5.
  omega = 2*np.pi/L
  ################
  if not os.path.isdir("./"+path):
    os.makedirs("./"+path)

  # 1d regular mesh
  deltax = float(L)/(npts-1)
  x = np.linspace(0.,L,npts)

  #
  # PART TO BE CHANGED
  #
  # bathymetry shape (at x-scale)
  b_t_0 = slope * (L - x) # flat bottom 
  b_t = b_t_0 + amp_wave_bathy * np.cos(n_wave_bathy * omega * x)
  b_t += amp_wave_bathy/2. * np.cos(2*n_wave_bathy * omega * x) + amp_wave_bathy/2. * np.cos(3*n_wave_bathy * omega * x)
  b_t += amp_wave_bathy/3. * np.random.uniform(-1.,1.,len(b_t_0)) # add noise
  
  #b_t_max = np.max(b_t); b_t_min = np.min(b_t); b_t_mean = np.mean(b_t)
  
  print("generate_case: definition of a background value b_b (which may be used in the regularization term...)")
  b_b = np.linspace(b_t[0],b_t[len(x)-1],npts) -href # - 2.*href/3
  #b_b = np.linspace(b_t[0],b_t[-1],npts) - np.abs( np.linspace(b_t[0],b_t[-1],npts)[np.around(npts/2)] - b_t[np.around(npts/2)] )
  
  # Dirichlet BC values are defined here  # MAY BE CHANGED
  H_in = L * slope + href
  H_out = href  
  #
  # END PART TO BE CHANGED
  #
  
  # dataframe
  d = {"L":[L,0.],"NP":[npts,0],"href":href,"BC":[H_in,H_out]}
  dfr = pd.DataFrame(data=d)

  # save in files
  file_dfr=path+'case.csv'
  dfr.to_csv(file_dfr)
  
  np.save(path+'bathy_t.npy',b_t)
  np.save(path+'background.npy',b_b)
  
#
# Define Href 
#
def define_Href(path, new='False'):
  """
  Define Href either as a simple linear function 
  or from a value H_t (= model output already performed for the same bathymetry).
  This may be a trick to consider a model closer to the non-linear one...
  Result written in file "Href.npy"
  Parameters: 
  path : str, path containing the result
  new : optional, boolean . Is the first time this routine is called ? 
  Returns: none
  """
  # read data of the case
  file_dfr=path+'case.csv'
  dfr = pd.read_csv(file_dfr)
  npts = dfr["NP"][0]
  H_up, H_down = dfr["BC"][0], dfr["BC"][1]
    
  # Default value of Href if not existing yet (value required to perform a 1st time the direct model !)
  if new==True:
    print("define_Href: Href value is first defined as a constant slope")
    Href_np = np.linspace(H_up,H_down,npts)
    np.save(path+"Href.npy",Href_np)
  else:
    print("define_Href: do we update Href as the model ouput H(b)? (<=> one fixed point iteration)")
    ans = input("       ...(type y or any other key)...")
    if ans=="y":
      print("define_Href: Href value has been updated to the previously computed model output value")
      Href_np = np.load(path+'H_t.npy') # read a previous model run output
      # save in file
      np.save(path+'Href.npy',Href_np)
      