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
from gradient_test import *
from generate_case import *
from plots import *

set_log_level(50) # to minimize automatic FeniCS comments
  
#
# Direct model
# 
def run_direct(path) : 

    """
    Run the direct model (stationnary linearized one): compute H for the test case saved in path
    Write result in file path+'H_t.npy' 
    -----------
    Parameters :
    path : str, path of the data to be used 
    Returns: None
    """

    # FE type
    te, deg = 'CG', 2 
    param_elements = {'type of elements': te, 'degree of freedom': deg}
    
    # read data of the case
    file_dfr=path+'case.csv'
    dfr = pd.read_csv(file_dfr)
    L = dfr["L"][0]; nb_cell = dfr["NP"][0]
    param_mesh = {'number cells':nb_cell,'length river':L}

    # bathy & Href read in files
    bathy_np = np.load(path+'bathy_t.npy')
    Href_np = np.load(path+'Href.npy')
    
    param_geom = {'bathymetry array': bathy_np, 'Href array': Href_np}

    if not( nb_cell == np.size(bathy_np) ) or not( nb_cell == np.size(Href_np) ) :
        sys.exit('run direct: dimensions mismatch')

    # Dirichlet bc values
    H_up, H_down = dfr["BC"][0], dfr["BC"][1]
    param_bc = {'H bc upstream': H_up,'H bc downstream': H_down}
    
    # create a vda_river object (without paramVDA)
    direct_vda = vda_river('linear_stationnary',param_mesh, param_elements, param_geom, param_bc, {})
    # convert in FEnicCS format
    bathy = array_to_FEfct(direct_vda.V,bathy_np)
    Href = array_to_FEfct(direct_vda.V,Href_np)
    
    # solve the direct model
    H = direct_vda.Direct_Model(bathy)

    # write in file
    print("run_direct: write the direct model output in file ", path+'H_t.npy')
    H_t_np = FEfct_to_array(direct_vda.mesh, H)
    np.save(path+'H_t.npy', H_t_np)   
  
  
#
# Generate observations
#
def generate_obs(path) : 
    """
    Generate observations from the model output H(b) 
    Write the result Hobs in file but also the potentialy new value of H_t (due to a potential change of Href)
    -----------
    Parameters :
    path : str, path of data to be used     
    """
    # basic data of the case
    file_dfr=path+'case.csv'
    dfr = pd.read_csv(file_dfr)
    L = dfr["L"][0]; nb_cell = dfr["NP"][0]
    np.save('./results/mesh.npy',[nb_cell,L]) # save mesh in file

    # bathy and Href read in files
    bathy_t_np = np.load(path+'bathy_t.npy')
    Href_np = np.load(path+'Href.npy')
    
    # create the vda_river object
    param_mesh = {'number cells':nb_cell,'length river':L}
    param_geom = {'bathymetry array': bathy_t_np, 'Href array': Href_np}
    H_up, H_down = dfr["BC"][0], dfr["BC"][1]
    param_bc = {'H bc upstream': H_up,'H bc downstream': H_down}
    param_elements = {'type of elements': 'CG', 'degree of freedom': 2}

    # initialization
    ind_obs_np =  np.ones(nb_cell) 
    Hobs_np = np.zeros(nb_cell) 
    b_b_np = np.zeros(nb_cell) # no VDA is performed => unused 
    alpha = 0. # no VDA is performed, so value 0. is here ok. 
    param_VDA = {'regul. coeff.': alpha, 'obs': Hobs_np, 'background': b_b_np, 'obs index': ind_obs_np}

    print("generate_obs: create the vda_river model named obs_vda")
    obs_vda = vda_river('linear_stationnary', param_mesh, param_elements, param_geom, param_bc, param_VDA)
    # convert in FEniCS object 
    bathy_t = array_to_FEfct(obs_vda.V,bathy_t_np)
    Href = array_to_FEfct(obs_vda.V,Href_np)
    
    if os.path.isdir(path+"Hobs.npy") :
      print("generate_obs: the observations are those already available in file",path+'Hobs.npy')
      Hobs_np = np.load(path+"Hobs.npy")
      #obs_vda.Hobs = array_to_FEfct(obs_vda.V,Hobs_np)
    
    else : # observations are generated from bathy_t: this will be twin experiments
        noise_bool = 1 # MAY BE CHANGED
        print("generate_obs: synthetic observations will be generated with noise_bool=", noise_bool,". (twin experiments plan)")
        Hobs = obs_vda.Observations(bathy_t, noise_bool)
        H_t = obs_vda.Direct_Model(bathy_t)

        # write in files
        print("generate_obs: write the potentially new values of H_t and Hobs in files, in ", path)
        Hobs_np = FEfct_to_array(obs_vda.mesh, Hobs)
        H_t_np = FEfct_to_array(obs_vda.mesh, H_t)
        np.save(path+"Hobs.npy",Hobs_np)    
        np.save(path+"H_t.npy",H_t_np)

#
# The VDA process
#
def run_vda(path) : 
    """
    Perform the VDA process that are performed:
    - the direct model, the adjoint model, the gradient evaluation with multiple calls to the minimizer 
    The complete VDA process starts from a 1st guess value defined here.
    The observations are either provided by file or computed by the model (synthetic observations / twin experiments)
    -----------
    Parameters:
    path : str, path of the data to be used 
    """

    # geometry and mesh
    file_dfr = path+'case.csv'
    dfr = pd.read_csv(file_dfr)
    L = dfr["L"][0]; nb_cell = dfr["NP"][0]; href = dfr["href"][0]
    param_mesh = {'number cells':nb_cell,'length river':L}

    # FE types
    te, dof = 'P', 2
    param_elements = {'type of elements': te, 'degree of freedom': dof}

    # read the true bathy, the observations (everywhere) and Href in files
    Href_np = np.load(path+'Href.npy')
    bathy_t_np = np.load(path+'bathy_t.npy') # the true bathy given Href above
    Hobs_np = np.load(path+'Hobs.npy')
    #
    # observations locations
    #
    ind_obs_np =  np.zeros(nb_cell)
    freq_obs = 10 # = 1 for all # MAY BE CHANGED
    for k in range(0,nb_cell,freq_obs): 
      ind_obs_np[k-1] = 1
    np.save(path+'ind_obs.npy', ind_obs_np)
    # actual observations# Hobs_partial_np = Hobs_np * ind_obs_np
    print("run_vda: observations are partial only, with freq_obs = ",freq_obs)
    
    # a trick to be closer to the non linear model solution...
    Href_eq_Hobs = False # MAY BE CHANGED
    if (Href_eq_Hobs == True):
        Href_np = np.load(path+'Hobs.npy') # Href = Hobs
        print("routine run_vda: Href is defined equal to Hobs(b_t) => b_t is not the true bathymetry anymore !...")
    
    # Dirichlet BC values
    H_up, H_down = dfr["BC"][0], dfr["BC"][1]
    param_bc = {'H bc upstream': H_up,'H bc downstream': H_down}
    #
    # 1st guess value 
    #    
    #bathy = np.linspace(np.max(bathy_1st),np.min(bathy_1st),np.size(bathy_1st))
    b_b_np = np.load(path+"background.npy")
    print("The background value b_b (which is potentially used in Jreg only) has been defined in generate_case.py")

    print("run_vda: the 1st guess value b_1st is now defined in some way either from b_b or from a prior depth (to be tuned)...")
    #bathy_1st_np = b_b_np - np.ones(len(b_b_np)); print("run_vda: the 1st guess is set from b_b value")
  
    prior_depth = href # PRIOR TO BE TUNED
    bathy_1st_np = Hobs_np - 1.5 * prior_depth # - (Hobs_np[0]-bathy_t_np[0]) * 1.5 
    #bathy_1st_np = np.linspace(np.max(bathy_ref_np)-mean_depth,np.min(bathy_ref_np)-mean_depth,np.size(bathy_ref_np))

    ans = input("...(type any key to resume)...\n")
  
    # save in file
    np.save("./results/bathy_1st.npy",bathy_1st_np)

    # river geometry param
    param_geom = {'bathymetry array': bathy_1st_np, 'Href array': Href_np}

    if not( nb_cell == np.size(bathy_t_np) ) or not( nb_cell == np.size(bathy_1st_np) ) or not( nb_cell == np.size(Href_np) ) : 
        print("run_vda: dimensions:",nb_cell, np.size(bathy_t_np), np.size(bathy_1st_np), np.size(Href_np)) 
        sys.exit('run_vda routine: dimensions mismatch !')

    #
    # Regularization term
    #
    # 'reg' must be equal to either 'grad' or 'b_b': MAY BE CHANGED
    
    regul_term = 'b_b' # regularization term form 
    print("run_vda: the form of the regularization term is: ",regul_term)

    #ans = input("run_vda: let us plot the geometry, the observations and the 1st guess ? (type y or any other key)")
    ans='y'
    if ans=='y':
        # plots
        plt.figure(1)
        plt.plot(np.linspace(0,L/1000,nb_cell),Href_np,'c--',label=r"$H_{ref}$")
        plt.plot(np.linspace(0,L/1000,nb_cell),Hobs_np,'b',label=r"$H_{obs}$")
        if regul_term == 'b_b':
          plt.plot(np.linspace(0,L/1000,nb_cell),b_b_np,'g--',label=r"$b_b$")
        plt.plot(np.linspace(0,L/1000,nb_cell),bathy_1st_np,'r--',label=r"$b_{1st}$")
        plt.plot(np.linspace(0,L/1000,nb_cell),bathy_t_np,'k',label=r"$b_t$")
        plt.fill_between(np.linspace(0,L/1000,nb_cell), np.min(bathy_t_np), bathy_t_np, facecolor='k', alpha=0.5)
        plt.xlabel(r"$x$ ($km$)"); plt.xlim(0,L/1000)
        plt.ylabel(r"$z$ ($m$)",rotation="horizontal"); plt.ylim(np.min(bathy_t_np),H_up+1)
        plt.title('Observations Hobs & first guess')
        plt.legend()
        plt.show(block=False)

    ###
    alpha = 1. # must be equal to 1 to be next properly tuned after iteration #0 (see self.VDA routine)
    
    param_VDA = {'regul. coeff.': alpha, 'obs': Hobs_np, 'background': b_b_np, 'obs index': ind_obs_np, 'regul. term form' : regul_term}
     
    print("creating the vda_river object...\n")
    my_vda = vda_river('linear_stationnary', param_mesh, param_elements, param_geom, param_bc, param_VDA)

    # convert in FEnicCS format
    bathy_t = array_to_FEfct(my_vda.V,bathy_t_np)
    bathy_1st = array_to_FEfct(my_vda.V,bathy_1st_np)
    Href = array_to_FEfct(my_vda.V,Href_np)
       
    print("\nrun_vda: metric(s) definition") 
    covar_matrix = 0 # MAY BE CHANGED
    if covar_matrix == 1 :
        L = 10 # MAY BE CHANGED
        C_chol = my_vda.Cov_Expo(L)
        print("     a covariance matrix has been defined. It can be used or not in the VDA iterations... \n")
    else:
        print("     no covariance matrix is considered\n")
    
    # perform the complete VDA process
    my_vda.VDA(bathy_1st_np, bathy_t_np)

    
#########################
#    MAIN PROGRAM
#########################

if __name__ == '__main__':
    print("***************")
    print("*** main.py ***")
    # by default, all input files are provided in this path
    path = './data/'
    
    # create a new void outputs directories
    if os.path.isdir('./results/'): 
      shutil.rmtree('./results')
    os.makedirs('./results')
    if not(os.path.isdir('./results/')):
      os.makedirs('./results-store/')
    #
    # Generate a new test case or not ? 
    # if yes, all data to run a new case will be generated
    #
    ans = input("* Do we generate a new case (bathymetry, Dirichlet BC, Href) ? (type y or any other key)")
    #ans='n'
    if ans=='y':
        print("###  Generate new case ###")
        path_n ='./data_new/'
        print("main: call generate_case")
        generate_case(path_n) # new data are written in files

        print("\nmain: call define_Href to first define Href value")
        define_Href(path_n,True) # basic Href value is written in file
        
        print("\nmain: call run_direct to compute H corresponding to the new data\n")
        run_direct(path_n) # result is written in file
        
        print("\nmain: call define_Href to potentially update Href value\n")
        define_Href(path_n) # updated Href value is written in file
        
        print("\nmain: call generate_obs to generate observations")
        generate_obs(path_n) # Hobs but also the new value of H_t are written in file

        
        # transfer files or not
        print("\nmain: do we copy the newly created case files in ", path," (therefore replacing the former ones) ?")
        ans = input("       ...(type y or any other key)...")
        if ans =='y':
          shutil.copytree(path_n, path, dirs_exist_ok=True)
          files = glob.glob(path_n+'*')
          for f in files:
            os.remove(f)
          print("main: the new case files have replaced the former ones in ", path)
          path_plot = path
        else:
          print("main: the case files in", path," are unchanged")
          path_plot = path_n
  
        # plot
        print("main: the plots are those of the new case: files are read from ", path_plot)
        plot_direct(path_plot)

        print("### Over ###")
        quit() # program is stopped

    #
    # Gradient test
    #
    ans = input("* Do we perform a gradient test ? (type y or any other key)")
    #ans="n"
    if ans=="y":
        print("### Gradient test ###")
        print("main.py: the gradient test is performed for the test case stored in ", path)
        # (the observations set will be re-generated from these data)
        test_gradient(path)
        print("### Over ###")
        quit()
    
    #   
    # VDA process
    #
    # run the vda process from the case previously generated and saved in files located in path
    print("\n### GO FOR VDA ###")
    print("main: the employed data are the ones available in ", path,"\n")
    run_vda(path)

    
    #
    # plots
    #
    print('\n### main: the VDA process is finished ###\n')
    print(' First read the reason why the process has stopped and analyse the convergence curves !')
    print(' Is the minimization process satisfying?')
    print(' IF yes, you are allowed to analyse the physical fields, otherwise...\n') 
    #print(' Do you need to set up differently problem parameters (e.g. regularization term) and re-run ?')
    plot_outputVDA(path)
    
  
    # Prepare closing
    '''
    ans = input('main: type any key to close all figures...')
    if ans=='y':
      plt.close('all')
    else:
      print("wait")
    '''
