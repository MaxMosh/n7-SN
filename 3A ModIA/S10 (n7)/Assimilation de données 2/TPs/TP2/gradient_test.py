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
from utils import * 

set_log_level(50) # to minimize automatic FeniCS comments


#
# The gradient test 
#
def test_gradient(path) : 

    """
    Gradient test
    in the case of the stationary semi-linearized model
    -----------
    Parameters :
    path : str, path of the data to be used 
    nothing is written in path :)
    """
    
    # geometry and mesh
    # dfr : dataframe containing: {"L":[L,0.],"NP":[n_cross_sec,0.],"BC":[H_in,H_out]}
    file_dfr=path+'case.csv'
    dfr = pd.read_csv(file_dfr)
    L = dfr["L"][0]; nb_cell = dfr["NP"][0]
    param_mesh = {'number cells':nb_cell,'length river':L}

    # FE type
    te,dof = 'P', 2
    param_elements = {'type of elements': te, 'degree of freedom': dof}

    # bathy and Href read in files
    bathy_np = np.load(path+'bathy_t.npy')
    Href_np = np.load(path+'Href.npy')
    
    param_geom = {'bathymetry array': bathy_np, 'Href array': Href_np}
    if not( nb_cell == np.size(bathy_np) ) or not( nb_cell == np.size(Href_np) ) :
        sys.exit('test gradient: dimensions mismatch')
        
    # Dirichlet BC
    H_up, H_down = dfr["BC"][0], dfr["BC"][1]
    param_bc = {'H bc upstream': H_up,'H bc downstream': H_down}

    # initialization
    ind_obs_np =  np.ones(nb_cell) # observations are available at each point
    Hobs_np = np.zeros(nb_cell)
    b_b_np = np.zeros(nb_cell) # unused    
    alpha = 0. # value 0. is here ok because no VDA process is performed
    param_VDA = {'regul. coeff.': alpha, 'obs': Hobs_np, 'background': b_b_np, 'obs index': ind_obs_np}
    
    # create the vda_river object
    print("gradient test: create the vda_river model for observations only")
    grad_vda = vda_river('linear_stationnary', param_mesh, param_elements, param_geom, param_bc, param_VDA)

    # convert in FEnicCS format
    bathy = array_to_FEfct(grad_vda.V,bathy_np)
    Href = array_to_FEfct(grad_vda.V,Href_np)
    
    #
    # generate the observations with/without noise, using bathymetry bathy_t
    noise_bool = 0
    grad_vda.Observations(bathy, noise_bool)
    print("gradient test. synthetic observations are generated with noise_bool=", noise_bool)
    
    # Point of computation b0 (bathymetry value). It is necessarily different than bathy_t otherwise gradient.approx.0
    # MAY BE INVESTIGATED
    print("The point of computation b0 is defined")
    b0_np = 0.8 * bathy_np - 1. # b0: a perturbation of the true bathy (bathy_np)
    #b0_np = np.linspace(np.max(bathy_np)-5.,np.min(bathy_np)-2.,np.size(bathy_np)) 
    b0 = array_to_FEfct(grad_vda.V,b0_np) # convert to FEniCS fct

    # perturbation of the control variable
    db_array = np.ones(nb_cell) # constant vector of good order of magnitude
    db = array_to_FEfct(grad_vda.V,db_array)

    # compute values at b0: H, cost 
    H_b0 = grad_vda.Direct_Model(b0) # H for b0
    cost_c = grad_vda.Cost(b0,H_b0)[0] # cost function at b0

    # compute the adjoint-based gradient 
    P_b0 = grad_vda.Adjoint_Model(b0,H_b0) # p for b0
    dj_norm = grad_vda.Gradient_Cost(b0,H_b0,P_b0,flag_val=True)[1] 
    print("norm of gradient (computed with the adjoint) at b0: ",dj_norm)    

    # convert to array format
    H_b0_np = FEfct_to_array(grad_vda.mesh,H_b0)
    Href_np = FEfct_to_array(grad_vda.mesh,grad_vda.Href)
    Hobs_np = FEfct_to_array(grad_vda.mesh,grad_vda.Hobs)
    
    # plots
    plt.figure(1)
    #plt.plot(np.linspace(0,L/1000,nb_cell),H,'b',label=r"$H(b^t)$")
    plt.plot(np.linspace(0,L/1000,nb_cell),H_b0_np,'b--',label=r"$H(b_0)$")
    plt.plot(np.linspace(0,L/1000,nb_cell),Href_np,'c--',label=r"$H_{ref}$")
    plt.plot(np.linspace(0,L/1000,nb_cell),Hobs_np,'r--',label=r"$H_{obs}$")
    plt.plot(np.linspace(0,L/1000,nb_cell),bathy_np,'k',label=r"$b^t$")
    plt.plot(np.linspace(0,L/1000,nb_cell),b0_np,'k--',label=r"$b_0$")
    plt.fill_between(np.linspace(0,L/1000,nb_cell), np.min(bathy_np), bathy_np, facecolor='k', alpha=0.5)
    plt.xlabel(r"$x$ ($km$)"); plt.xlim(0,L/1000)
    plt.ylabel(r"$z$ ($m$)",rotation="horizontal"); plt.ylim(np.min(bathy_np),H_up+1)
    plt.title("Gradient test. Bathymetry and water surface elevations")
    plt.legend()
    plt.show(block=False)
    

    n = 15 # number of computed points
    print("number of computed points: n=",n)
    i_c, i_r, i_l = np.zeros((n,1)), np.zeros((n,1)), np.zeros((n,1)) # array of the ratios I
    dj_r, dj_c, dj_l = np.zeros((n,1)), np.zeros((n,1)), np.zeros((n,1)) # array of the cost values
    epsilon = np.zeros((n,1)) # array of the n epsilon values

    for i in np.arange(1,n+1,1): # loop on the epsilon values
        j = (i-1)/3.
        eps = 10**(-j)
        epsilon[i-1] = eps # store the epsilon values for plot

        # the perturbed bathy at left / right, corresponding solutions H and cost function values
        # at right (plus)
        b0_np_p = b0_np + eps * db_array
        b0_p = array_to_FEfct(grad_vda.V,b0_np_p)
        H_p = grad_vda.Direct_Model(b0_p) # direct model provides H
        cost_p = grad_vda.Cost(b0_p,H_p)[0]# cost 
        # at left (minus)
        b0_np_m = b0_np - eps * db_array  
        b0_m = array_to_FEfct(grad_vda.V,b0_np_m)
        H_m = grad_vda.Direct_Model(b0_m) # direct model provides H
        cost_m = grad_vda.Cost(b0_m,H_m)[0] # cost

        # Why the convergence curve increases so soon in epsilon ? 
        print(" epsilon =",eps)
        #print("cost function value at (b0_m, b0, b0_p):", cost_m, cost_c, cost_p)
        #print("numerateur du ratio I : delta(j)_m, delta(j)_p =", (cost_c-cost_m), (cost_c-cost_p))
                
        # FD approximations
        dj_r[i-1] = (cost_p - cost_c) / eps # uncentered FD at right
        dj_l[i-1] = (cost_c - cost_m) / eps # uncentered FD at left
        dj_c[i-1] = (cost_p - cost_m) / (2.*eps) # centered FD
        print("uncentered FD approx of dj at right : ", dj_r[i-1])
        print("uncentered FD approx of dj at left: ", dj_l[i-1])
        print("centered FD approx of dj : ", dj_c[i-1])

        # plotted ratios (I-1)
        i_r[i-1] = np.abs( 1. - ( dj_r[i-1] / dj_norm ) ) # Iratio, at right
        i_l[i-1] = np.abs( 1. - ( dj_l[i-1] / dj_norm ) ) # Iratio, at left
        i_c[i-1] = np.abs( 1. - ( dj_c[i-1] / dj_norm ) ) # Iratio, at left

    # slope of the (k-1)-st segment
    k=4
    slope_r = (np.log(i_r[k-1])-np.log(i_r[k])) / (np.log(epsilon[k-1])-np.log(epsilon[k]))
    slope_l = (np.log(i_l[k-1])-np.log(i_l[k])) / (np.log(epsilon[k-1])-np.log(epsilon[k]))
    slope_c = (np.log(i_c[k-1])-np.log(i_c[k])) / (np.log(epsilon[k-1])-np.log(epsilon[k]))
    
    print("")
    print("epsilon=",np.transpose(epsilon)[0])
    print("uncentered cases, at (left, right). slopes (of the 1st segment)= ",slope_l[0],slope_r[0])
#    print("   |1-I| = ",np.transpose(i_r)[0]")
    print("centered case. slope (of the 1st segment) = ",slope_c[0])
    #    print("   |1-I| = ",np.transpose(i_c)[0])

    
    print("\n test_gradient: is the gradient test ok ?")
    print("If yes, next you can run a VDA process...")
    
    # plots
    plt.figure(2)  
    plt.plot(epsilon,dj_r,'b--',label=r'$dj_r$')
    plt.plot(epsilon,dj_l,'c',label=r'$dj_l$')	
    plt.plot(epsilon,dj_c,'r',label=r'$dj_c$')	
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$FD approx$',rotation='horizontal')
    plt.xlabel(r'$\epsilon$')
    plt.legend(loc='upper left')
    plt.show(block=False)
    
    # plots
    plt.figure(3)  
#    plt.plot(epsilon,i_r,'b',label=r'$|1-I_{\epsilon ,d} |$, slope = '+str(slope_r[0]))
    plt.plot(epsilon,i_r,'b',label=r'$|1-I_{\epsilon ,d} |$')
    plt.plot(epsilon,i_l,'c',label=r'$|1-I_{\epsilon ,l} |$')	
    plt.plot(epsilon,i_c,'r',label=r'$|1-I_{\epsilon ,c} |$')
    plt.yscale('log')
    plt.xscale('log')
    plt.ylabel(r'$|1-I_{\epsilon} |$',rotation='horizontal')
    plt.xlabel(r'$\epsilon$')
    plt.legend(loc='upper left')
    plt.show()

