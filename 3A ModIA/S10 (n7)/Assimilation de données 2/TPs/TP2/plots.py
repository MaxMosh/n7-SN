import matplotlib.pyplot as plt
import numpy as np
from pdb import set_trace
import os 
from dolfin import *
#from fenics import *
import pandas as pd #data frame


#
# Plot the direct model output + the observations + Href
#

def plot_direct(path):
    """
    Plot bathymetry + Href + Hobs + H 
    (case of the stationary semi-linearized model)
    -----------
    Parameters :
    path : str, path of the data to be used 
    """
    # basic data of the case
    file_dfr=path+'case.csv'
    dfr = pd.read_csv(file_dfr)
    L = dfr["L"][0]; nb_cell = dfr["NP"][0]
    H_up, H_down = dfr["BC"][0], dfr["BC"][1]
    
    #x = np.linspace(0,L/1000,nb_cell)

    # read in files
    bathy_t_np = np.load(path+'bathy_t.npy')
    Href_np = np.load(path+'Href.npy')
    H_t_np = np.load(path+'H_t.npy')
    Hobs_np = np.load(path+'Hobs.npy')
    b_b_np = np.load(path+'background.npy')
    
    plt.figure()
    plt.plot(np.linspace(0,L/1000,nb_cell),Href_np,'c--',label=r"$H_{ref}$")
    plt.plot(np.linspace(0,L/1000,nb_cell),H_t_np,'b',label=r"$H_{t}$")
    plt.plot(np.linspace(0,L/1000,nb_cell),Hobs_np,'r--',label=r"$H_{obs}$")
    plt.plot(np.linspace(0,L/1000,nb_cell),bathy_t_np,'k',label=r"$b^t$")
    plt.plot(np.linspace(0,L/1000,nb_cell),b_b_np,'g--',label=r"$b_b$")
    plt.fill_between(np.linspace(0,L/1000,nb_cell), np.min(bathy_t_np), bathy_t_np, facecolor='k', alpha=0.5)
    plt.xlabel(r"$x$ ($km$)"); plt.xlim(0,L/1000)
    plt.ylabel(r"$z$ ($m$)",rotation="horizontal"); plt.ylim(np.min(bathy_t_np),H_up+1)
    plt.title('Direct model & observations')
    plt.legend()
    plt.show()


    
#
# Plot outputs resulting from the VDA process
#

def plot_outputVDA(path_data, path_out='./results/'):

  """
  plot the outputs of the VDA process
  Parameters : 
  path_data : str, path to the folder that contains the data
  path_out : optional, str, path to the folder that contains the output, by default './results/'
  """
    
  # load data
  file_dfr=path_data+'case.csv'
  dfr = pd.read_csv(file_dfr)
  L = dfr["L"][0]; nb_cell = dfr["NP"][0]
  H_up, H_down = dfr["BC"][0], dfr["BC"][1]

  # x array = point locations in km and not in m
  x = np.linspace(0,L/1000,int(nb_cell))
    
  # load
  fitting = np.load(path_out+'fitting-threshold.npy')
  # load fields vs iterations
  cost = np.load(path_out+'cost.npy')
  cost_obs = np.load(path_out+'cost_obs.npy')
  cost_reg = np.load(path_out+'cost_reg.npy')
  grad_cost = np.load(path_out+'grad_cost.npy')
  b_vs_ite = np.load(path_out+'bathy_vs_ite.npy')
  H_vs_ite = np.load(path_out+'H_vs_ite.npy')
  RMSE_b = np.load(path_out+'RMSE_b.npy')
  RMSE_H = np.load(path_out+'RMSE_H.npy')

  # vs x (reshape)
  b_t = np.load(path_data+'bathy_t.npy').reshape(np.shape(x))
  b_b = np.load(path_data+'background.npy').reshape(np.shape(x))
  b_1st = np.load(path_out+'bathy_1st.npy').reshape(np.shape(x))
  Href = np.load(path_data+'Href.npy').reshape(np.shape(x))
  H_t = np.load(path_data+'H_t.npy').reshape(np.shape(x))
  b_star = np.load(path_out+'bathy_star.npy').reshape(np.shape(x))
  H_star = np.load(path_out+'H_star.npy').reshape(np.shape(x))
  # obs
  Hobs_full = np.load(path_data+'Hobs.npy').reshape(np.shape(x))
  ind_obs = np.load(path_data+'ind_obs.npy').reshape(np.shape(x))
  Hobs_sparse = Hobs_full * ind_obs
  for i in range(0,nb_cell):
    if (Hobs_sparse[i]< 1.e-7):
      np.delete(Hobs_sparse, i)

  #### PLOTS #####
  fsize = 15 # legend size
  
  # cost terms vs iterations from iteration#0 
  plt.figure()
  plt.xlim(0,np.size(cost)-1)
  plt.plot(cost/cost[0],'b',label=r'$j$')
  plt.plot(cost_obs/cost[0],'c--',label=r'$j_{obs}$')
  plt.plot(cost_reg/cost[0],'r--',label=r'$\alpha . j_{reg}$')
  plt.plot(np.ones(nb_cell) * fitting[0]/cost[0],'--',label ='10 cm')
  plt.plot(np.ones(nb_cell) * fitting[1]/cost[0],'--',label ='20 cm')
  plt.plot(np.ones(nb_cell) * fitting[2]/cost[0],'--',label ='30 cm')
  plt.xlabel(r"iteration",fontsize=fsize)
  plt.ylabel(r"normalized cost terms",fontsize=fsize)
  plt.legend(loc="upper right",prop={'size': fsize})
  plt.show(block=False)
  plt.savefig('./results_store/cost_'+str(round(b_b[0],3))+'.png')

  # cost terms vs iterations from iteration #ite_cut for better readibility
  dim_ite = np.size(cost)
  if (dim_ite>10):
    ite_cut = 10 #np.maximum(10,(dim_ite-100))
  else:
    ite_cut = 2
    
  cost1 = np.delete(cost, np.arange(ite_cut)); cost_obs1 = np.delete(cost_obs, np.arange(ite_cut)); cost_reg1 = np.delete(cost_reg, np.arange(ite_cut))
  xk = np.delete(np.arange(np.size(cost)), np.arange(ite_cut))
  # the constant ref values
  #Jobs_ref1, Jobs_ref2 = np.load('./results/Jobs_ref1.npy'), np.load('./results/Jobs_ref2.npy')
  #Jobs_ref1, Jobs_ref2 = np.delete(Jobs_ref1, np.arange(ite_cut))/cost[0], np.delete(Jobs_ref2, np.arange(ite_cut))/cost[0] # BUG TO BE SOLVED
  
  plt.figure()
  plt.xlim(xk[0],np.size(xk)-1)
  plt.plot(xk,cost1/cost[0],'b',label=r'$j$')
  plt.plot(xk,cost_obs1/cost[0],'c--',label=r'$j_{obs}$')
  plt.plot(xk,cost_reg1/cost[0],'r--',label=r'$\alpha . j_{reg}$')
  plt.plot(np.ones(nb_cell) * fitting[0]/cost[0],'--',label ='10 cm')
  #plt.plot(np.ones(nb_cell) * fitting[1]/cost[0],'--',label ='20 cm')
  #plt.plot(np.ones(nb_cell) * fitting[2]/cost[0],'--',label ='30 cm')
  plt.xlabel(r"iteration",fontsize=fsize)
  plt.ylabel(r"normalized cost terms",fontsize=fsize)
  plt.legend(loc="upper right",prop={'size': fsize})
  plt.show(block=False)
  
  # gradient vs iterations
  plt.figure()
  plt.plot(grad_cost,'k',label=r"$||\nabla j||^2$")
  plt.xlabel(r"iteration",fontsize=fsize)
  plt.ylabel(r"gradient norm",fontsize=fsize)
  plt.yscale('log')
  plt.legend(prop={'size': fsize})
  plt.xlim(0,np.size(grad_cost)-1)
  plt.show(block=False)

  
  
  # RMSEs vs iterations
  f3=plt.figure()
  ax2 = f3.add_subplot(1,1,1)
  ax2.plot(RMSE_b,'r',label=r"$RMSE~b$")
  ax2.set_yscale('log')
  ax2.set_xlim(0,np.size(RMSE_b)-1)
  ax2.set_xlabel(r"iteration",fontsize=fsize)
  ax2.set_ylabel(r"$RMSE~b$ ($m$)",fontsize=fsize)
  ax3=ax2.twinx()
  ax3.plot(RMSE_H,'b',label=r"$RMSE~H$")
  ax3.set_yscale('log')
  ax3.set_xlim(0,np.size(RMSE_H)-1)
  ax3.set_ylabel(r"$RMSE~H$ ($m$)",fontsize=fsize)
  lines2, labels2 = ax2.get_legend_handles_labels()
  lines3, labels3 = ax3.get_legend_handles_labels()
  ax3.legend(lines2+lines3,labels2+labels3,loc='upper right',prop={'size': fsize})
  
  
  #
  # misfit (H vs iterations - Hobs)  MINUS the mean slope (for better readibility)
  #
  H_mean = H_up - (H_up-H_down)/(L*1.e-3) * x
  freq_p = 20 # plot every freq_b iterations : MAY BE CHANGED
  #
  plt.figure()
  H_slope = H_vs_ite # init
  for i in range(np.shape(H_vs_ite)[0]):
    H_slope[i,0,:]= H_vs_ite[i,0,:] - H_mean # substract the mean slope value
    if i%freq_p ==0 : 
      plt.plot(x,H_slope[i,0,:],label=r'iteration'+str(i))
  if (np.shape(H_vs_ite)[0]-1)%10 != 0: 
    plt.plot(x,H_slope[i,0,:],label='iteration'+str(np.shape(H_vs_ite)[0]-1))
  #plt.plot(x,(H_t- H_mean),'r--',label='H_t')
  obs_index = np.nonzero(Hobs_sparse) # non zero values only
  
  #plt.plot(x,(Hobs_sparse - H_mean),'c.',label='H_obs')
  plt.plot(x[obs_index],(Hobs_sparse[obs_index] - H_mean[obs_index]),'c.',label='H_obs')
  plt.xlim(0,np.max(x))
  plt.xlabel(r"$x$ ($km$)",fontsize=fsize)
  plt.ylabel(r"${H}$ ($m$)",fontsize=fsize)
  plt.title('(WS elevation H MINUS mean slope) vs iterations')
  plt.legend(prop={'size': fsize})
  plt.show(block=False)
  #plt.savefig('./results_store/H_n_'+str(round(b_b[0],0))+'.png')

 
  # bathymetry vs several intermediate iterations (including the last iteration = optimal value) + true value
  # values of b MINUS the mean slope (for better readibility)
  plt.figure()
  b_slope = b_vs_ite # init
  for i in range(np.shape(b_vs_ite)[0]):
    b_slope[i,0,:]= b_vs_ite[i,0,:] - H_mean # substract the mean slope value
    if i%freq_p == 0: 
      plt.plot(x,b_slope[i,0,:],label=r'iteration'+str(i))
  if (np.shape(b_vs_ite)[0]-1)%10 != 0: 
    plt.plot(x,b_slope[i,0,:],label='iteration'+str(np.shape(b_vs_ite)[0]-1))
  plt.plot(x,(b_t-H_mean),'r',label='b_t')
  plt.xlim(0,np.max(x))
  plt.xlabel(r"$x$ ($km$)",fontsize=fsize)
  plt.ylabel(r"${b}$ ($m$)",fontsize=fsize)
  plt.title('(bathymetry MINUS mean slope) vs iterations')
  plt.legend(prop={'size': fsize})
  plt.show(block=False)
  plt.savefig('./results_store/bathy_n_'+str(round(b_b[0],0))+'.png')



  # Mediane errors made on the bathy
  median_error_b = np.load(path_out+'median_error_b.npy')
  '''
  plt.figure()
  plt.plot(median_error_b,label=r'median error for ... (to be completed)')
  plt.title('Jreg terms (grad(b), b-b_b) through median value error on the bathymetry [m]')
  plt.ylabel('median error on b_t [m]')
  plt.xlabel('iterations')
  plt.legend()
  plt.show(block=False)
  '''
  
  # water elevations and bathymetries
  plt.figure()
  plt.plot(x,H_star,'c',label=r"$H_{*}$")
  #plt.plot(x,Hobs_sparse,'c.',label=r"$H_{obs}$")
  plt.plot(x[obs_index],Hobs_sparse[obs_index],'c.',label=r"$H_{obs}$")
  plt.plot(x,b_1st,'r--',label=r"$b_{1st}$")
  plt.plot(x,b_b,'g--',label=r"$b_b$")
  plt.plot(x,b_star,'b--',label=r"$b_{*}$")
  plt.plot(x,b_t,'k',label=r"$b_t$")
  plt.fill_between(x, np.min(b_t), b_t, facecolor='k', alpha=0.5)
  plt.xlabel(r"$x$ ($km$)"); plt.xlim(0,L/1000)
  plt.ylabel(r"$z$ ($m$)",rotation="horizontal") #; plt.ylim(np.min(b_t),H_up+1)
  plt.title('1st guess, true & final')
  plt.legend()
  plt.show()
  plt.savefig('./results_store/profiles_'+str(round(b_b[0],0))+'.png')
  

