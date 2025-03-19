import numpy as np
import torch
import matplotlib.pyplot as plt

from Backwater_model import Ks_function, bathymetry_interpolator

def display_data(model, col, ref_solution, obs): 
    """
    Function used to display the data for the inverse problem. 
    """
        
    if (model.k_interpolation == 'P0'):
        subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                    max(ref_solution['domain']).item(),
                                    model.dim_k + 1)
        indices = ((subdomains/ref_solution['dx']).clamp(max = ref_solution['domain'].shape[0] - 1)).int().detach().clone().cpu().numpy()
        
    if (model.k_interpolation == 'P1'):
        subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                    max(ref_solution['domain']).item(),
                                    model.dim_k)
        indices = ((subdomains/ref_solution['dx']).clamp(max = ref_solution['domain'].shape[0] - 1)).int().detach().clone().cpu().numpy()

    fig, ax = plt.subplots()
    if (obs.N_obs < 5):
        if (obs.N_obs != 0):
            ax.set_title('Reference solution and data, $K_s = {} \ m^{{1/3}}.s^{{-1}}$'.format(list(np.around(ref_solution['parameter'].detach().clone().cpu().numpy(),2))))
        else:
            ax.set_title('Reference solution, $K_s = {} \ m^{{1/3}}.s^{{-1}}$'.format(list(np.around(ref_solution['parameter'].detach().clone().cpu().numpy(),2))))
    else:
        ax.set_title('Reference solution and data')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['solution'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            color='#1f77b4', label = 'RK4 solution')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['normal height'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            'y--', label = 'normal height')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['critical height'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            'r--', label = 'critical height')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            'g', label = 'bathymetry')
    if (obs.N_obs != 0):
        ax.scatter(obs.data[:, 0].detach().clone().cpu().numpy(), 
                   obs.data[:, 1].detach().clone().cpu().numpy()+ref_solution['bathymetry'][obs.indices].flatten().detach().clone().cpu().numpy(), 
                   label = 'obs', color='#1f77b4')
    ax.scatter(subdomains.detach().clone().cpu().numpy(), 
               ref_solution['bathymetry'].detach().clone().cpu().numpy()[indices], 
               marker = '|', c = 'k', s = 100, label = 'subdomains')
    
    ax.set_xlabel(r'$x \ [m]$')
    ax.set_ylabel(r'$y \ [m]$')
    
    ax.fill_between(ref_solution['domain'].flatten().detach().clone().cpu().numpy(), 
                    ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(), 
                    0, color='green', alpha=.3)
    ax.fill_between(ref_solution['domain'].flatten().detach().clone().cpu().numpy(), 
                    ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(),
                    ref_solution['solution'].flatten().detach().clone().cpu().numpy() + ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(), 
                    color='blue', alpha=.2)
    
    ax.set_ylim(top = 1.1*max(ref_solution['bathymetry'] + ref_solution['solution']).item(),
                bottom = 0)
    
    ax2 = ax.twinx()

    ax2.set_ylabel(r'$K_s \ [m^{1/3}/s]$', y = 0.14)  
    ax2.plot(ref_solution['domain'].flatten().detach().clone().cpu().numpy(),
             ref_solution['parameter_function'].flatten().detach().clone().cpu().numpy(),
             '-.', label = r'$K_s^{true}(x)$', c = 'grey')
    ax2.set_ylim(0, 300)
    custom_ticks = np.arange(0, 100, 20)  
    ax2.set_yticks(custom_ticks) 
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', 
               ncol = 3, prop={'size': 7.5})
        
    plt.show()
    
def display_results(model, col, ref_solution, obs, plot_col = False):
    """
    Function used to display the result of the training. 
    """

    if (model.k_interpolation == 'P0'):
        subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                    max(ref_solution['domain']).item(),
                                    model.dim_k + 1)
        indices = ((subdomains/ref_solution['dx']).clamp(max = ref_solution['domain'].shape[0] - 1)).int().detach().clone().cpu().numpy()
        
    if (model.k_interpolation == 'P1'):
        subdomains = torch.linspace(min(ref_solution['domain']).item(), 
                                    max(ref_solution['domain']).item(),
                                    model.dim_k)
        indices = ((subdomains/ref_solution['dx']).clamp(max = ref_solution['domain'].shape[0] - 1)).int().detach().clone().cpu().numpy()

    fig, ax = plt.subplots()
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), ref_solution['solution'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            color='#1f77b4', label = '$h_{RK4}(x)$')
    ax.plot(col.grid.detach().clone().cpu().numpy(), 
            model(col.grid).detach().clone().cpu().numpy()+ref_solution['bathymetry_col'].detach().clone().cpu().numpy(), 
            'k--', label = r'$\tilde{h}(x)$')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            'g', label = '$b(x)$')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['normal height'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            'y--', label = '$h_n(x)$')
    ax.plot(ref_solution['domain'].detach().clone().cpu().numpy(), 
            ref_solution['critical height'].detach().clone().cpu().numpy()+ref_solution['bathymetry'].detach().clone().cpu().numpy(), 
            'r--', label = '$h_c(x)$')
    if (obs.N_obs != 0):
        ax.scatter(obs.data[:, 0].detach().clone().cpu().numpy(), 
                   obs.data[:, 1].detach().clone().cpu().numpy()+ref_solution['bathymetry'][obs.indices].flatten().detach().clone().cpu().numpy(), 
                   label = 'obs', color='#1f77b4')
    if plot_col:
        b_col = bathymetry_interpolator(col.device, col.all)[0]
        ax.scatter(col.all.detach().clone().cpu().numpy(), 
                   b_col.detach().clone().cpu().numpy(), 
                   label = 'col', color='black', s = 10)
    ax.scatter(subdomains.detach().clone().cpu().numpy(), 
               ref_solution['bathymetry'].detach().clone().cpu().numpy()[indices], 
               marker = '|', c = 'k', s = 100, label = 'sub.')

    ax.set_xlabel(r'$x \ [m]$')
    ax.set_ylabel(r'$y \ [m]$')

    Ks_true = ref_solution['parameter_function'].flatten().detach().clone().cpu().numpy()
    Ks_est = model.k_ref*Ks_function(ref_solution['domain'], model.k, model, col).flatten().detach().clone().cpu().numpy()

    ax.set_title('Calibrated model at iteration {}, RMSE = {:.4e}'.format(model.iter, np.linalg.norm(Ks_true - Ks_est, ord = 2)/np.linalg.norm(Ks_true, ord = 2)))
    ax.fill_between(ref_solution['domain'].flatten().detach().clone().cpu().numpy(), 
                    ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(), 
                    0, color='green', alpha=.3)
    ax.fill_between(ref_solution['domain'].flatten().detach().clone().cpu().numpy(), 
                    ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(),
                    ref_solution['solution'].flatten().detach().clone().cpu().numpy() + ref_solution['bathymetry'].flatten().detach().clone().cpu().numpy(), 
                    color='blue', alpha=.2)
    
    ax.set_ylim(top = 1.1*max(ref_solution['bathymetry'] + ref_solution['solution']).item(),
                bottom = 0)

    ax2 = ax.twinx()

    ax2.set_ylabel(r'$K_s \ [m^{1/3}/s]$', y = 0.14)  
    ax2.plot(ref_solution['domain'].flatten().detach().clone().cpu().numpy(),
             ref_solution['parameter_function'].flatten().detach().clone().cpu().numpy(),
             '-.', label = r'$K_s^{true}(x)$', c = 'grey')
    ax2.plot(ref_solution['domain'].flatten().detach().clone().cpu().numpy(),
             model.k_ref*Ks_function(ref_solution['domain'], model.k, model, col).flatten().detach().clone().cpu().numpy(),
             '-.', label = r'$K_s(x)$', c = 'k')
    ax2.set_ylim(0, 300)
    custom_ticks = np.arange(0, 100, 20)  # Créez des ticks jusqu'à la moitié du graphique
    ax2.set_yticks(custom_ticks)  # Définissez les ticks personnalisés sur l'axe y

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', 
               ncol = 2, prop={'size': 7.5})
    
    plt.show()
    
def display_training(model, col, ref_solution): 
    """
    Function used to display the training of the Neural Network. 
    """
    
    Ks_true = ref_solution['parameter_function'].flatten().detach().clone().cpu().numpy()
    Ks_est = model.k_ref*Ks_function(ref_solution['domain'], model.k, model, col).flatten().detach().clone().cpu().numpy()
    
    h_true = ref_solution['solution'].flatten().detach().clone().cpu().numpy()
    h_est = np.array(model.list_y).squeeze().transpose()[:,-1]
    
    print('#'*50)
    print('Final parameter RMSE : {:.2e}'.format(np.linalg.norm(Ks_true - Ks_est, ord = 2)/np.linalg.norm(Ks_true, ord = 2)))
    if (h_true.shape[0] == h_est.shape[0]):
        print('Final variable RMSE : {:.2e}'.format(np.linalg.norm(h_true - h_est, ord = 2)/np.linalg.norm(h_true, ord = 2)))
    print('#'*50)
    
    J_train = np.asarray(model.list_J_train)[model.list_iter_flag]
    if (col.test_size > 0):
        J_test = np.asarray(model.list_J_test)[model.list_iter_flag]
    
    fig, ax = plt.subplots()
    ax.set_title('J function during network optimization')
    ax.set_xlabel('L-BFGS iterations')
    ax.set_ylabel('J function') 
    ax.set_yscale('log')
    if (col.test_size > 0):
        ax.plot(J_train[:, 0], label = '$J$ train', color='#1f77b4')
        ax.plot(J_test[:, 0], label = '$J$ test', color='#ff7f0e')
        ax.plot(J_train[:, 1], '--*', label = '$J_{res}$ train', color='#1f77b4')
        ax.plot(J_test[:, 1], '--*', label = '$J_{res}$ test', color='#ff7f0e')
    else:
        ax.plot(J_train[:, 0], label = '$J$', color='black')
        ax.plot(J_train[:, 1], label = '$J_{res}$', color='#1f77b4')
    ax.plot(J_train[:, 2], label = '$J_{obs}$', color='#ff7f0e')
    ax.plot(J_train[:, 3], ':', label = '$J_{BC}$', color='#1f77b4')
    ax.legend(loc = 'best', ncol = 2, prop={'size': 8})

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.plot(np.asarray(model.list_grad)[model.list_iter_flag], label = r'$\| \nabla_{(\theta, k)} \cdot J \|_2$')
    ax.set_xlabel('L-BFGS iterations')
    ax.set_ylabel('gradient norm')
    ax.set_title('gradient norm during optimization')
    ax.legend(loc = 'best')
    
    if (len(model.list_J_gradients) > 0):
        
        fig, ax = plt.subplots()
        data = np.asarray(model.list_J_gradients)[model.list_iter_flag]
        ax.set_yscale('log')
        ax.plot(data[:, 0], label =r'$\| \frac{\partial \ J_{res}}{\partial \ k} \|_2$')
        ax.plot(data[:, 1], label = r'$\| \frac{\partial \ J_{res}}{\partial \ \theta} \|_2$')
        ax.plot(data[:, 2], label = r'$\| \frac{\partial \ J_{obs}}{\partial \ \theta} \|_2$')
        ax.plot(data[:, 3], label = r'$\| \frac{\partial \ J_{BC}}{\partial \ \theta} \|_2$')
        ax.set_xlabel('L-BFGS iterations')
        ax.set_ylabel('residual term gradient')
        ax.set_title('Residual term gradient during optimization')
        ax.legend(loc = 'best')
        
        boundaries = (100, 200)
        
        fig, ax = plt.subplots()
        data = np.asarray(model.list_J_gradients)[model.list_iter_flag]
        ax.set_xlim(left = boundaries[0], right = boundaries[1])
        ax.set_yscale('log')
        ax.plot(data[:, 0], label =r'$\| \frac{\partial \ J_{res}}{\partial \ k} \|_2$')
        ax.plot(data[:, 1], label = r'$\| \frac{\partial \ J_{res}}{\partial \ \theta} \|_2$')
        ax.plot(data[:, 2], label = r'$\| \frac{\partial \ J_{obs}}{\partial \ \theta} \|_2$')
        ax.plot(data[:, 3], label = r'$\| \frac{\partial \ J_{BC}}{\partial \ \theta} \|_2$')
        ax.set_xlabel('L-BFGS iterations')
        ax.set_ylabel('residual term gradient')
        ax.set_title('Residual term gradient during optimization between [{:.0f}, {:.0f}]'.format(boundaries[0],boundaries[1]))
        ax.legend(loc = 'best')

    h_true = ref_solution['solution'].detach().clone().cpu().numpy()
    h_est = np.array(model.list_y).squeeze()[model.list_iter_flag].transpose()

    Ks_true = ref_solution['parameter_function'].detach().clone().cpu().numpy()

    list_Ks_est = []
    list_k = list(np.array(model.list_k_matrix)[model.list_iter_flag])
    for k in list_k:
        list_Ks_est.append(model.k_ref*Ks_function(ref_solution['domain'], torch.tensor(k), model, col).flatten().detach().clone().cpu().numpy())
    Ks_est = np.array(list_Ks_est).transpose()

    fig, ax = plt.subplots()
    ax.set_yscale('log')
    if (h_true.shape[0] == h_est.shape[0]):
        ax.plot(np.linalg.norm(h_est-h_true, ord = 2, axis = 0)/np.linalg.norm(h_true, ord = 2), label = r'$h$')
    ax.plot(np.linalg.norm(Ks_est-Ks_true, ord = 2, axis = 0)/np.linalg.norm(Ks_true, ord = 2), label = r'$k$')
    ax.set_xlabel('L-BFGS iterations')
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE between RK4 solution and NN solution')
    ax.legend(loc='best')

    if (model.dim_k == 1):
        fig, ax = plt.subplots()
        ax.plot(model.k_ref*np.array(model.list_k[model.list_iter_flag]), label = r'$k_{est}$')
        ax.plot(ref_solution['parameter'].detach().clone().cpu().numpy()*np.ones(len(model.list_k)), '--', label = r'$k_{true}$')
        ax.set_xlabel('L-BFGS iterations')
        ax.set_ylabel(r'$K_s \ (m^{1/3}/s)$')
        ax.set_title('parameter inference')
        ax.legend(loc='best')

    else:

        fig, axs = plt.subplots(1, 2)
        fig.suptitle('Parameter convergence analysis')

        estimation = model.k_ref*np.array(model.list_k_matrix)[model.list_iter_flag]
        sol = ref_solution['parameter'].detach().clone().cpu().numpy()
        relative_error = abs((estimation - sol)/sol)

        ax1 = axs[0]
        im_est = ax1.imshow(estimation, cmap = 'viridis', aspect = 'auto',
                            interpolation='none')
        ax1.set_title('$K_s^{(i)} \ [m^{1/3}.s^{-1}]$')
        ax1.set_xlabel('$K_s$ component')
        ax1.set_ylabel('L-BFGS iterations')
        ax1.set_xticks(range(model.dim_k))
        fig.colorbar(im_est)

        ax2 = axs[1]
        im_err = ax2.imshow(relative_error*100, cmap = 'viridis', aspect = 'auto',
                            interpolation='none')
        ax2.set_title(r'$ \vert \frac{K_s^{(i)} - K_s^{true}}{K_s^{true}} \vert \ [\%] $', y = 1.03)
        ax2.set_xlabel('$K_s$ component')
        ax2.set_xticks(range(model.dim_k))
        fig.colorbar(im_err)

        fig.tight_layout()