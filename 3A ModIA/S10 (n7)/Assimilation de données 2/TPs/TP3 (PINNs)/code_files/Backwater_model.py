#%% Library imports

import torch
import numpy as np

#%% Physical constants

g = 9.81 #Gravity in m/s²
q = 1 # Volumetric flow rate per unit width, in m^2/s
h_BC = 1 # Water height boundary condition for reference solution
regime = "subcritical" # Flow regime

hc = (q**2/g)**(1/3)

bathy = torch.tensor(np.load('bathy.npy'))/5
bathy_prime = bathy[1:] - bathy[:-1]
bathy_prime = torch.hstack((bathy_prime, bathy_prime[-1]))
bathy_x = torch.arange(bathy.shape[0])

#%% Helper functions

def numpy_interpolator(device, x, domain, y):
    """
    Simple numpy interpolator for Pytorch tensors. 
    """

    y_interpolated = torch.from_numpy(np.interp(x.detach().clone().cpu().numpy().flatten(), 
                                             domain.detach().clone().cpu().numpy().flatten(), 
                                             y.detach().clone().cpu().numpy().flatten())).view(-1, 1).to(device)
    
    return y_interpolated

def bathymetry_interpolator(device, x):
    """
    Bathyemetry function that outputs the value of the bathymetry and its derivative for any location x in the domain. 
    """

    b = numpy_interpolator(device, x, bathy_x, bathy)
    b_prime = numpy_interpolator(device, x, bathy_x, bathy_prime)

    return b, b_prime

def Ks_function(x, k, model, col):
    """
    Function used for the reconstruction of the spatially-distributed Ks parameter with P0 or P1 interpolation. 
    """
    
    if (model.k_interpolation == 'P0'):
        
        subdomains = torch.linspace(col.variable_min, col.variable_max,
                                    k.shape[0] + 1)      
        
        indices = (torch.bucketize(x, subdomains) - 1).clamp(min = 0, max = k.shape[0] - 1) 
        
        return k[indices]
    
    elif (model.k_interpolation == 'P1'):
        
        if (k.shape[0] == 1):
            return k*torch.ones(x.shape[0])
        
        else:
            
            subdomains = torch.linspace(col.variable_min, col.variable_max,
                                        k.shape[0])   
            
            subdomains_sizes = subdomains[1:] - subdomains[:-1]
            
            indices = (torch.bucketize(x, subdomains) - 1).clamp(min = 0, max = k.shape[0] - 1) 
        
            alpha = (x - subdomains[indices])/subdomains_sizes[indices]
        
            return k[indices+1]*alpha + k[indices]*(1-alpha)
    
#%% Loss functions

def J_res(model, col, domain):
    """
    Loss function for physical model residual calculated on a given domain. 
    """

    Ks = model.k_ref*Ks_function(domain, model.k, model, col)
    b_prime = bathymetry_interpolator(model.device, domain)[1]

    # You have to complete the rest of the J_res loss function here !
    
    #pass
    h = torch.clamp(model(domain),min=1e-6)
    #h_prime = torch.autograd.grad(h, domain, torch.ones_like(h), create_graph=True)[0]
    h_prime = torch.autograd.grad(h,
                                domain, 
                                grad_outputs=torch.ones_like(h), 
                                create_graph=True,
                                retain_graph=True)[0]
    j = q**2/((Ks**2)*(h**(10/3)))   #
    Fr_2 = (q**2)/(g*h**3)           #
    #return torch.norm(h_prime + (b_prime + j)/(1 - Fr_2 + 1e-6))**2
    return torch.mean((h_prime + (b_prime + j)/(1 - Fr_2))**2) #

def J_obs(model, obs):
    """
    Loss function for observations discrepancy. 
    """
    
    # You have to complete the J_obs loss function here !
    
    #pass
    #print(obs.data[:,0])
    #print(obs.data[:,1])
    return torch.mean((model(obs.data[:,0].view(-1,1)) - obs.data[:,1].view(-1,1))**2)     
    # NOTE : torch.view permet de redimensionner le tenseur, -1 permet d'adapter la première dimension automatiquement

def J_BC(h_tilde_BC):
    """
    Loss function for boundary condition. 
    """
    
    # You have to complete the J_BC loss function here !
    
    #pass
    return torch.mean((h_tilde_BC - h_BC)**2)

def J(model, col, obs, domain):
    """
    Total J function.
    """
        
    if (obs.regime == 'subcritical'):
        h_tilde_BC = model(col.all[-1])
    elif (obs.regime == 'supercritical'):
        h_tilde_BC = model(col.all[0])
        
    if model.normalize_J:
        model.J_res_0 = J_res(model, col, col.all).detach().clone()
        model.J_BC_0 = J_BC(h_tilde_BC).detach().clone()
        if (obs.N_obs == 0):
            model.J_obs_0 = torch.inf
        else:
            model.J_obs_0 = J_obs(model, obs).detach().clone()
        model.normalize_J = False
   
    return (model.lambdas['res']*1/model.J_res_0*J_res(model, col, domain),
            model.lambdas['obs']*1/model.J_obs_0*J_obs(model, obs),
            model.lambdas['BC']*1/model.J_BC_0*J_BC(h_tilde_BC))   


#%% Reference solution from classical solver RK4
    
def compute_ref_solution(model, col, k, dx):
    """
    Function used to compute the reference solution with RK4 method, for observations generation and solutions comparison. 
    """
    
    k = k.float()
    
    domain = torch.linspace(col.variable_min, col.variable_max, 
                            int((col.variable_max-col.variable_min)/dx)).view(-1, 1)
    
    bathy = bathymetry_interpolator(model.device, domain)[0]
    slope = -bathymetry_interpolator(model.device, domain)[1]
    
    if (regime == 'subcritical'):
        def backwater_model(x, h, k):
            Fr = q/(g*h**3)**(1/2)
            return -(numpy_interpolator(model.device, x, domain, slope)-(q/k)**2/h**(10/3))/(1-Fr**2)
    elif (regime == 'supercritical'):
        def backwater_model(x, h, k):
            Fr = q/(g*h**3)**(1/2)
            return (numpy_interpolator(model.device, x, domain, slope)-(q/k)**2/h**(10/3))/(1-Fr**2)
        
    def RK4_integrator(k):
        
        i = domain.shape[0]-1

        list_h = []
        list_h.append(h_BC)
        
        list_hn = []
        list_hn.append(((q**2/(numpy_interpolator(model.device, domain[i], domain, slope)*Ks_function(domain[i], k, model, col)**2))**(3/10)).item())
        if (regime == 'subcritical'):
        
            while(i > 0):
            
                k1 = backwater_model(domain[i], list_h[-1], Ks_function(domain[i], k, model, col))
                k2 = backwater_model(domain[i]-dx/2, list_h[-1]-dx/2*k1, Ks_function(domain[i], k, model, col))
                k3 = backwater_model(domain[i]-dx/2, list_h[-1]-dx/2*k2, Ks_function(domain[i], k, model, col))
                k4 = backwater_model(domain[i]-dx, list_h[-1]-dx*k3, Ks_function(domain[i], k, model, col))
                
                list_h.append((list_h[-1] + dx/6*(k1+2*k2+2*k3+k4)).item())
                
                if (numpy_interpolator(model.device, domain[i], domain, slope) < 0):
                    list_hn.append(np.nan)
                else:
                    list_hn.append(((q**2/(numpy_interpolator(model.device, domain[i], domain, slope)*Ks_function(domain[i], k, model, col)**2))**(3/10)).item())
                    
                if (list_h[-1] < hc):
                    raise Warning('You reached supercritical regime !')
                    break
                
                i -= 1
                
            return np.flip(np.array(list_h, dtype=np.float32)).reshape(-1, 1), np.flip(np.array(list_hn, dtype=np.float32)).reshape(-1, 1)
        
        elif (regime == 'supercritical'):
            
            while(i > 0):
            
                k1 = backwater_model(domain[i], list_h[-1], Ks_function(domain[i], k, model, col))
                k2 = backwater_model(domain[i]+dx/2, list_h[-1]+dx/2*k1, Ks_function(domain[i], k, model, col))
                k3 = backwater_model(domain[i]+dx/2, list_h[-1]+dx/2*k2, Ks_function(domain[i], k, model, col))
                k4 = backwater_model(domain[i]+dx, list_h[-1]+dx*k3, Ks_function(domain[i], k, model, col))
                
                list_h.append((list_h[-1] + dx/6*(k1+2*k2+2*k3+k4)).item())
                
                if (numpy_interpolator(model.device, domain[i], domain, slope) < 0):
                    list_hn.append(np.nan)
                else:
                    list_hn.append(((q**2/(numpy_interpolator(model.device, domain[i], domain, slope)*Ks_function(domain[i], k, model, col)**2))**(3/10)).item())
                    
                if (list_h[-1] > hc):
                    raise Warning('You reached subcritical regime !')
                    break
                
                i -= 1
                
            return np.array(list_h, dtype=np.float32).reshape(-1, 1), np.array(list_hn, dtype=np.float32).reshape(-1, 1)
        
    results = RK4_integrator(k)
    
    h = torch.tensor(results[0].copy())
    h_n = torch.tensor(results[1].copy())
    
    h_c = hc*torch.ones(domain.shape[0], 1)
        
    return {'solution': h, 
            'dx': dx,
            'critical height':h_c, 
            'normal height':h_n, 
            'bathymetry':bathy, 
            'bathymetry_col':bathymetry_interpolator(model.device, col.grid)[0], 
            'domain':domain, 
            'parameter':k,
            'parameter_function':Ks_function(domain, k, model, col), 
            'regime':regime}
