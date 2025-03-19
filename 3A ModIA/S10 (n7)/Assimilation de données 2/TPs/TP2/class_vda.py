#
# Class to solve the complete VDA process
# This class includes all the FEniCS solvers
#

from dolfin import *
from fenics import *
import matplotlib.pyplot as plt
from utils import *
import scipy.optimize as spo
from pdb import set_trace
import sys
import numpy.linalg as npl
import pandas as pd



class vda_river(object):

  def __init__(self, model_type, param_mesh, param_elements, param_geom, param_bc, param_VDA):
  
    """
    Initialize the vda_river model object
    Parameters : 
    ------------
    model_type     : str, describes the model, value is: 'linear_stationnary' or 'nonlinear_stationnary'
    param_mesh     : dict, contains the parameter of the mesh: mesh nodes, the upstream and downstream location 
    param_elements : dict, contains the parameters to construct the FE: type and degree of elements
    param_geom     : dict, contains the parameters to construct the geometry of the problem: bathymetry, Href (array)
    param_bc       : dict, contains the parameters of Dirichler BC: values at upstream and downstream (if not Neumann bc)
    param_VDA      : dict, contains the parameters for the VDA process: alpha the weight coefficient, the observations Hobs (array) and their locations ind_obs
    """
    
    self.model_type = model_type
    
    # Test if the model_type exists
    if not(self.model_type == 'nonlinear_stationnary' or self.model_type == 'linear_stationnary') : 
      print('This model type does not exist or has not been implemented yet')
      
    # Create  (mesh, bdry_up, bdry_down) FEniCS objects
    (self.mesh, self.bdry_up, self.bdry_down) = create_mesh(param_mesh)
    
    # Create the FE space V
    self.V = FunctionSpace(self.mesh, param_elements['type of elements'], param_elements['degree of freedom']) 
    self.nb_cell = param_mesh['number cells']
    self.L = param_mesh['length river']
                                 
    self.H_up = param_bc['H bc upstream']       # Dirichlet BC
    self.H_down = param_bc['H bc downstream']   # Dirichlet BC (not employed if Neumann)
    self.bc_type_out = 'dirichlet'              # MAY BE CHANGED
    
    # Create FEniCS vectors
    self.b_1st = array_to_FEfct(self.V, param_geom['bathymetry array']) # bathy_1st
    self.b = array_to_FEfct(self.V, param_geom['bathymetry array']) # init the bathy to bathy_1st
    self.Href = array_to_FEfct(self.V,param_geom['Href array'])     # H_ref 
    self.b_t = None
    self.error_b_med = []
    self.path_out = './results/'

    if not(len(param_VDA)==0):
      self.alpha = param_VDA['regul. coeff.'] 
      self.Hobs = array_to_FEfct(self.V,param_VDA['obs']) # create Hobs as a FEniCS function
      self.ind_obs = array_to_FEfct(self.V, param_VDA['obs index']) # ind_obs = 1 if obs available, 0 otherwise. create as array
      self.b_b = array_to_FEfct(self.V, param_VDA['background']) # create b_b as a FEniCS function
      
    else: # param_VDA is not provided
      self.alpha = 1.
      self.Hobs = Constant(0.0) # Null FEniCS object 
      self.ind_obs = None # null array
      self.b_b = Constant(0.0) # Null FEniCS object 
          
    # Type of regularization
    if 'regul. term form' in param_VDA.keys():
      self.jreg_type = param_VDA['regul. term form']
    else: 
      self.jreg_type = 'grad' # Default value

    # Iteration number of the minimization algo
    self.n_iter = 0
    
    # Matrix C^0.5 with C the covariance matrix, initialized to the identity matrix
    self.C_chol = np.eye(param_mesh['number cells']) 

      
  ###########################
  # The complete VDA process
  def VDA(self,b0,b_t): 
  
    """
    Perform the complete VDA iterative process
    NB. The bathymetry self.b is automatically updated

    The results of the VDA (cost, the gradient, norm of the gradient, control value vs iterations, RMSE) are saved in './results'
    The results are saved at each iteration of the optimization process. 
    The optimization process can be re-resumed from the last iteration.
    Parameters : 
    self  : vda_river object
    b0    : array, the 1st guess bathymetry
    b_t   : array, the target bathymetry
    Returns:
    """
    
    self.b_t = b_t
    
    # convert
    b0_FE = array_to_FEfct(self.V,b0)
    b_t_FE = array_to_FEfct(self.V,b_t)
    b_1st_np = FEfct_to_array(self.mesh,self.b_1st)
    b_b_np = FEfct_to_array(self.mesh,self.b_b)
    Hobs_np = FEfct_to_array(self.mesh,self.Hobs)
    Href_np = FEfct_to_array(self.mesh,self.Href)
    
    #self.b_b = np.linspace(b_t[0],b_t[-1],np.size(b_t)) - np.abs(np.linspace(b_t[0],b_t[-1],np.size(b_t))[500] - b_t[500])

    # Test if dimensions issue
    if not(self.nb_cell == np.size(b_t)) or not (self.nb_cell == np.size(b0))\
       or not (self.nb_cell == np.size(Href_np)) or not(self.nb_cell == np.size(Hobs_np)):
      sys.exit('VDA callback_fct: dimensions mismatch (1)')

    # Save parameters in self.path_out directory
    np.save(self.path_out+'bathy_t.npy', b_t)
    np.save(self.path_out+'Hobs.npy', Hobs_np)

    # init
    cost = []; cost_reg = []; cost_obs = []; grad_cost = []; b_vs_ite = []; H_vs_ite = []
    RMSE_b = []; RMSE_H = []
    
    ######################
    # CallBack
    ######################
    def callback_fct(bc) :
      # callback function used at each iteration of minimization (spo.minimize() routine)
      # Print out results & save results in files
      # Parameters
      # bc: array, bathymetry (the current point)
      # Returns

      print("\nentering callback_fct...")
      # Update self.b
      if self.jreg_type == 'b_b': 
        #bc = np.dot(self.C_chol, bc) + FEfct_to_array(self.mesh, self.b_b) # inverse change of variable
        self.b = array_to_FEfct(self.V, bc)
      else:
        self.b = array_to_FEfct(self.V, bc)

      # median error value made on the true bathymetry
      error_b_np = FEfct_to_array(self.mesh,self.b) - self.b_t
      error_b_med = np.median( np.abs(error_b_np) )
      self.error_b_med.append( error_b_med )
      
      # Solve the direct model
      Hb = self.Direct_Model(self.b)
      Hb_np = FEfct_to_array(self.mesh, Hb)

      # Compute the cost fct terms & save in arrays vs iterations
      if self.jreg_type == 'b_b':
        cost_prov, costobs_prov, costreg_prov = self.Cost(array_to_FEfct(self.V, bc), Hb)
      else:
        cost_prov, costobs_prov, costreg_prov = self.Cost(self.b, Hb)
        
      # save value for each iteration
      cost.append(cost_prov) # total cost fct
      cost_obs.append(costobs_prov) # misfit term
      cost_reg.append(costreg_prov) # regularization term
              
      # Solve the adjoint model
      Pb_FE = self.Adjoint_Model(self.b,Hb)
      Pb_np = FEfct_to_array(self.mesh,Pb_FE)
      
      # Compute the gradient norm & save in array vs iteration
      if self.jreg_type == 'b_b':
        djnorm_prov = self.Gradient_Cost(array_to_FEfct(self.V, bc), Hb, Pb_FE)[1] # norm of the gradient power 2
      else:
        djnorm_prov = self.Gradient_Cost(self.b, Hb, Pb_FE)[1] # norm of the gradient power 2
      
      # save value for each iteration
      grad_cost.append(djnorm_prov) # norm of the gradient power 2

      # initial weight of the regularization term      
      if (self.jreg_type=='grad') and (self.n_iter==0): # @1st iteration if 'grad'
        beta = 0. # TO BE ADJUSTED
        self.alpha = beta * costobs_prov / costreg_prov # tune the initial value of alpha * TO BE IMPROVED
        print("\nInitial weight of Jreg setting @ ite =", self.n_iter,". beta =", beta*100.,"%. alpha=", self.alpha); print(' ')
      elif (self.jreg_type=='b_b') and (self.n_iter==1):  # @2nd iteration if 'b_b'
        if (costreg_prov > 1.e-12):
          beta = 0. # TO BE ADJUSTED
          self.alpha = beta * costobs_prov / costreg_prov # tune the initial value of alpha * TO BE IMPROVED
          print("\nInitial weight of Jreg setting @ ite =", self.n_iter,". beta =", beta*100.,"%. alpha=", self.alpha); print(' ')
        else:
          sys.exit("callback function: jreg_type='b_b', difficulty to set up the initial value of alpha...")
          
      # decreasing regularization coeff. 
      if (self.n_iter%10 == 0 and self.n_iter!=0):	
        if (self.jreg_type=='grad'):
          self.alpha *= 0. # TO BE ADJUSTED
        else:
          self.alpha *= 0. # TO BE ADJUSTED
          
      # dimensions test
      if not(self.nb_cell == np.size(FEfct_to_array(self.mesh,self.b))) or not(self.nb_cell == np.size(Hb_np))\
         or not(self.nb_cell == np.size(Pb_np)):
        sys.exit('VDA callback_fct: dimensions mismatch (2)')

      # save value for each iteration
      b_vs_ite.append([bc])
      H_vs_ite.append([Hb_np])
      
      mse_b = (np.square(bc - b_t)).mean; mse_H = (np.square(Hb_np - Hobs_np)).mean
      RMSE_b.append( np.sqrt( np.mean(bc - b_t)**2 ) ) 
      RMSE_H.append( np.sqrt( np.mean(Hb_np - Hobs_np)**2 ) )

      # Print out frequently
      if np.mod(self.n_iter,5)==0:
        print(""); print("@ iteration ",self.n_iter)
        print("total cost J = ",cost[-1],", Jobs = ",cost_obs[-1],", (alpha*Jreg) = ",cost_reg[-1],". alpha_reg = ",self.alpha)
        print("Normalized values")
        print("J =",cost[-1]/cost[0],", Jobs =",cost_obs[-1]/cost[0],", (alpha*Jreg) =",cost_reg[-1]/cost[0])
        print("norm of the gradient**2 =",grad_cost[-1])
        print("")

      # Save in file at each iteration
      np.save(self.path_out+'cost.npy',cost)
      np.save(self.path_out+'cost_obs.npy',cost_obs)
      np.save(self.path_out+'cost_reg.npy',cost_reg)
      np.save(self.path_out+'grad_cost.npy',grad_cost)
      np.save(self.path_out+'bathy_vs_ite.npy',b_vs_ite)
      np.save(self.path_out+'H_vs_ite.npy',H_vs_ite)
      np.save(self.path_out+'RMSE_b.npy',RMSE_b)
      np.save(self.path_out+'RMSE_H.npy',RMSE_H)

      # New iteration
      self.n_iter += 1
      
    
    ################  
    # Initialization: perform callback_fct() from the 1st guess value
    if self.jreg_type == 'b_b': 
      b0 = b0
      #b0 = npl.solve(self.C_chol, (b0 - FEfct_to_array(self.mesh,self.b_b))) # change of variable
    	
    callback_fct(b0)
    
    # Error steps for different values (10 / 20 / 30 cm)
    c1,c2,c3 = Function(self.V),Function(self.V),Function(self.V)
    c1.vector()[:] = 0.1 * np.ones(len(c1.vector()))
    c2.vector()[:] = 0.2 * np.ones(len(c2.vector()))
    c3.vector()[:] = 0.3 * np.ones(len(c3.vector()))
    cost_1 = assemble(0.5 * (c1 * c1) * dx)
    cost_2 = assemble(0.5 * (c2 * c2) * dx)
    cost_3 = assemble(0.5 * (c3 * c3) * dx)
    np.save(self.path_out+'fitting-threshold.npy', np.array([cost_1,cost_2,cost_3]))

    #
    # Equality constraint on the control variable i.e b or k if change of var has been made
    #
    
    #constr = None # default value = no constraint # MAY BE CHANGED
    #print('class_vda / VDA: no equlaity / inequality constraint on the control var (type any key to resume)')
    print('class_vda / VDA: constraints on the control var are imposed (type any key to resume)')
    constr = self.ControlConstraint(b_t, b_1st_np, Hobs_np) # inequality / equality constraints on b

    # case control var = k (change of variable)
    #k_t = npl.solve(self.C_chol, (b_t - FEfct_to_array(self.mesh,self.b_b)))
    #constr = self.ControlConstraint(k_t) 
    
    #
    # parameters of the minimizer: min evolution of the cost, gradient and max of iterations
    #
    ftol = 1e-15; gtol = 1e-10; maxiter = 50 # MAY BE ADJUSTED
    
    print(' '); print('minimization method = L-BFGS-B, min evol. of j=', ftol,', min gradient value=', gtol,', maxiter=', maxiter); print(' ')
    ans = input("type any key to start the VDA process...")
    plt.close()
      
    # Call the minimizer (with callback)
    res = spo.minimize(self.Cost_np, b0, jac=self.Gradient_Cost_np, method='L-BFGS-B',\
    options={'disp':True, 'ftol':ftol, 'gtol':gtol, 'maxiter':maxiter}, callback=callback_fct, bounds = constr) 

    # update self.b value
    b_star = res.x # the computed optimal value of b

    if self.jreg_type == 'b_b':
      b_star = b_star
      #b_star = np.dot(self.C_chol, b_star) + FEfct_to_array(self.mesh,self.b_b)  # inverse change of variable
    
    self.b = array_to_FEfct(self.V,b_star)

    # Compute H_star corresponding to b_star (the optimal value)
    H_star = self.Direct_Model(array_to_FEfct(self.V,b_star))
    H_star_np = FEfct_to_array(self.mesh,H_star)
      
    # write in files
    print("\nclass_vda.py: write optimal values in ", self.path_out)
    np.save(self.path_out+'bathy_star.npy', b_star)
    np.save(self.path_out+'H_star.npy', H_star_np)
    np.save(self.path_out+'median_error_b.npy',np.array(self.error_b_med))

  #  
  #
  #
  def Observations(self, b, noise_bool) :
    """
    Built up the observations FE function self.Hobs = the direct model output, FEniCS function type
    Parameters:
    self     : vda_river object
    b : bathymetry, FEniCS fct type
    noise_bool: 0 for perfect data, 1 for perturbed synthetic data (noise)
    Returns:
    Hobs, FEniCS object
    """
    # noise definition
    noise_np = np.zeros(self.nb_cell)
    if noise_bool == 1 : 
      noise_np = np.random.normal(0., 0.05, np.size(noise_np)) # (standard deviation in SI units) # MAY BE CHANGED
     
    # solve the direct model & convert
    H = self.Direct_Model(b)
    H_np = FEfct_to_array(self.mesh,H)

    # add noise
    Hobs_np = H_np + noise_np 
    
    # FEniCS object
    self.Hobs = array_to_FEfct(self.V, Hobs_np)
    
    return self.Hobs

  

  def Direct_Model(self, b) : 
    """
    Solve the direct model (in the linearized steady-state case) by FEM (Fenics)
    -----------
    Parameters 
    self   : vda_river object
    b : bathymetry of the river, FEniCS function object
    ---------
    Returns
    H : the solution of the linearized direct model, FEniCS function object 
    """
    H = TrialFunction(self.V)  # trial function, unknown of the direct model 
    phi = TestFunction(self.V) # test function

    # Boundary conditions
    self.bc_type_out = 'dirichlet' 
    #bc_type_out = 'neumann'
    bc_up = DirichletBC(self.V, Constant(self.H_up), self.bdry_up) 
    bc_diri = [bc_up]
    if self.bc_type_out == 'dirichlet':
      bc_down = DirichletBC(self.V, Constant(self.H_down), self.bdry_down) 
      bc_diri.append(bc_down)

    # LHS of the equation
    a = inner( grad(H) , grad( 0.3*(self.Href-b)/abs(grad(self.Href)[0]) * phi ) )*dx + grad(H)[0]*phi*dx # original linear model
    
    # RHS of the equation
    L =  grad(b)[0]*phi*dx

    # The unknown field
    H = Function(self.V)
    
    # Call the linear algebra solver, including the Dirichlet BC 
    solve(a == L, H, bc_diri)

    print("   the direct model has been solved (with ",self.bc_type_out," bc at outflow)")
    
    return H      

  #
  # Cost function
  def Cost(self,b,H) :
    """
    Compute the cost function
    Parameters:
    self   : vda_river object
    b      : control variable, FEniCS fct type
    H      : the state (water elavation), solution of the direct model, FEniCS fct type
    Returns 
    the total cost, the observation cost and alpha * Jreg ; type: Float from FeniCS object...
    """
    # self.ind_obs : index = 1 if obs available, 0 otherwise, array type
    Jobs = assemble( pow(H-self.Hobs,2) * self.ind_obs * dx)
    #Jobs = assemble( pow(H-self.Hobs,2) * dx ) # all observed 
    
    #Jreg = ...
    if not(self.alpha == 0.):
      if self.jreg_type == 'grad':  # regularization term = "grad"
        mean_slope = (self.H_down-self.H_up)/self.L
        Jreg = assemble( pow(grad(b)[0] - mean_slope, 2)*dx ) # MAY BE CHANGED
                
      elif self.jreg_type == 'b_b':  # regularization term = "background"
        Jreg = assemble( pow(b - self.b_b, 2) * dx ) # MAY BE CHANGED
        
      else : 
          sys.exit('The regularization ' + self.jreg_type + ' is not implemented, it must be either <b_b> or <grad>')
    else:	
      Jreg = 0.0

    # Add the different costs
    Jtotal = Jobs + self.alpha * Jreg  
     
    return Jtotal, Jobs, self.alpha*Jreg

  #    
  # Adjoint model
  def Adjoint_Model(self, b, Hb) : 
  
    """
    Solve the adjoint model of the linearized steady-state equation by FEM (Fenics)
    -----------
    Parameters 
    self   : vda_river object
    b : bathymetry of the river, FEniCS function object
    Hb    : the solution of the direct model (at b given), FEniCS function object 
    ---------
    Returns
    p : the solution of the adjoint model, FEniCS function object 

    """ 
    p = TrialFunction(self.V)  # trial function, the adjoint state, solution of the adjoint eqn
    phi = TestFunction(self.V) # test function
        
    # Boundary conditions
    bc_up = DirichletBC(self.V, Constant(0.0), self.bdry_up)  # upstream boundary condition
    bc_diri = [bc_up]
    if self.bc_type_out == 'dirichlet':
      bc_down = DirichletBC(self.V, Constant(0.0), self.bdry_down) # downstream boundary condition
      bc_diri.append(bc_down)
    
    # LHS of the eqn:  <d_H A*(b,Hb).p,phi>
    a = inner( grad( (0.3*(self.Href-b) / abs(grad(self.Href)[0])) * p ) , grad(phi) )*dx - grad(p)[0]*phi*dx

    # RHS of the eqn: <d_H J*(b,Hb),phi> the observations term
    #    self.ind_obs : index = 1 if obs available, 0 otherwise, array type
    L = 2. * (Hb - self.Hobs) * self.ind_obs * phi * dx 
    
    p = Function(self.V)

    solve(a == L, p, bc_diri)

    print("   the adjoint model has been solved")
      
    return p

  #
  # Gradient
  def Gradient_Cost(self, b, Hb, Pb, flag_val=False): 
  
    """
    Expression of the gradient in function of the state and the adjoint solutions
    The gradient is solved in the FE basis
    -----------
    Parameters
    self    : vda_river object
    b      : control variable, FEniCS function object
    Hb     : state, solution of the direct model, FEniCS function object 
    Pb     : adjoint state, solution of the adjoint model, FEniCS function object
    flag_val : boolean, gradient test case or not
    ---------
    Returns
    grad_j : value of the gradient of the cost function, FEniCS function object
    AND either the gradient norm power 2 OR the scalar prod with 1 if it is a validation case (flag_val=True), FEniCS fct object
    """
    
    grad_j = TrialFunction(self.V) # trial function, gradient of the cost function
    phi = TestFunction(self.V)     # test function

    # Dirichlet bc: none
    bc_diri = [] 
    
    # LHS of the "equation": the gradient value
    a = grad_j * phi * dx

    # RHS = the gradient expression: -<[d_b A(H,b) - d_b L(b)] . P ,phi>
    L = grad(phi)[0] * Pb * dx
    L -= ( 0.3 * phi * Pb * div(grad(Hb)) / abs(grad(self.Href)[0]) ) * dx
    
    # if there is a regularization term, RHS is enriched
    if not(self.alpha==0.):

        if self.jreg_type == 'grad':  # regularization term = "grad"
          mean_slope = (self.H_down-self.H_up)/self.L
          L += self.alpha*2. * inner( (grad(b)[0] - mean_slope),grad(phi)[0] ) * dx 

        elif self.jreg_type == 'b_b':  # regularization term = "background"
          #k_array = FEfct_to_array(self.mesh,b) 
          #aux_np = npl.solve(self.C_chol.T, k_array)
          #aux = array_to_FEfct(self.V,aux_np)
          #L += self.alpha*2. * aux * phi * dx
          L += self.alpha*2. * (b-self.b_b) * phi * dx  
          
    grad_j = Function(self.V)

    solve(a == L, grad_j, bc_diri)

    if flag_val : # case "gradient test" (validation), scalar prod with 1 is returned
      print("routine Gradient_Cost: this is a gradient test")
      return grad_j, assemble( grad_j*interpolate(Constant(1.0),self.V)*dx )
      
    if self.jreg_type == 'b_b': 
      grad_j_np = FEfct_to_array(self.mesh,grad_j)
      grad_j_np = np.dot(self.C_chol.T, grad_j_np)
      grad_j = array_to_FEfct(self.V,grad_j_np)
      
    # usual case: grad_j & norm(grad_j)**2 are returned
    return grad_j, assemble( pow(grad_j,2)*dx )

  #
  # Routines_np useful for the routine spo.minimize 
  #
  def Cost_np(self,k_array) :
  
    """
    Compute the total cost function term with in array format
    This is required by the spo.minimize() routine
    Parameters:
    self         : vpa_process object
    k_array      : control variable, array type
    Returns:
    the total cost, array type
    """ 
    if self.jreg_type == 'grad':
      b = array_to_FEfct(self.V,k_array)
      H = self.Direct_Model(b)
      # compute the cost
      j_np = self.Cost(b,H)[0] # type=Float, a-priori...
      
    elif self.jreg_type == 'b_b':
      #b_array = np.dot(self.C_chol,k_array) + FEfct_to_array(self.mesh,self.b_b) # change of variable
      k = array_to_FEfct(self.V,k_array)
      #k = array_to_FEfct(self.V,k_array)
      H = self.Direct_Model(k)
      # compute the cost
      j_np = self.Cost(k,H)[0]
    
    return j_np

  

  # useful for the routine spo.minimize    
  def Gradient_Cost_np(self,k_array) :
  
    """
    Compute the gradient of the cost function but with the scipy optimize format 
    This is required by the spo.minimize routine
    Parameters:
    self         : model object
    b_array      : control variable, array type
    Returns:
    the gradient, array type
    """  
    
    if self.jreg_type == 'grad':
      b = array_to_FEfct(self.V,k_array)
      H = self.Direct_Model(b)
      
      # solve the adjoint model
      P = self.Adjoint_Model(b,H)
      
      # compute the gradient
      dj = self.Gradient_Cost(b, H, P, flag_val=False)[0]
      
    elif self.jreg_type == 'b_b':
      # solve the direct model
      #b_array = np.dot(self.C_chol,k_array) + FEfct_to_array(self.mesh,self.b_b) # inverse change of variable
      k = array_to_FEfct(self.V,k_array)
      #k = array_to_FEfct(self.V,k_array)
      
      # Solve direct model
      H = self.Direct_Model(k)
      
      # solve the adjoint model
      P = self.Adjoint_Model(k,H)
      
      # compute the gradient
      dj = self.Gradient_Cost(k, H, P, flag_val=False)[0]

    # convert
    dj_np = FEfct_to_array(self.mesh,dj)
    
    return dj_np 

  
  #############################################
  # Constraints on the optimal control variable
  ##############################################
  def ControlConstraint(self, var, b_1st, Hobs):
    '''
    Impose inequality and/or equality constraint on the control variable var
    Parameters:
    var, b_1st, Hobs: arrays
    Returns:
    constr, array [nb_cell,2]
    '''
    # inequality constraints
    v_min = b_1st - np.mean(Hobs - b_1st) * 2. # MAY BE CHANGED
    v_max = Hobs - 1.
    
    # equality constraint at one location (=measurement available)
    ind_measure = -1 # int(self.nb_cell/1000.) # point index # MAY BE CHANGED
    v_min[ind_measure] = b_1st[ind_measure]
    v_max[ind_measure] = v_min[ind_measure]
      
    # constraint array
    constr = np.zeros((self.nb_cell, 2))
    constr[:,0] = v_min; constr[:,1] = v_max
    
    return constr

  
  #################################
  # Covariance matrices
  # (to introduce non-trivial metrics)
  #################################
  
  def Cov_Expo(self,L):
    """
    compute the exponential covariance kernel & its Cholesky factorization
    Parameters:
    self : vda_river object
    L    : the regularization scale   
    Returns:
    The matrix self.C_chol
    """
    x, xp = self.mesh.coordinates()[:,0], self.mesh.coordinates()[:,0]
    i = 0
    n = np.size(self.mesh.coordinates()[:,0])
    C = np.zeros((n,n))
    
    for xi in x :
      C[i,:] =  np.exp(-( np.abs(xp-xi) )/L)
      i +=1
      
    self.C_chol = npl.cholesky(C)

    # Test 
    #w,v = npl.eig(C)
    #self.C_chol = v @ np.diag(np.sqrt(w)) @ v.T

    return self.C_chol
    
  #   
  def Cov_Gaussian(self,L):
  
    """
    compute the Gaussian covariance kernel and its Cholesky factorization
    Parameters:
    self : vda_river object
    L    : the regularization scale   
    Returns:
    The matrix self.C_chol
    """
    
    x = self.mesh.coordinates()[:,0]
    xp = self.mesh.coordinates()[:,0]
    i = 0
    n = np.size(self.mesh.coordinates()[:,0])
    
    C = np.zeros((n,n))
    
    for xi in x :
      C[i,:] =  np.exp(-0.5*( (xp-xi)**2 )/L**2) / (np.sqrt(2*np.pi)*L)
      i +=1
      
    self.C_chol = npl.cholesky(C)
    return self.C_chol

  
 
