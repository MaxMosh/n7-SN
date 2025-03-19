import torch 
import torch.nn as nn
from torch.nn.utils import parameters_to_vector as Params2Vec

import time 

import display

class PINN (nn.Module):
    """
    This is the PINN class, the Neural Network will be defined as a instance of that class. 
    """
    
    def __init__(self, device, col, layers, k_0, k_ref, k_interpolation,
                 seed = None):
        """
        Initialization of the PINN, with the number of layers and the first guess for the physical parameter. 
        """
        
        super(PINN, self).__init__()
        
        # Seed for initialization reproductibility
        if (seed != None):
            torch.manual_seed(seed)
        
        self.device = device
            
        self.hidden = nn.ModuleList()
        self.layers = layers 
        self.activation = nn.Tanh() #You can change the activation function here !

        self.k_ref = k_ref  
        self.k = k_0.float()/self.k_ref   
        self.k_0 = k_0.detach().clone()

        self.k.requires_grad = True
            
        self.dim_k = self.k.shape[0]
        self.k_interpolation = k_interpolation
        
        self.variable_min = col.variable_min
        self.variable_max = col.variable_max
        
        #Input Layer
        input_layer = nn.Linear(self.layers[0], self.layers[1], bias = True)
        nn.init.xavier_normal_(input_layer.weight.data, gain = 1.0) #You can change the initialization method for theta here !
        nn.init.zeros_(input_layer.bias.data) #You can change the initialization method for theta here !
        self.hidden.append(input_layer)
        
        #Hidden layers
        for input_size, output_size in zip(self.layers[1:-1], self.layers[2:-1]):
            linear = nn.Linear(input_size, output_size, bias = True)
            nn.init.xavier_normal_(linear.weight.data, gain = 1.0) #You can change the initialization method for theta here !
            nn.init.zeros_(linear.bias.data) #You can change the initialization method for theta here !
            self.hidden.append(linear)
            
        #Output layer
        linear = nn.Linear(self.layers[-2], self.layers[-1], bias = False) 
        nn.init.xavier_normal_(linear.weight.data, gain = 1.0) #You can change the initialization method for theta here !
        self.hidden.append(linear)
        
        #Lists initialization
        self.list_J_train = []
        self.list_J_test = []
        
        self.list_y = []
        
        self.list_grad = []
        self.list_J_gradients = []
        
        self.list_params = []
        
        self.list_LBFGS_n_iter = []
        self.list_iter_flag = []
        
        self.list_theta_optim = []
        self.list_k_optim = []
        self.list_res_optim = []
        
        self.list_lr = []
        
        self.list_k = []
        self.list_grad_k = []
        self.list_k_matrix = []
        
        self.iter = 0
        self.iter_eval = 0
        self.alter_iter = 0
        
        self.list_subdomains = []
        
        self.lambdas = {'res' : 1, 'obs' : 1, 'BC' : 1}
        self.normalize_J = False
        self.end_training = False
        
        self.J_res_0 = torch.tensor(1.)
        self.J_obs_0 = torch.tensor(1.)
        self.J_BC_0 = torch.tensor(1.)
        
    def forward(self, input_tensor):
        """
        Forward method that will be called with model(input_tensor).
        """
        
        #Normalization layer
        input_tensor = (input_tensor-self.variable_min)/(self.variable_max-self.variable_min)
        
        #Forward
        for (l, linear_transform) in zip (range(len(self.hidden)), self.hidden):
            #For input and hidden layers, apply activation function after linear transformation
            if l < len(self.hidden) -1:
                input_tensor = self.activation(linear_transform(input_tensor))
            #For output layer, apply only linear tranformation
            else:
                output = linear_transform(input_tensor)
                
        return output
                    
    def train_model(self, J, col, obs, ref_solution, 
                    normalize_J = True, pre_train_iter = 100, 
                    alter_steps = 10, alter_freq = (50, 50), 
                    display_freq = (20, 100)):
        """
        Method used for training the model.
        """
        
        start_time = time.time()

        # Pre-training 
        self.lambdas = {'res' : 0, 'obs' : 1, 'BC' : 1} #You can change the values of the lambdas for pre-training here !
        self.normalize_J = normalize_J
        
        self.optimizing_theta = True
        self.optimizing_k = False
            
        optimizer = torch.optim.LBFGS(self.parameters(), 
                                      lr = 1, 
                                      max_iter = pre_train_iter - 1,
                                      max_eval = 10*pre_train_iter,
                                      line_search_fn = "strong_wolfe", 
                                      tolerance_grad = -1, 
                                      tolerance_change = -1)
            
        self.gradient_descent(J, optimizer, col, obs, 
                              ref_solution, start_time,
                              display_freq = display_freq)

        # Alternating minimization 
        self.lambdas = {'res' : 1, 'obs' : 1, 'BC' : 1} #You can change the values of the lambdas for alternating minimization here !

        for self.alter_step in range (alter_steps):
            
            #Minimizing on k
            self.optimizing_theta = False
            self.optimizing_k = True
            
            optimizer = torch.optim.LBFGS([self.k], 
                                          lr = 1, 
                                          max_iter = alter_freq[1] - 1,
                                          max_eval = 10*alter_freq[1],
                                          line_search_fn = "strong_wolfe", 
                                          tolerance_grad = -1, 
                                          tolerance_change = -1)
                
            self.gradient_descent(J, optimizer, col, obs, 
                                  ref_solution, start_time, 
                                  display_freq = display_freq)
            
            #Minimizing on theta
            self.optimizing_theta = True
            self.optimizing_k = False
                
            optimizer = torch.optim.LBFGS(self.parameters(), 
                                          lr = 1, 
                                          max_iter = alter_freq[0] - 1,
                                          max_eval = 10*alter_freq[0],
                                          line_search_fn = "strong_wolfe", 
                                          tolerance_grad = -1, 
                                          tolerance_change = -1)
            
            self.gradient_descent(J, optimizer, col, obs, 
                                  ref_solution, start_time, 
                                  display_freq = display_freq)
            
            self.alter_iter += 1
        
        self.end_training = True
        self.list_iter_flag.append(True)
                
    def gradient_descent(self, J, optimizer, col, obs, 
                         ref_solution, start_time, 
                         display_freq = (20, 100)):
        """
        Gradient descent method used during the training for updating parameters. 
        """
        
            
        def closure():  
            
            optimizer.zero_grad()
            
            self.J_train = J(self, col, obs, col.train)
            
            self.J_res_train = self.J_train[0]
            self.J_obs_train = self.J_train[1]
            self.J_BC_train = self.J_train[2]
            
            self.J_train = sum(self.J_train)
            
            self.J_train.backward(retain_graph = True)
            
            # Clipping the gradient to avoid diverging during the training
            nn.utils.clip_grad_norm_(self.parameters(), max_norm = 1e3, 
                                     norm_type = 2.0)
            
            # Simple constraint to the keep physical parameter box-constrained
            with torch.no_grad():
                self.k.clamp_(min = 10/self.k_ref, max = 100/self.k_ref)

            if (col.test_size > 0):
                
                self.J_test = J(self, col, obs, col.test)
                
                self.J_res_test = self.J_test[0].detach().clone()
                self.J_obs_test = self.J_test[1].detach().clone()
                self.J_BC_test = self.J_test[2].detach().clone()
                
                self.J_test = sum(self.J_test).detach().clone()
            
            self.iter_eval += 1
            
            with torch.no_grad():
                self.update_lists(col, optimizer)
            
            if (self.iter%display_freq[0]==0 and self.list_iter_flag[-1]):
                with torch.no_grad():
                    self.display_training(col, obs, ref_solution, 
                                          start_time, display_freq)
            
            return self.J_train
        
        optimizer.step(closure)
                        
    def update_lists(self, col, optimizer):
        """
        Method used for updating the model lists, to keep track of the values of interest during the training.
        """
        
        self.list_J_train.append([self.J_train.item(), self.J_res_train.item(), 
                                  self.J_obs_train.item(), self.J_BC_train.item()])
        
        if (col.test_size > 0):
            self.list_J_test.append([self.J_test.item(), self.J_res_test.item(), 
                                     self.J_obs_test.item(), self.J_BC_test.item()]) 

        self.list_y.append(self(col.grid).detach().clone().cpu().numpy())   
        
        list_params_iter = []
        list_params_grad = []
        for param in self.parameters():
            list_params_iter.append(param.detach().clone().cpu().numpy())
            if param.requires_grad:
                list_params_grad.append(param.grad.detach().clone())
            
        self.list_params.append(list_params_iter)
                
        if (len(list_params_grad) > 0):
            self.list_grad.append((torch.norm(Params2Vec(list_params_grad),p=2)).item())
        else:
            self.list_grad.append(0)
        
        self.list_lr.append(optimizer.state_dict()['param_groups'][0]['lr'])
            
        self.list_k.append(self.k.detach().clone().cpu().numpy())
        
        self.list_k_matrix.append(self.k.repeat_interleave(self.dim_k//self.dim_k).detach().clone().cpu().numpy())
    
        if self.optimizing_k:
            
            self.list_k_optim.append(True)
            self.list_theta_optim.append(False)
            
            self.list_grad_k.append(self.k.grad.detach().clone().cpu().numpy())
            
            J_k_res = torch.autograd.grad(self.J_res_train, self.k, 
                                          grad_outputs = torch.ones_like(self.J_res_train), 
                                          create_graph = True, retain_graph = True)[0].detach().clone()
            
        else:
            
            J_k_res = torch.tensor(0.)
            
            self.list_grad_k.append(torch.zeros(self.dim_k).detach().clone().cpu().numpy())
            
        J_k_res = torch.norm(J_k_res, p = 2)
            
        if self.optimizing_theta:
            
            self.list_k_optim.append(False)
            self.list_theta_optim.append(True)
            
            J_theta_res = []
            J_theta_obs = []
            J_theta_BC = []
            for param in self.named_parameters():
                if param[0] != 'k':
                    J_theta_res.append(torch.autograd.grad(self.J_res_train, param[1], 
                                                          grad_outputs = torch.ones_like(self.J_res_train), 
                                                          create_graph = True, retain_graph = True)[0].detach().clone().flatten().view(-1, 1))
                    J_theta_obs.append(torch.autograd.grad(self.J_obs_train, param[1], 
                                                          grad_outputs = torch.ones_like(self.J_obs_train), 
                                                          create_graph = True, retain_graph = True)[0].detach().clone().flatten().view(-1, 1))
                    J_theta_BC.append(torch.autograd.grad(self.J_BC_train, param[1], 
                                                          grad_outputs = torch.ones_like(self.J_BC_train), 
                                                          create_graph = True, retain_graph = True)[0].detach().clone().flatten().view(-1, 1))
                    
            J_theta_res = torch.cat(J_theta_res)
            J_theta_obs = torch.cat(J_theta_obs)
            J_theta_BC = torch.cat(J_theta_BC)
                
        else:
            
            J_theta_res = torch.tensor(0.)
            J_theta_obs = torch.tensor(0.)
            J_theta_BC = torch.tensor(0.)
            
        J_theta_res = torch.norm(J_theta_res, p = 2)
        J_theta_obs = torch.norm(J_theta_obs, p = 2)
        J_theta_BC = torch.norm(J_theta_BC, p = 2)
            
        self.list_J_gradients.append([J_k_res.item(), J_theta_res.item(),
                                      J_theta_obs.item(), J_theta_BC.item()])
        
        self.J_train = self.J_train.detach().clone()
        self.J_res_train = self.J_res_train.detach().clone()
        self.J_obs_train = self.J_obs_train.detach().clone()
        self.J_BC_train = self.J_BC_train.detach().clone()
        
        self.list_LBFGS_n_iter.append(optimizer.state_dict()['state'][0]['n_iter'])
        
        if self.end_training:
            self.list_iter_flag.pop()
            self.end_training = False
        
        if (len(self.list_LBFGS_n_iter) > 1):
            if (self.list_LBFGS_n_iter[-1] == self.list_LBFGS_n_iter[-2]):
                self.list_iter_flag.append(False)
            else:
                self.list_iter_flag.append(True)
                self.iter += 1
        else:
            self.iter += 1
            
        if (self.lambdas['res'] == 0):
            self.list_res_optim.append(False)
        else:
            self.list_res_optim.append(True)

    def display_training(self, col, obs, ref_solution, start_time, display_freq):
        """
        Method used to display values of interest during the training.
        """
        
        print('#'*50)
        
        print('Processing iteration {:.0f} (iter + eval = {:.0f})'.format(self.iter, 
                                                                          self.iter_eval))
                    
        print('-'*25)
        print('J           = {:.2e} (residual : {:.2e}, BC : {:.2e}, obs : {:.2e})'.format(self.list_J_train[-1][0],
                                                                                               self.list_J_train[-1][1],
                                                                                               self.list_J_train[-1][2],
                                                                                               self.list_J_train[-1][3]))
        print('||grad(J)|| = {:.2e}'.format(self.list_grad[-1]))
        
        print('parameter      = {}'.format(self.k_ref*self.k.detach().clone()))
            
        print('time           = {:.2f} s'.format(time.time() - start_time))
        
        if self.device.type == 'cuda':
            print('-'*25)
            print('GPU :', torch.cuda.get_device_name(0))
            print('Total memory         :', 
                  round(torch.cuda.get_device_properties(0).total_memory/1024**3, 1), 'GB')
            print('Max Allocated memory :', 
                  round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')
            print('Allocated memory     :', 
                  round(torch.cuda.memory_allocated(0)/1024**3, 1), 'GB')
            print('Reserved memory        :', 
                  round(torch.cuda.memory_reserved(0)/1024**3, 1), 'GB')
        
        if (self.iter%display_freq[1] == 0):
            if not(self.list_iter_flag[-1]):
                print("/!\ Last gradient descent step was just an evaluation without iteration, results might look weird but don't worry.")
            display.display_results(self, col, ref_solution, obs)
