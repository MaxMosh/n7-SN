import torch

class CollocationPoints:
    """
    CollocationPoints class, used to define the collocation points sampled in the physical domain. 
    """
    
    def __init__(self, device, random, N_col, variable_boundaries, 
                 test_size = 0, seed = None):
        
        # Seed for random collocation points and training/testing sets reproductibility
        if (seed != None):
            torch.manual_seed(seed)
        
        self.device = device
        self.random = random
        self.test_size = test_size
        
        self.N_col = N_col
        self.variable_min = variable_boundaries[0]
        self.variable_max = variable_boundaries[1]
            
        self.N_col_train = int(self.N_col*(1-test_size))
        self.N_col_test = int(self.N_col*test_size)
        
        self.grid = torch.linspace(self.variable_min, 
                                   self.variable_max, 
                                   N_col).view(-1, 1)
            
        if random:
            
            self.all = (self.variable_max-self.variable_min)*torch.rand(N_col, 1) + self.variable_min*torch.ones(N_col, 1)
            
            self.all[0] = self.variable_min
            self.all[-1] = self.variable_max
            
        else:
        
            self.all = self.grid.detach().clone()
        
        self.all.requires_grad = True
        
        self.grid_shuffled = self.grid[torch.randperm(self.N_col)]
        
        self.train = self.grid_shuffled[:int(self.N_col*(1-self.test_size))]
        self.train.requires_grad = True
        
        if (self.test_size > 0):
            self.test = self.grid_shuffled[-int(self.N_col*self.test_size):]
            self.test.requires_grad = True    